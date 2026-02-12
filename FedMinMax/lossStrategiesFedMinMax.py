import torch
import torch.nn as nn
import numpy as np

# --- Helper: Euclidean Projection onto Simplex ---
def project_simplex(v, eps=0.0):
    """
    Project v onto the simplex {x: sum x = 1, x_i >= eps}.
    If eps=0.0, this reduces to the standard probability simplex projection.
    """
    v = np.asarray(v, dtype=float)
    d = v.size

    if eps < 0:
        raise ValueError("eps must be >= 0")
    if d * eps > 1.0 + 1e-12:
        raise ValueError(f"Infeasible: d*eps={d*eps} > 1. Choose smaller eps.")

    # Shift by eps so we can project onto the standard simplex of mass (1 - d*eps)
    # Let x = eps + y, with y_i >= 0 and sum y_i = 1 - d*eps
    mass = 1.0 - d * eps
    u = v - eps

    # If mass == 0, the only feasible point is all-eps
    if mass <= 1e-15:
        return np.full(d, eps, dtype=float)

    # Standard simplex projection for y: sum y = mass, y >= 0
    # Implementation: sort, find threshold, clamp.
    s = np.sort(u)[::-1]
    cssv = np.cumsum(s) - mass
    ind = np.arange(1, d + 1)
    cond = s - cssv / ind > 0
    if not np.any(cond):
        # Fallback (should be rare): all mass goes to the max coordinate
        y = np.zeros(d, dtype=float)
        y[np.argmax(u)] = mass
    else:
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
        y = np.maximum(u - theta, 0.0)

    x = y + eps

    # Numerical cleanup to enforce constraints tightly
    x = np.maximum(x, eps)
    x /= x.sum()

    return torch.tensor(x, device="cpu")

# --- Client Strategy: FedMinMax Loss ---
def loss_fedminmax(outputs, targets, sensitive_attrs, context):
    """
    Calculates weighted loss based on global importance weights (w).
    Eq (8): r_k(theta, w) = sum( (n_ak/n_k) * w_a * r_ak(theta) )[cite: 222].
    """
    weights_dict = context.get('group_weights', {})
    criterion_elementwise = nn.BCELoss(reduction='none')

    # Ensure integer group ids for masking
    s = sensitive_attrs.view(-1).long()
    y = targets.view(-1)
    o = outputs.view(-1)

    loss_per_sample = criterion_elementwise(o, y)
    sample_weights = torch.zeros_like(loss_per_sample)

    for gid, weight in weights_dict.items():
        gid_int = int(gid)
        mask = (s == gid_int)
        if mask.any():
            sample_weights[mask] = float(weight)

    return (loss_per_sample * sample_weights).sum() / len(targets)

# --- Server Strategy: FedMinMax Aggregation ---
def agg_fedminmax(client_reports, global_model, device, server_state):
    """
    1. Aggregates model weights (Standard FedAvg).
    2. Updates adversarial weights mu based on reported group risks.
    """
    # 1) FedAvg aggregation
    avg_weights = {}
    total_samples = sum(r['samples'] for r in client_reports)

    for r in client_reports:
        scale = r['samples'] / total_samples
        for k, v in r['weights'].items():
            v = v.to(device)  # optional: keep on device
            if k in avg_weights:
                avg_weights[k] += v * scale
            else:
                avg_weights[k] = v * scale

    global_model.load_state_dict(avg_weights)

    # 2) Mu update (projected gradient ascent)
    mu = server_state['mu']
    lr_mu = server_state['lr_mu']
    global_group_counts = server_state['group_counts']
    group_ids = server_state['group_ids']  # IMPORTANT ordering

    risk_vector = torch.zeros(len(group_ids), device=device, dtype=torch.float32)

    

    for i, gid in enumerate(group_ids):
        numerator = 0.0
        for r in client_reports:
            c_risks = r.get('group_risks', {})
            c_counts = r.get('group_counts', {})
            if gid in c_risks and gid in c_counts and c_counts[gid] > 0:
                numerator += float(c_risks[gid]) * int(c_counts[gid])

        denom = int(global_group_counts.get(gid, 0))
        if denom > 0:
            risk_vector[i] = numerator / denom

    mu_new = project_simplex(mu + lr_mu * risk_vector, eps=1e-3)
    server_state['mu'] = mu_new

    # DEBUG: attach for logging (optional)
    server_state['debug_risk_vector'] = risk_vector.detach().clone()

    return avg_weights