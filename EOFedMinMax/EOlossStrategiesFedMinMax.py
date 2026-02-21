import torch
import torch.nn as nn

# --- Helper: Euclidean Projection onto Simplex ---
def project_simplex(v, z=1):
    """
    Projects vector v onto the simplex (sum(v) = z, v >= 0).
    """
    n_features = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - z
    ind = torch.arange(n_features, device=v.device) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho.float()
    w = torch.clamp(v - theta, min=0)
    return w

# --- Client Strategy: FedMinMax Loss ---
def loss_EOfedminmax(outputs, targets, sensitive_attrs, context):
    """
    Calculates weighted loss based on global importance weights (w).
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
def agg_EOfedminmax(client_reports, global_model, device, server_state):
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

    mu_new = project_simplex(mu + lr_mu * risk_vector)
    server_state['mu'] = mu_new

    # DEBUG: attach for logging (optional)
    server_state['debug_risk_vector'] = risk_vector.detach().clone()

    return avg_weights