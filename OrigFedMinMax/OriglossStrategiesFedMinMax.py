import torch
import torch.nn as nn
import numpy as np

def project_simplex(v, eps=0.0, device="cpu"):
    """
    Project v onto the simplex {x: sum x = 1, x_i >= eps}.
    If eps=0.0, this reduces to the standard probability simplex projection.
    """
    # Ensure v is a float tensor on the correct device
    if not torch.is_tensor(v):
        v = torch.tensor(v, dtype=torch.float32, device=device)
    else:
        v = v.to(device).float()
    d = v.shape[0]

    if eps < 0:
        raise ValueError("eps must be >= 0")
    if d * eps > 1.0 + 1e-12:
        raise ValueError(f"Infeasible: d*eps={d*eps} > 1. Choose smaller eps.")

    # Shift by eps so we can project onto the standard simplex of mass (1 - d*eps)
    mass = 1.0 - d * eps
    u = v - eps

    # If mass == 0, the only feasible point is all-eps
    if mass <= 1e-15:
        return torch.full((d,), eps, device=device, dtype=torch.float32)

    # Standard simplex projection for y: sum y = mass, y >= 0
    s, _ = torch.sort(u, descending=True)
    cssv = torch.cumsum(s, dim=0) - mass
    ind = torch.arange(1, d + 1, device=device).float()
    cond = s - cssv / ind > 0
    indices = torch.nonzero(cond)
    if indices.numel() == 0:
        # Fallback: all mass to the max coordinate
        y = torch.zeros(d, device=device)
        y[torch.argmax(u)] = mass
    else:
        rho_idx = indices[-1].item()
        theta = cssv[rho_idx] / (rho_idx + 1)
        y = torch.clamp(u - theta, min=0.0)

    x = y + eps

    # Numerical cleanup to enforce constraints tightly
    x = torch.clamp(x, min=eps)
    x /= x.sum()

    return x.detach().clone().to(device)

#Client Strategy: FedMinMax Loss
def loss_Origfedminmax(outputs, targets, sensitive_attrs, context):
    """
    Calculates weighted loss based on global importance weights (w).
    """
    weights_dict = context.get('group_weights', {})
    criterion_elementwise = nn.BCEWithLogitsLoss(reduction='none')

    # Ensure integer group ids for masking
    s = sensitive_attrs.view(-1).long()
    y = targets.view(-1)
    o = outputs.view(-1)

    loss_per_sample = criterion_elementwise(o, y)
    sample_weights = torch.ones_like(loss_per_sample)

    for gid, weight in weights_dict.items():
        gid_int = int(gid)
        mask = (s == gid_int)
        if mask.any():
            sample_weights[mask] = float(weight)
    weighted = loss_per_sample * sample_weights
    return weighted.mean()

#Server Strategy: FedMinMax Aggregation
def agg_Origfedminmax(client_reports, global_model, device, server_state):
    """
    1. Aggregates model weights (Standard FedAvg).
    2. Updates adversarial weights mu based on reported group risks.
    """
    #FedAvg aggregation
    avg_weights = {}
    total_samples = sum(r['samples'] for r in client_reports)

    for r in client_reports:
        scale = r['samples'] / total_samples
        for k, v in r['weights'].items():
            v = v.to(device)
            if k in avg_weights:
                avg_weights[k] += v * scale
            else:
                avg_weights[k] = v * scale

    global_model.load_state_dict(avg_weights)

    #Mu update (projected gradient ascent)
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

    mu_new = project_simplex(mu + lr_mu * risk_vector, eps=1e-3, device=device)
    server_state['mu'] = mu_new


    return avg_weights