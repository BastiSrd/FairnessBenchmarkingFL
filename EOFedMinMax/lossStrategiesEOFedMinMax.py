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


#Client Strategy: Utility + EO
def loss_EOfedminmax(outputs, targets, sensitive_attrs, context, device):
    """
    Utility + Equalized Odds penalty (soft version).
    - Utility: BCE(outputs, targets)
    - EO penalty: weighted soft-FPR and soft-FNR per group with server-sent weights
    lambda_eo is provided by the server and is adaptive (not fixed).
    """
    # From server
    w_y0 = context.get("group_weights_y0", {})  
    w_y1 = context.get("group_weights_y1", {})  
    lambda_eo = float(context.get("lambda_eo", 0.0))

    # From client injection
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # Flatten
    s = sensitive_attrs.view(-1).long()
    y = targets.view(-1).float()
    logits = outputs.view(-1).float()      
    probs  = torch.sigmoid(logits)         


    utility_loss = criterion(logits, y)
    
    # If lambda is 0, avoid extra compute
    if lambda_eo <= 0.0:
        return utility_loss

    fairness_fpr = torch.tensor(0.0).to(device)
    fairness_fnr = torch.tensor(0.0).to(device)
    sum_w0 = 0.0
    sum_w1 = 0.0

    # Use union of keys so missing groups don't crash training
    group_keys = set([int(k) for k in w_y0.keys()]) | set([int(k) for k in w_y1.keys()])
    if len(group_keys) == 0:
        return utility_loss  # nothing to weight

    for gid in group_keys:
        gid_int = int(gid)

        mask_g = (s == gid_int)
        if not mask_g.any():
            continue

        # y==0 slice -> FPR proxy
        mask0 = mask_g & (y < 0.5)
        if mask0.any():
            w0 = float(w_y0.get(gid_int, w_y0.get(str(gid_int), 0.0)))
            if w0 != 0.0:
                fairness_fpr = fairness_fpr + w0 * probs[mask0].mean()
                sum_w0 += abs(w0)

        # y==1 slice -> FNR proxy
        mask1 = mask_g & (y >= 0.5)
        if mask1.any():
            w1 = float(w_y1.get(gid_int, w_y1.get(str(gid_int), 0.0)))
            if w1 != 0.0:
                fairness_fnr = fairness_fnr + w1 * (1.0 - probs[mask1].mean())
                sum_w1 += abs(w1)

    

    fairness_loss = fairness_fpr + fairness_fnr
   

    return utility_loss + lambda_eo * (fairness_loss)


#Server Strategy: FedAvg + EO adversary update
def agg_EOfedminmax(client_reports, global_model, device, server_state):
   
    
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

    mu_y0 = server_state["mu_y0"]
    mu_y1 = server_state["mu_y1"]
    lr_mu = float(server_state["lr_mu"])
    group_ids = server_state["group_ids"]

    global_count_y0 = server_state["group_counts_y0"]  
    global_count_y1 = server_state["group_counts_y1"] 

    # risk vectors: FPR (y=0) and FNR (y=1)
    fpr_vec = torch.zeros(len(group_ids), device=device, dtype=torch.float32)
    fnr_vec = torch.zeros(len(group_ids), device=device, dtype=torch.float32)

    for i, gid in enumerate(group_ids):
        gid_int = int(gid)

        # aggregate FPR proxy
        num0 = 0.0
        for r in client_reports:
            c_fpr = r.get("group_fpr", {})
            c_n0 = r.get("count_y0", {})
            if gid_int in c_fpr and gid_int in c_n0 and int(c_n0[gid_int]) > 0:
                num0 += float(c_fpr[gid_int]) * int(c_n0[gid_int])
        den0 = int(global_count_y0.get(gid_int, 0))
        if den0 > 0:
            fpr_vec[i] = num0 / den0

        # aggregate FNR proxy
        num1 = 0.0
        for r in client_reports:
            c_fnr = r.get("group_fnr", {})
            c_n1 = r.get("count_y1", {})
            if gid_int in c_fnr and gid_int in c_n1 and int(c_n1[gid_int]) > 0:
                num1 += float(c_fnr[gid_int]) * int(c_n1[gid_int])
        den1 = int(global_count_y1.get(gid_int, 0))
        if den1 > 0:
            fnr_vec[i] = num1 / den1

    # projected gradient ascent (simplex)
    server_state["mu_y0"] = project_simplex(mu_y0 + lr_mu * fpr_vec, eps=1e-3, device=device)
    server_state["mu_y1"] = project_simplex(mu_y1 + lr_mu * fnr_vec, eps=1e-3, device=device)

    return avg_weights