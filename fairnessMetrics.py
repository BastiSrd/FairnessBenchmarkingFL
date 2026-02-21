import torch

def get_rates(preds, labels, sensitive, s_value):
    """
    Helper to calculate TPR, FPR, and PPR.
    Args:
        preds (torch.Tensor): Binary predictions (0 or 1) of shape (N,).
        labels (torch.Tensor): Ground truth labels (0 or 1) of shape (N,). 
                                Can be None if only PPR is needed.
        sensitive (torch.Tensor): Sensitive attribute values of shape (N,).
        s_value (int or float): The specific value in 'sensitive' that defines 
                                the group to calculate metrics for (e.g., 0).

    Returns:
        tuple: A tuple containing (TPR, FPR, PPR) as floats. 
                Returns (0.0, 0.0, 0.0) if the group is empty.
    """
    s_value = int(s_value)
    mask = (sensitive == s_value)
    
    if mask.sum() == 0:

        print("mask.sum is zero therefore TPR, FPR, PPR all 0")
        return 0.0, 0.0, 0.0

    # Filter predictions
    g_preds = preds[mask]

    # --- Positive Prediction Rate (PPR) ---
    ppr = g_preds.float().mean().item()

    if labels is None:
        return 0.0, 0.0, ppr

    # Filter labels
    g_labels = labels[mask]

    # --- True Positive Rate (TPR) ---
    mask_pos = (g_labels == 1)
    if mask_pos.sum() > 0:
        tpr = g_preds[mask_pos].float().mean().item()
    else:
        print("TPR is zero, the Equalized Odds is artificially affected")
        tpr = 0.0

    # --- False Positive Rate (FPR) ---
    mask_neg = (g_labels == 0)
    if mask_neg.sum() > 0:
        fpr = g_preds[mask_neg].float().mean().item()
    else:
        print("FPR is zero, the Equalized Odds is artificially affected")
        fpr = 0.0

    return tpr, fpr, ppr

def compute_statistical_parity(preds, sensitive):
    """
    Calculates Statistical Parity: P(Pred=1 | Non-Prot) - P(Pred=1 | Prot)

    Args:
        preds (torch.Tensor): Binary predictions (0 or 1) of shape (N,).
        sensitive (torch.Tensor): Sensitive attribute values of shape (N,). 
                                    Must contain exactly two unique values 
                                    (e.g., 0 and 1) to calculate a difference.

    Returns:
        float: The signed difference in acceptance rates. Positive values 
                indicate Group B is accepted more often than Group A.
                Returns 0.0 if fewer than 2 groups are found.
    """
    groups = torch.unique(sensitive).sort()[0]
    if len(groups) < 2:
        return 0.0
    
    group_a = groups[0].item()
    group_b = groups[1].item()
    
    _, _, ppr_a = get_rates(preds, None, sensitive, group_a)
    _, _, ppr_b = get_rates(preds, None, sensitive, group_b)
    
    return ppr_b - ppr_a

def compute_equalized_odds(preds, labels, sensitive):
    """
    Calculates the Equalized Odds score, defined as the average absolute 
    difference in True Positive Rates (TPR) and False Positive Rates (FPR) 
    between two groups.

    Formula: 0.5 * (|TPR_A - TPR_B| + |FPR_A - FPR_B|)

    Args:
        preds (torch.Tensor): Binary predictions (0 or 1) of shape (N,).
        labels (torch.Tensor): Ground truth labels (0 or 1) of shape (N,).
        sensitive (torch.Tensor): Sensitive attribute values of shape (N,). 
                                    Must contain exactly two unique values.

    Returns:
        float: The Equalized Odds gap.
                Returns 0.0 if fewer than 2 groups are found.
    """
    groups = torch.unique(sensitive).sort()[0]
    
    if len(groups) < 2:
        return 0.0
        
    group_a = groups[0].item()
    group_b = groups[1].item()
    
    # Get rates for both groups
    tpr_a, fpr_a, _ = get_rates(preds, labels, sensitive, group_a)
    tpr_b, fpr_b, _ = get_rates(preds, labels, sensitive, group_b)

    # Calculate EO
    tpr_diff = abs(tpr_b - tpr_a)
    fpr_diff = abs(fpr_b - fpr_a)

    return 0.5 * (tpr_diff + fpr_diff)

def compute_balanced_accuracy(preds, labels):
    """
    Calculates Balanced Accuracy, defined as the average of TPR and TNR.

    Formula: 0.5 * (TPR + TNR)

    Args:
        preds (torch.Tensor): Binary predictions (0 or 1) of shape (N,).
        labels (torch.Tensor): Ground truth labels (0 or 1) of shape (N,).

    Returns:
        float: The Balanced Accuracy score.
    """
    # True Positives and Negatives
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    
    # Positives and Negatives
    pos = (labels == 1).sum().item()
    neg = (labels == 0).sum().item()

    # Calculate TPR and TNR
    tpr = tp / pos if pos > 0 else 0.0
    tnr = tn / neg if neg > 0 else 0.0

    return 0.5 * (tpr + tnr)
