import torch

def accuracy(p, y):
    return (torch.sigmoid(p).round().flatten() == y).float().mean()

def P1(p, a):
    return ((torch.sigmoid(p[a == 0]).round().flatten() == 1).float().mean() - (torch.sigmoid(p[a == 1]).round().flatten() == 1).float().mean()).abs()

def Eodd(p, a, y):
    # True Positive Rate difference (Y=1)
    # Mask for Y=1
    mask_y1 = (y == 1)
    tpr_0 = (torch.sigmoid(p[(a == 0) & mask_y1]).round().flatten() == 1).float().mean()
    tpr_1 = (torch.sigmoid(p[(a == 1) & mask_y1]).round().flatten() == 1).float().mean()
    
    # False Positive Rate difference (Y=0)
    # Mask for Y=0
    mask_y0 = (y == 0)
    fpr_0 = (torch.sigmoid(p[(a == 0) & mask_y0]).round().flatten() == 1).float().mean()
    fpr_1 = (torch.sigmoid(p[(a == 1) & mask_y0]).round().flatten() == 1).float().mean()

    # Handle NaNs if a group is empty
    if torch.isnan(tpr_0): tpr_0 = 0.0
    if torch.isnan(tpr_1): tpr_1 = 0.0
    if torch.isnan(fpr_0): fpr_0 = 0.0
    if torch.isnan(fpr_1): fpr_1 = 0.0

    return 0.5 * (abs(tpr_0 - tpr_1) + abs(fpr_0 - fpr_1))