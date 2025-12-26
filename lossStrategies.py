import torch

def loss_standard(preds, targets, sensitive, context):
    """Standard Cross Entropy"""
    return context['criterion'](preds, targets)

def agg_fedavg(reports, global_model, context):
    """Standard Weighted Average by sample size"""
    total_samples = sum([r['samples'] for r in reports])
    new_weights = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
    
    for r in reports:
        w = r['samples'] / total_samples
        for k, v in r['weights'].items():
            new_weights[k] += w * v.to(context['device'])
            
    return new_weights