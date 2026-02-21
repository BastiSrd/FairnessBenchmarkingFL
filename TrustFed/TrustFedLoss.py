import torch

from TrustFed.LossStrategies.loss import DemographicParityLoss, EqualiedOddsLoss


def _prob_to_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert predicted probabilities to logits.

    Parameters:
        p (torch.Tensor): Model outputs after sigmoid, shape (B, 1) or (B,).
        eps (float): Small value for numerical stability (clamps p to (eps, 1-eps)).

    Returns:
        torch.Tensor: Logits corresponding to the input probabilities.
    """
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def loss_trustfed(preds, targets, sensitive, context):
    """
      Compute the TrustFed training loss (classification + fairness constraint).

      Parameters:
          preds (torch.Tensor): Predicted probabilities from the model (after sigmoid),
                                shape (B, 1) or (B,).
          targets (torch.Tensor): Ground truth labels (0/1), shape (B, 1) or (B,).
          sensitive (torch.Tensor): Sensitive attribute values per sample, shape (B,).
          context (dict): Contains required configuration:
              - criterion: Base loss (e.g., BCELoss with reduction="none").
              - X_batch: Feature tensor of the current batch (needed for constraints).
              - fairness_notion: "SP" (Demographic Parity) or "EO" (Equalized Odds).
              - alpha (float): Fairness constraint weight (handled inside constraint).
              - p_norm (int): Norm used inside the constraint.
              - sensitive_classes (list): Sensitive group values (default [0,1]).
              - cost_false_negatives (float, optional): Weight for positive samples.
              - cost_false_positives (float, optional): Weight for negative samples.

      Returns:
          total_loss (torch.Tensor): Combined classification + fairness loss.
          fairness_loss (torch.Tensor): Fairness constraint term only.
      """

    criterion = context["criterion"]
    X_batch = context["X_batch"]

    fairness_notion = context.get("fairness_notion", "SP")
    alpha = float(context.get("alpha", 1.0))
    p_norm = int(context.get("p_norm", 2))
    sensitive_classes = context.get("sensitive_classes", [0, 1])

    # Base classification loss (cost-sensitive, per-sample)
    eps = 1e-6
    preds = torch.clamp(preds, eps, 1.0 - eps)

    # defaults
    cost_false_negatives = float(context.get("cost_false_negatives", 10.0))
    cost_false_positives = float(context.get("cost_false_positives", 5.0))

    # BCELoss(reduction="none") -> shape (B,1)
    per_sample_loss = criterion(preds, targets)

    weights = torch.where(
        targets >= 0.5,
        torch.tensor(cost_false_negatives, device=targets.device, dtype=targets.dtype),
        torch.tensor(cost_false_positives, device=targets.device, dtype=targets.dtype),
    )

    # apply weights + mean
    base_loss = (per_sample_loss * weights).mean()

    out_logits = _prob_to_logit(preds)

    if fairness_notion == "SP":
        constraint = DemographicParityLoss(
            sensitive_classes=sensitive_classes,
            alpha=alpha,
            p_norm=p_norm
        )
        fairness_loss = constraint(X_batch, out_logits, sensitive)

    elif fairness_notion == "EO":
        constraint = EqualiedOddsLoss(
            sensitive_classes=sensitive_classes,
            alpha=alpha,
            p_norm=p_norm
        )
        fairness_loss = constraint(X_batch, out_logits, sensitive, y=targets)

    else:
        raise ValueError("fairness_notion must be 'SP' or 'EO'")

    total_loss = base_loss + fairness_loss
    if not torch.isfinite(total_loss):
        total_loss = base_loss  # fallback: nur Klassifikationsloss, falls NAN

    return total_loss, fairness_loss


def agg_trustfed(reports, global_model, device):
    """
    Perform weighted FedAvg aggregation of client model updates.

    Parameters:
        reports (list of dict): Each report must contain:
            - "weights": state_dict of the client model.
            - "samples": Number of local training samples.
        global_model (torch.nn.Module): Global model to match parameter structure.
        device (torch.device): Device to place aggregated weights on.

    Returns:
        dict: Aggregated state_dict with weighted average parameters.
    """

    total_samples = sum(r["samples"] for r in reports)
    new_weights = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}

    for r in reports:
        w = r["samples"] / total_samples
        for k, v in r["weights"].items():
            new_weights[k] += w * v.to(device)

    return new_weights
