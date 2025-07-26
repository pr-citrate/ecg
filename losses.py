import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    """
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

def classification_loss(preds: torch.Tensor,
                        targets: torch.Tensor,
                        loss_type: str = 'focal',
                        gamma: float = 2.0,
                        pos_weight: torch.Tensor = None) -> torch.Tensor:
    """
    Computes multi-label loss. Supports 'bce' or 'focal'.

    Args:
        preds: Tensor of shape (batch_size, num_labels), values in [0,1]
        targets: Tensor of same shape with binary 0/1 labels
        loss_type: 'bce' or 'focal'
        gamma: focusing parameter for focal loss
        pos_weight: optional tensor of shape (num_labels,) for weighting positives

    Returns:
        Scalar loss tensor
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(preds, targets)
    elif loss_type == 'focal':
        # focal loss implementation
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight,
                                   reduction='none')(preds, targets)
        p_t = torch.exp(-bce)
        loss = ((1 - p_t) ** gamma * bce).mean()
        return loss
    else:
        raise ValueError(f"Unknown loss_type {loss_type}")


def prototype_loss(model: nn.Module, l2_coeff: float = 1e-3) -> torch.Tensor:
    """
    Prototype regularization loss: L2 penalty on learnable prototype vectors.

    Args:
        model: The HybridECGModel instance with attribute "prototype"
        l2_coeff: coefficient for L2 regularization

    Returns:
        Scalar loss tensor
    """
    # Encourage prototypes to remain bounded
    protos = model.prototype.prototypes  # shape (num_prototypes, d_model)
    return l2_coeff * torch.norm(protos, p=2)


def attention_regularization(attn_feats: torch.Tensor, l1_coeff: float = 1e-3) -> torch.Tensor:
    """
    Attention regularization: L1 sparsity penalty on attention-derived features.

    Args:
        attn_feats: Tensor of shape (batch_size, d_model, seq_len)
        l1_coeff: coefficient for L1 regularization

    Returns:
        Scalar loss tensor
    """
    # Encourage sparse/concise attention output
    return l1_coeff * torch.mean(torch.abs(attn_feats))
