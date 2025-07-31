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

def compute_jaccard(Y: torch.Tensor) -> torch.Tensor:
    inter = (Y.unsqueeze(2) * Y.unsqueeze(1)).sum(0).float()
    union = ((Y.unsqueeze(2) + Y.unsqueeze(1)) >= 1).sum(0).float()
    return inter / (union + 1e-8)                   # (C, C)

def contrastive_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    batch_size = z_i.size(0)
    z_i = nn.functional.normalize(z_i, dim=1)
    z_j = nn.functional.normalize(z_j, dim=1)
    z = torch.cat([z_i, z_j], dim=0)               # (2B, D)
    sim = torch.matmul(z, z.T) / temperature       # (2B,2B)
    mask = torch.eye(2*batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    pos = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)])
    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1)
    loss = -torch.log(torch.exp(pos) / denom)
    return loss.mean()

def prototype_contrastive_loss(prototypes: torch.Tensor, W_proto: torch.Tensor) -> torch.Tensor:
    P, D = prototypes.shape
    p_norm = nn.functional.normalize(prototypes, dim=1)        # (P, D)
    sim = p_norm @ p_norm.T                        # (P, P)
    sim = sim.masked_fill(torch.eye(P, device=sim.device).bool(), 0.0)
    pos_mask = W_proto
    neg_mask = 1.0 - W_proto
    pos_mean = (pos_mask * sim).sum() / pos_mask.sum().clamp(min=1.0)
    neg_mean = (neg_mask * sim).sum() / neg_mask.sum().clamp(min=1.0)
    return neg_mean - pos_mean
