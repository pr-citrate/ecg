import torch
import torch.nn as nn

# Classification loss: binary cross-entropy for multi-label
bce_loss = nn.BCELoss()

def classification_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes multi-label classification loss using binary cross-entropy.

    Args:
        preds: Tensor of shape (batch_size, num_labels), values in [0,1]
        targets: Tensor of same shape with binary 0/1 labels

    Returns:
        Scalar loss tensor
    """
    return bce_loss(preds, targets)


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
