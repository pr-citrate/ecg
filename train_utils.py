from typing import Sequence

import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve


def get_optimizer_scheduler(model: torch.nn.Module,
                            lr: float = 1e-4,
                            weight_decay: float = 1e-5,
                            scheduler_type: str = 'step',
                            scheduler_step_size: int = 10,
                            scheduler_gamma: float = 0.1,
                            scheduler_patience: int = 5,
                            scheduler_factor: float = 0.5):
    """
    Initialize AdamW optimizer and a scheduler of choice.

    Args:
        model: nn.Module
        lr: learning rate
        weight_decay: weight decay
        scheduler_type: 'step', 'cosine', or 'plateau'
        scheduler_step_size: for 'step' or T_max for 'cosine'
        scheduler_gamma: gamma for 'step'
        scheduler_patience: patience for ReduceLROnPlateau
        scheduler_factor: factor for ReduceLROnPlateau

    Returns:
        optimizer, scheduler
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_step_size,
            eta_min=0
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience
        )
    else:
        raise ValueError(f"Unknown scheduler_type {scheduler_type}")

    return optimizer, scheduler


def calculate_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float | Sequence[float] = 0.5
) -> dict:
    """
    Compute micro/macro F1 and mean AUROC for multi-label predictions.

    Args:
        preds: (N, K) float array of probabilities
        targets: (N, K) binary array
        threshold: either
            - single float: apply same cutoff for all K labels, or
            - sequence of length K: per-label cutoffs

    Returns:
        dict with keys: 'micro_f1', 'macro_f1', 'mean_auroc'
    """
    # Binarize preds
    binarized = (preds >= threshold).astype(int)
    micro = f1_score(targets, binarized, average='micro', zero_division=0)
    macro = f1_score(targets, binarized, average='macro', zero_division=0)

    # AUROC per label
    aurocs = []
    num_labels = targets.shape[1]
    for i in range(num_labels):
        try:
            score = roc_auc_score(targets[:, i], preds[:, i])
        except ValueError:
            score = np.nan
        aurocs.append(score)
    mean_auroc = np.nanmean(aurocs)

    return {'micro_f1': micro, 'macro_f1': macro, 'mean_auroc': mean_auroc}


def train_one_epoch(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion_fn,
                    device: torch.device):
    """
    Train model for one epoch.

    Returns:
        avg_loss: float
    """
    model.train()
    total_loss = 0.0

    for signals, labels in dataloader:
        signals = signals.to(device)
        labels = labels.to(device)
        outputs = model(signals)
        preds = outputs['logits']
        loss = criterion_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * signals.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion_fn,
    device: torch.device,
    threshold: float | Sequence[float] = 0.5
):
    """
    Evaluate model on validation/test set.

    Args:
        model: the PyTorch model
        dataloader: DataLoader for eval
        criterion_fn: loss function taking (logits, labels)
        device: torch.device
        threshold: float or sequence of length K for binarization

    Returns:
        avg_loss (float),
        metrics (dict),
        probs (np.ndarray of shape (N, K)),
        targets (np.ndarray of shape (N, K))
    """
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)
            logits = outputs['logits']
            loss = criterion_fn(logits, labels)

            total_loss += loss.item() * signals.size(0)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    all_logits = np.vstack(all_logits)
    all_targets = np.vstack(all_targets)

    # convert logits to probabilities
    probs = 1 / (1 + np.exp(-all_logits))

    # compute metrics with possibly per-class thresholds
    metrics = calculate_metrics(probs, all_targets, threshold)

    return avg_loss, metrics, probs, all_targets


def find_optimal_thresholds(probs: np.ndarray, targets: np.ndarray):
    """
    Find per-class threshold that maximizes F1.
    """
    K = targets.shape[1]
    thresholds = np.zeros(K)

    for i in range(K):
        precision, recall, th = precision_recall_curve(targets[:, i], probs[:, i])
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        # drop last threshold as it's nan for recall=0
        thresholds[i] = th[np.nanargmax(f1[:-1])]

    return thresholds
