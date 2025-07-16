import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

def get_optimizer_scheduler(model: torch.nn.Module,
                            lr: float = 1e-4,
                            weight_decay: float = 1e-5,
                            scheduler_step_size: int = 10,
                            scheduler_gamma: float = 0.1):
    """
    Initialize AdamW optimizer and StepLR scheduler.

    Args:
        model: nn.Module
        lr: learning rate
        weight_decay: weight decay
        scheduler_step_size: step size in epochs for scheduler
        scheduler_gamma: multiplicative factor of learning rate decay

    Returns:
        optimizer, scheduler
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=scheduler_step_size,
                                                gamma=scheduler_gamma)
    return optimizer, scheduler


def calculate_metrics(preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5):
    """
    Compute micro/macro F1 and mean AUROC for multi-label predictions.

    Args:
        preds: (N, K) float array of probabilities
        targets: (N, K) binary array
        threshold: float threshold to binarize preds

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

    Args:
        model
        dataloader
        optimizer
        criterion_fn: function taking preds, targets, return loss
        device

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
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def evaluate(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             criterion_fn,
             device: torch.device,
             threshold: float = 0.5):
    """
    Evaluate model on validation/test set.

    Args:
        model
        dataloader
        criterion_fn
        device
        threshold

    Returns:
        avg_loss, metrics dict
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)
            preds = outputs['logits']
            loss = criterion_fn(preds, labels)
            total_loss += loss.item() * signals.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    metrics = calculate_metrics(all_preds, all_targets, threshold)
    return avg_loss, metrics
