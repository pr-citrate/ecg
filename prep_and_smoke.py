import os
import ast
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import ECGDataset
from models.hybrid import HybridECGModel


def prepare_dataloaders(
    meta_csv: str,
    data_dir: str,
    batch_size: int = 32,
    split_fold: int = 0,
    n_splits: int = 5,
    random_state: int = 42
):
    """
    Prepare stratified train/validation DataLoaders with weighted sampling.

    Args:
        meta_csv: Path to metadata CSV with 'labels' and optional 'strat_fold'.
        data_dir: Root directory containing ECG WFDB files.
        batch_size: Batch size for DataLoaders.
        split_fold: Which fold to use as validation.
        n_splits: Number of folds for StratifiedGroupKFold if 'strat_fold' not present.
        random_state: Random seed.

    Returns:
        train_loader, val_loader
    """
    df = pd.read_csv(meta_csv)
    df['labels'] = df['labels'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Determine train/val indices
    if 'strat_fold' in df.columns:
        train_idx = df['strat_fold'] != split_fold
        val_idx = df['strat_fold'] == split_fold
    else:
        labels = np.array(df['labels'].tolist())
        # stratify by first positive label or majority class
        y = labels.argmax(axis=1)
        groups = df['patient_id'].values if 'patient_id' in df.columns else None
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for i, (train_i, val_i) in enumerate(sgkf.split(df, y, groups)):
            if i == split_fold:
                train_idx = np.zeros(len(df), dtype=bool)
                val_idx = np.zeros(len(df), dtype=bool)
                train_idx[train_i] = True
                val_idx[val_i] = True
                break

    df_train = df[train_idx].reset_index(drop=True)
    df_val = df[val_idx].reset_index(drop=True)

    # WeightedRandomSampler for train to handle class imbalance
    train_labels = np.array(df_train['labels'].tolist())  # shape (N_train, K)
    class_counts = train_labels.sum(axis=0)  # per-class counts
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = (train_labels * class_weights).sum(axis=1)

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Save split CSVs (optional)
    train_csv = os.path.join(os.path.dirname(meta_csv), 'train_split.csv')
    val_csv = os.path.join(os.path.dirname(meta_csv), 'val_split.csv')
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)

    # Create datasets and loaders
    train_ds = ECGDataset(train_csv, data_dir, use_lowres=False)
    val_ds = ECGDataset(val_csv, data_dir, use_lowres=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def sanity_check(meta_csv: str, data_dir: str):
    """
    Load a single sample and print its shape and label info.
    """
    ds = ECGDataset(meta_csv, data_dir, use_lowres=False)
    signal, label = ds[0]
    print('=== Sanity Check ===')
    print('Signal shape:', signal.shape)
    print('Number of classes:', len(label))
    print('Positive label indices:', torch.nonzero(label).squeeze().tolist())


def smoke_run(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = torch.device('cpu')
):
    """
    Perform a single optimization step on a small batch to verify train loop.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    # Take one batch
    signals, labels = next(iter(loader))
    signals = signals.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(signals)['logits']  # (batch_size, num_labels)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print('=== Smoke Run ===')
    print('Batch size:', signals.size(0))
    print('Loss:', loss.item())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Data Prep and Smoke Test')
    parser.add_argument('--meta_csv', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--split_fold', type=int, default=0)
    args = parser.parse_args()

    # Sanity check
    sanity_check(args.meta_csv, args.data_dir)

    # Prepare loaders
    train_loader, val_loader = prepare_dataloaders(
        args.meta_csv, args.data_dir,
        batch_size=args.batch_size,
        split_fold=args.split_fold
    )

    # Smoke run on train_loader
    model = HybridECGModel(
        in_channels=12, d_model=64,
        num_prototypes=32, n_heads=4,
        n_layers=2, num_concepts=10,
        num_labels=int(pd.read_csv(args.meta_csv).iloc[0]['labels'] and len(eval(pd.read_csv(args.meta_csv).iloc[0]['labels'])))
    )
    smoke_run(model, train_loader)
