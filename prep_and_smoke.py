import os
import ast
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import ECGDataset, ECGContrastiveDataset
from models.hybrid import HybridECGModel
from losses import (
    classification_loss,
    contrastive_loss,
    prototype_loss,
    attention_regularization
)


def prepare_dataloaders(
    meta_csv: str,
    data_dir: str,
    batch_size: int = 32,
    split_fold: int = 0,
    n_splits: int = 5,
    random_state: int = 42,
    contrastive: bool = False
):
    """
    Prepare stratified train/validation DataLoaders.
    If contrastive=True, uses ECGContrastiveDataset for training.
    """
    df = pd.read_csv(meta_csv)
    df['labels'] = df['labels'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Determine train/val indices
    if 'strat_fold' in df.columns:
        train_idx = df['strat_fold'] != split_fold
        val_idx   = df['strat_fold'] == split_fold
    else:
        labels = np.array(df['labels'].tolist())
        y      = labels.argmax(axis=1)
        groups = df['patient_id'].values if 'patient_id' in df.columns else None
        sgkf   = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for i, (train_i, val_i) in enumerate(sgkf.split(df, y, groups)):
            if i == split_fold:
                train_idx = np.zeros(len(df), dtype=bool)
                val_idx   = np.zeros(len(df), dtype=bool)
                train_idx[train_i] = True
                val_idx[val_i]     = True
                break

    df_train = df[train_idx].reset_index(drop=True)
    df_val   = df[val_idx].reset_index(drop=True)

    # WeightedRandomSampler for train to handle class imbalance
    train_labels  = np.array(df_train['labels'].tolist())
    class_counts  = train_labels.sum(axis=0)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights= (train_labels * class_weights).sum(axis=1)
    sampler       = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Save split CSVs
    base_dir = os.path.dirname(meta_csv)
    train_csv = os.path.join(base_dir, 'train_split.csv')
    val_csv   = os.path.join(base_dir, 'val_split.csv')
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)

    # Create datasets and loaders
    if contrastive:
        train_ds = ECGContrastiveDataset(train_csv, data_dir, use_lowres=False)
    else:
        train_ds = ECGDataset(train_csv, data_dir, use_lowres=False)
    val_ds   = ECGDataset(val_csv, data_dir, use_lowres=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

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
    device: torch.device,
    args
):
    """
    Perform a single optimization step on a small batch to verify train loop,
    using contrastive loss if args.contrastive=True.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Take one batch
    batch = next(iter(loader))

    if args.contrastive:
        # Expect (view1, view2, label)
        v1, v2, labels = batch
        v1, v2, labels = v1.to(device), v2.to(device), labels.to(device)

        out1 = model(v1)
        out2 = model(v2)

        # classification loss (binary cross-entropy)
        clf_loss = classification_loss(
            out1['logits'], labels,
            loss_type='bce'
        )

        # contrastive InfoNCE loss on concept_scores
        con_loss = contrastive_loss(
            out1['concept_scores'],
            out2['concept_scores'],
            temperature=args.temperature
        )

        loss = clf_loss + args.alpha * con_loss

        print('=== Smoke Run (Contrastive) ===')
        print('Batch size:', v1.size(0))
        print(f'Classification Loss: {clf_loss.item():.4f}')
        print(f'Contrastive Loss:    {con_loss.item():.4f}')
    else:
        # Expect (signal, label)
        signals, labels = batch
        signals, labels = signals.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(signals)['logits']
        loss    = classification_loss(outputs, labels, loss_type='bce')

        print('=== Smoke Run (Standard) ===')
        print('Batch size:', signals.size(0))
        print('Loss:', loss.item())

    # backward & step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Data Prep and Smoke Test')
    parser.add_argument('--meta_csv', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--split_fold', type=int, default=0)

    # Contrastive options
    parser.add_argument('--contrastive', action='store_true',
                        help='use InfoNCE contrastive loss in smoke run')
    parser.add_argument('--alpha',       type=float, default=1.0,
                        help='weight for InfoNCE loss')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='temperature for InfoNCE loss')

    parser.add_argument('--device', default='cpu',
                        help='torch device')

    args = parser.parse_args()

    # Sanity check
    sanity_check(args.meta_csv, args.data_dir)

    # Prepare loaders
    train_loader, val_loader = prepare_dataloaders(
        args.meta_csv,
        args.data_dir,
        batch_size=args.batch_size,
        split_fold=args.split_fold,
        contrastive=args.contrastive
    )

    # Smoke run on train_loader
    # determine num_labels from CSV
    sample_label = pd.read_csv(args.meta_csv)['labels'].iloc[0]
    num_labels = len(ast.literal_eval(sample_label)) if isinstance(sample_label, str) else len(sample_label)
    model = HybridECGModel(
        in_channels=12,
        d_model=64,
        num_prototypes=32,
        n_heads=4,
        n_layers=2,
        num_concepts=10,
        num_labels=num_labels
    )
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    smoke_run(model, train_loader, device, args)
