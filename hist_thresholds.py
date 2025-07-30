#!/usr/bin/env python
import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data import ECGDataset
from models.hybrid import HybridECGModel
from losses import classification_loss, prototype_loss, attention_regularization
from train_utils import evaluate, find_optimal_thresholds


def main():
    parser = argparse.ArgumentParser(
        description="Plot histogram of per-class optimal thresholds"
    )
    parser.add_argument('--checkpoint',   required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--meta_csv',     required=True,
                        help='Path to PTB-XL metadata CSV')
    parser.add_argument('--data_dir',     required=True,
                        help='Directory with WFDB files')
    parser.add_argument('--num_labels',   type=int, required=True,
                        help='Number of output labels')
    parser.add_argument('--batch_size',   type=int, default=64)
    parser.add_argument('--d_model',      type=int, default=128)
    parser.add_argument('--n_heads',      type=int, default=8)
    parser.add_argument('--n_layers',     type=int, default=4)
    parser.add_argument('--num_prototypes', type=int, default=64)
    parser.add_argument('--num_concepts',   type=int, default=20)
    parser.add_argument('--device',       default='cuda')
    parser.add_argument('--output',       default='thresholds_hist.png',
                        help='Where to save the histogram')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- DataLoader for validation set ---
    val_ds = ECGDataset(args.meta_csv, args.data_dir, use_lowres=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # --- Load model ---
    model = HybridECGModel(
        in_channels=12,
        d_model=args.d_model,
        num_prototypes=args.num_prototypes,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_concepts=args.num_concepts,
        num_labels=args.num_labels
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # --- Evaluate to get raw probabilities and true labels ---
    # Note: criterion_fn is only used for loss; thresholds come from probs/targets
    def criterion(p, t):
        return classification_loss(p, t) + prototype_loss(model) + attention_regularization(p)

    val_loss, metrics, probs, targets = evaluate(
        model, val_loader, criterion, device
    )
    print(f"Val loss {val_loss:.4f}, metrics {metrics}")

    # --- Compute per-class optimal thresholds ---
    thresholds = find_optimal_thresholds(probs, targets)  # shape (num_labels,)

    # --- Plot histogram ---
    plt.figure(figsize=(8, 6))
    plt.hist(thresholds, bins=20, range=(0.0, 1.0), edgecolor='black')
    plt.xlabel('Optimal Threshold per Class')
    plt.ylabel('Number of Classes')
    plt.title('Distribution of Class-wise Optimal Thresholds')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output)
    print(f"Saved histogram to {args.output}")


if __name__ == '__main__':
    main()
