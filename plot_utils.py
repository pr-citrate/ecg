# plot_utils.py

import os
import torch
import matplotlib.pyplot as plt
from data import ECGDataset

def plot_counterfactual(cf_path: str,
                        meta_csv: str,
                        data_dir: str,
                        orig_index: int,
                        target_label: int,
                        model: torch.nn.Module,
                        device: str = 'cpu',
                        output_path: str = None):
    """
    For each of the 12 leads, plot original vs CF on the primary y-axis
    and delta = CF - original on a secondary y-axis, in a 4×3 grid.
    Each subplot is wider than it is tall.
    """

    # 1) Load counterfactual & original signal
    ckpt = torch.load(cf_path, map_location='cpu')
    x_cf = ckpt['x_cf'][0].numpy()   # (12, T)
    ds = ECGDataset(meta_csv, data_dir, use_lowres=False)
    x_orig, _ = ds[orig_index]
    x_orig = x_orig.numpy()          # (12, T)

    # 2) Compute model probabilities for display
    model.to(device).eval()
    with torch.no_grad():
        p_o = model(torch.from_numpy(x_orig)[None].to(device))['logits'][0, target_label].item()
        p_c = model(torch.from_numpy(x_cf)[None].to(device))['logits'][0, target_label].item()

    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

    # 3) Prepare figure: 4 rows × 3 cols
    fig, axes = plt.subplots(4, 3,
                             figsize=(18, 12),
                             constrained_layout=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Primary plot: original (orange) and CF (blue)
        ax.plot(x_orig[i], color='orange', label='Orig', linewidth=1)
        ax.plot(x_cf[i],   color='blue',   label='CF',   linewidth=1)

        # Secondary axis: delta
        ax2 = ax.twinx()
        delta = x_cf[i] - x_orig[i]
        ax2.plot(delta, color='purple', alpha=0.7, label='Δ', linewidth=1)

        # Titles and labels
        ax.set_title(f'Lead {lead_names[i]}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('mV')
        ax2.set_ylabel('Δ mV')

        # Legends only on first subplot
        if i == 0:
            orig_legend = ax.legend(loc='upper left')
            ax2.legend(loc='upper right')

    # Super-title with index, label, and probabilities to 4 decimals
    fig.suptitle(
        f"Idx={orig_index}, Label={target_label}, prob {p_o:.4f} → {p_c:.4f}",
        fontsize=18,
        y=1.02
    )

    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
