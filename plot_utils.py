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
    Plot original ECG and delta (CF – Orig) overlaid in the same axes
    with separate y-scales using twin axes. Arranged as a 4×3 grid for 12 leads.
    """

    # Load CF and original signals
    ckpt = torch.load(cf_path, map_location='cpu')
    x_cf = ckpt['x_cf'][0].numpy()   # shape (12, T)

    ds = ECGDataset(meta_csv, data_dir, use_lowres=False)
    x_orig, _ = ds[orig_index]
    x_orig = x_orig.numpy()          # shape (12, T)

    # Compute probabilities for display
    model.to(device).eval()
    with torch.no_grad():
        p_o = model(torch.from_numpy(x_orig)[None].to(device))['logits'][0, target_label].item()
        p_c = model(torch.from_numpy(x_cf)[None].to(device))['logits'][0, target_label].item()

    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

    fig, axes = plt.subplots(4, 3, figsize=(18, 16), constrained_layout=True)
    axes = axes.flatten()

    for i in range(12):
        ax = axes[i]
        ax2 = ax.twinx()  # second y-axis for delta

        # Plot original signal
        ax.plot(x_orig[i], label='Orig', linewidth=1)
        ax.set_ylabel('mV')

        # Plot delta signal
        delta = x_cf[i] - x_orig[i]
        ax2.plot(delta, linestyle='--', label='Delta', linewidth=1)
        ax2.set_ylabel('Δ mV')

        # Title and ticks
        ax.set_title(f'Lead {lead_names[i]}')
        if i < 9:
            ax.set_xticks([])
            ax2.set_xticks([])
        else:
            ax.set_xlabel('Sample')
            ax2.set_xlabel('Sample')

        # Combined legend
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

    # Super title with index, label, and probability change
    fig.suptitle(
        f"Idx={orig_index}, Label={target_label}, prob {p_o:.4f} → {p_c:.4f}",
        fontsize=18,
        y=1.02
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
