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
    Plot each of the 12 leads in a 2×6 grid with:
     - blue solid = CF
     - orange solid = Original (drawn on top)
    Figure size is (24, 8) for wider subplots.
    """

    # 1) Load counterfactual and original signals
    ckpt = torch.load(cf_path, map_location='cpu')
    x_cf   = ckpt['x_cf'][0].numpy()   # (12, T)
    ds     = ECGDataset(meta_csv, data_dir, use_lowres=False)
    x_orig,_ = ds[orig_index]
    x_orig = x_orig.numpy()            # (12, T)

    # 2) Compute model probabilities for display
    model.to(device).eval()
    with torch.no_grad():
        p_o = model(torch.from_numpy(x_orig)[None].to(device))['logits'][0, target_label].item()
        p_c = model(torch.from_numpy(x_cf)[None].to(device))['logits'][0, target_label].item()

    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

    # 3) Create 2×6 grid, wider figure
    fig, axes = plt.subplots(6, 2,
                             figsize=(12, 12))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Plot CF (blue)
        ax.plot(x_cf[i],   color='blue',   linewidth=1.0, label='CF',   zorder=1)
        # Plot original on top (orange)
        ax.plot(x_orig[i], color='orange', linewidth=1.0, label='Orig', zorder=2)

        # Titles and labels
        ax.set_title(f'Lead {lead_names[i]}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('mV')

        # Legend only on first subplot
        if i == 0:
            ax.legend(loc='upper right')

    # Super-title with index, label, and precise probability change
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
    else:
        plt.show()
