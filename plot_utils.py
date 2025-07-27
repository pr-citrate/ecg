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
    4×3 grid: lightgreen Δ behind, blue CF, orange Orig.
    figsize=(21,12)
    """

    # Load signals
    ckpt = torch.load(cf_path, map_location='cpu')
    x_cf = ckpt['x_cf'][0].numpy()
    ds = ECGDataset(meta_csv, data_dir, use_lowres=False)
    x_orig, _ = ds[orig_index]
    x_orig = x_orig.numpy()

    # Compute probs
    model.to(device).eval()
    with torch.no_grad():
        p_o = model(torch.from_numpy(x_orig)[None].to(device))['logits'][0, target_label].item()
        p_c = model(torch.from_numpy(x_cf)[None].to(device))  ['logits'][0, target_label].item()

    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    fig, axes = plt.subplots(4, 3, figsize=(21, 12), constrained_layout=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # twin x-axis for Δ
        ax2 = ax.twinx()
        # make its background invisible
        ax2.patch.set_visible(False)
        ax2.set_frame_on(False)

        # Δ line: lightgreen, thin, above any patch
        delta = x_cf[i] - x_orig[i]
        ax2.plot(delta, color='lightgreen', linewidth=0.5, zorder=0.5, label='Δ')
        ax2.set_ylabel('Δ mV')

        # CF (blue) and Orig (orange) on main ax
        ax.plot(x_cf[i],   color='blue',   linewidth=1.0, zorder=1, label='CF')
        ax.plot(x_orig[i], color='orange', linewidth=1.0, zorder=2, label='Orig')

        # labels
        ax.set_title(f'Lead {lead_names[i]}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('mV')

        # combined legend
        if i == 0:
            lines, labs = ax.get_legend_handles_labels()
            l2, lab2 = ax2.get_legend_handles_labels()
            ax.legend(lines + l2, labs + lab2, loc='upper right')

    # super title
    fig.suptitle(f"Idx={orig_index}, Label={target_label}, prob {p_o:.4f} → {p_c:.4f}",
                 fontsize=18, y=1.02)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
