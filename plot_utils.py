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
    4×6 grid:
     [ Lead1 Orig | Lead1 Δ | Lead2 Orig | Lead2 Δ | Lead3 Orig | Lead3 Δ ]
     ...
     [ Lead10 Orig| Lead10 Δ| Lead11 Orig| Lead11 Δ| Lead12 Orig| Lead12 Δ]
    """

    # 1) Load CF & original
    ckpt = torch.load(cf_path, map_location='cpu')
    x_cf   = ckpt['x_cf'][0].numpy()   # (12, T)
    ds     = ECGDataset(meta_csv, data_dir, use_lowres=False)
    x_orig,_ = ds[orig_index]
    x_orig = x_orig.numpy()

    # 2) Compute probs (for display, to 4 decimals)
    model.to(device).eval()
    with torch.no_grad():
        p_o = model(torch.from_numpy(x_orig)[None].to(device))['logits'][0,target_label].item()
        p_c = model(torch.from_numpy(x_cf)[None].to(device))['logits'][0,target_label].item()

    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    fig, axes = plt.subplots(4,6, figsize=(24,16), constrained_layout=True)
    axes = axes.flatten()

    for i in range(12):
        # Orig plot (orange)
        ax_o = axes[2*i]
        ax_o.plot(x_orig[i], color='orange', linewidth=1)
        ax_o.set_title(f'Lead {lead_names[i]}')
        ax_o.set_ylabel('mV')
        if i < 8:
            ax_o.set_xticks([])
        else:
            ax_o.set_xlabel('Sample')

        # Delta plot (blue)
        ax_d = axes[2*i+1]
        delta = x_cf[i] - x_orig[i]
        ax_d.plot(delta, color='blue', linewidth=1)
        ax_d.set_title(f'Δ {lead_names[i]}')
        ax_d.set_ylabel('Δ mV')
        if i < 8:
            ax_d.set_xticks([])
        else:
            ax_d.set_xlabel('Sample')

    # Super title
    fig.suptitle(
        f"Idx={orig_index}, Label={target_label}, prob {p_o:.4f}→{p_c:.4f}",
        fontsize=18, y=1.02
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
