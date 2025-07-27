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
    12-lead overlay (orig vs CF) + per-lead delta plot

    Left col: original (파란) vs CF (주황)
    Right col: delta = CF - original
    """

    # 1) load CF
    ckpt = torch.load(cf_path, map_location='cpu')
    x_cf = ckpt['x_cf'][0].numpy()   # (12, T)

    # 2) load original
    ds = ECGDataset(meta_csv, data_dir, use_lowres=False)
    x_orig, _ = ds[orig_index]
    x_orig = x_orig.numpy()          # (12, T)

    # 3) compute probs (optional)
    model.to(device).eval()
    with torch.no_grad():
        logits_orig = model(torch.from_numpy(x_orig)[None].to(device))['logits'][0, target_label].item()
        logits_cf   = model(torch.from_numpy(x_cf)[None].to(device))['logits'][0, target_label].item()

    # 4) prepare figure: 12 rows × 2 cols
    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    fig, axes = plt.subplots(12, 2, figsize=(12, 36), constrained_layout=True)

    for i in range(12):
        # overlay orig vs CF
        ax1 = axes[i, 0]
        ax1.plot(x_orig[i], label='Orig', linewidth=1)
        ax1.plot(x_cf[i],   label='CF',   linewidth=1)
        ax1.set_ylabel('mV')
        ax1.set_title(f'Lead {lead_names[i]}')
        if i == 0:
            ax1.legend(loc='upper right')

        # delta only
        ax2 = axes[i, 1]
        delta = x_cf[i] - x_orig[i]
        ax2.plot(delta, color='tab:purple', linewidth=1)
        ax2.set_ylabel('Δ mV')
        if i == 0:
            ax2.set_title('Delta (CF – Orig)')

        # x-label only on last row
        if i == 11:
            ax1.set_xlabel('Sample')
            ax2.set_xlabel('Sample')
        else:
            ax1.set_xticks([])
            ax2.set_xticks([])

    # super-title with index, label, prob change
    fig.suptitle(f"Idx={orig_index}, Label={target_label},  prob {logits_orig:.2f} → {logits_cf:.2f}",
                 fontsize=16, y=1.02)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
