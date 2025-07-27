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
    # load CF
    ckpt = torch.load(cf_path, map_location='cpu')
    x_cf = ckpt['x_cf'][0].numpy()   # (12, T)

    # load original
    ds = ECGDataset(meta_csv, data_dir, use_lowres=False)
    x_orig, _ = ds[orig_index]
    x_orig = x_orig.numpy()          # (12, T)

    # compute probabilities
    model.to(device).eval()
    with torch.no_grad():
        logit_orig = model(torch.from_numpy(x_orig)[None].to(device))['logits'][0, target_label].item()
        logit_cf   = model(torch.from_numpy(x_cf)[None].to(device))['logits'][0, target_label].item()

    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    fig, axes = plt.subplots(7,2,figsize=(12,14))
    axes = axes.flatten()

    # 12 leads
    for i in range(12):
        ax = axes[i]
        ax.plot(x_orig[i], label='Orig', linewidth=1)
        ax.plot(x_cf[i],   label='CF',   linewidth=1)
        ax.set_title(f'Lead {lead_names[i]}')
        if i==0: ax.legend()

    # delta
    delta = x_cf - x_orig
    ax = axes[12]
    ax.plot(delta.T, alpha=0.7)
    ax.set_title('Delta (CF − Orig)')
    axes[13].axis('off')

    # super title
    supt = f"Idx={orig_index}, Label={target_label}, prob {logit_orig:.2f}→{logit_cf:.2f}"
    fig.suptitle(supt, fontsize=16)

    plt.tight_layout(rect=[0,0,1,0.96])
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close(fig)
