# plot_utils.py

import os
import torch
import matplotlib.pyplot as plt
from data import ECGDataset

def plot_counterfactual(cf_path: str,
                        meta_csv: str,
                        data_dir: str,
                        index: int,
                        output_path: str = None):
    """
    Load cf .pth and original ECG, then plot all 12 leads side by side.
    """
    # load cf
    ckpt = torch.load(cf_path, map_location='cpu')
    x_cf = ckpt['x_cf'][index].numpy()
    # load original
    ds = ECGDataset(meta_csv, data_dir, use_lowres=False)
    x_orig, _ = ds[index]
    x_orig = x_orig.numpy()

    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    fig, axes = plt.subplots(6,2,figsize=(12,12))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot(x_orig[i], label='Orig', linewidth=1)
        ax.plot(x_cf[i],  label='CF',   linewidth=1)
        ax.set_title(f'Lead {lead_names[i]}')
        ax.set_xlabel('Sample'); ax.set_ylabel('mV')
        if i==0: ax.legend()
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close(fig)
