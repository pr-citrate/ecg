import argparse
import torch
import matplotlib.pyplot as plt
from data import ECGDataset

def main():
    parser = argparse.ArgumentParser(description="Plot Original vs Counterfactual ECG for all 12 leads")
    parser.add_argument('--cf_path',   type=str, required=True, help="Path to counterfactual .pth file")
    parser.add_argument('--meta_csv',  type=str, required=True, help="Path to metadata CSV")
    parser.add_argument('--data_dir',  type=str, required=True, help="Directory with WFDB files")
    parser.add_argument('--index',     type=int, default=0, help="Record index to plot")
    parser.add_argument('--output',    type=str, default=None, help="If set, save figure to this file")
    args = parser.parse_args()

    # Load counterfactual
    ckpt = torch.load(args.cf_path, map_location='cpu')
    x_cf = ckpt['x_cf'][args.index].numpy()  # shape (12, T)

    # Load original
    ds = ECGDataset(args.meta_csv, args.data_dir, use_lowres=False)
    x_orig, _ = ds[args.index]
    x_orig = x_orig.numpy()  # shape (12, T)

    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

    fig, axes = plt.subplots(6, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(12):
        ax = axes[i]
        ax.plot(x_orig[i], label='Original', linewidth=1)
        ax.plot(x_cf[i],  label='Counterfactual', linewidth=1)
        ax.set_title(f'Lead {lead_names[i]}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('mV')
        if i == 0:
            ax.legend(loc='upper right')

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
