import os
import argparse
import torch
import matplotlib.pyplot as plt
from data import ECGDataset
from models.hybrid import HybridECGModel

def plot_cf_twinscale(cf_path: str,
                      meta_csv: str,
                      data_dir: str,
                      orig_index: int,
                      target_label: int,
                      model: torch.nn.Module,
                      device: str = 'cpu',
                      output_path: str = None):
    """
    For each of the 12 leads, overlay:
      - original ECG (orange solid)
      - counterfactual ECG (green solid)
      - delta = CF − orig (blue, on secondary y-axis)
    in a 3×4 grid of wide subplots.
    """
    # load counterfactual
    ckpt = torch.load(cf_path, map_location='cpu')
    x_cf = ckpt['x_cf'][0].numpy()   # (12, T)
    # load original
    ds = ECGDataset(meta_csv, data_dir, use_lowres=False)
    x_orig, _ = ds[orig_index]
    x_orig = x_orig.numpy()          # (12, T)

    # optionally compute probs
    model.to(device).eval()
    with torch.no_grad():
        p_orig = model(torch.from_numpy(x_orig)[None].to(device))['logits'][0, target_label].item()
        p_cf   = model(torch.from_numpy(x_cf)[None].to(device))['logits'][0, target_label].item()

    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    fig, axes = plt.subplots(3, 4, figsize=(20, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        orig = x_orig[i]
        cf   = x_cf[i]
        delta = cf - orig

        # primary axis: orig & CF
        ax.plot(orig, label='Orig', color='orange', linewidth=1)
        ax.plot(cf,   label='CF',   color='green',  linewidth=1)

        # secondary axis: delta
        ax2 = ax.twinx()
        ax2.plot(delta, label='Δ', color='blue', linewidth=1, alpha=0.7)

        # titles & labels
        ax.set_title(f'Lead {lead_names[i]}')
        ax.set_ylabel('mV')
        ax2.set_ylabel('Δ mV')
        ax.set_xticks([])
        ax2.set_xticks([])

        # legend on first subplot
        if i == 0:
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')

    # supertitle with index, label, probs
    fig.suptitle(f"Idx={orig_index}, Label={target_label}, prob {p_orig:.4f}→{p_cf:.4f}",
                 fontsize=16, y=1.02)

    # save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plot original vs CF vs delta for 12-lead ECG in twin-scale subplots"
    )
    parser.add_argument('--cf_path',      type=str, required=True)
    parser.add_argument('--meta_csv',     type=str, required=True)
    parser.add_argument('--data_dir',     type=str, required=True)
    parser.add_argument('--orig_index',   type=int, required=True)
    parser.add_argument('--target_label', type=int, required=True)
    parser.add_argument('--checkpoint',   type=str, required=True,
                        help="path to trained model .pt for probability calc")
    parser.add_argument('--d_model',        type=int, default=64)
    parser.add_argument('--n_heads',        type=int, default=4)
    parser.add_argument('--n_layers',       type=int, default=2)
    parser.add_argument('--num_prototypes', type=int, default=32)
    parser.add_argument('--num_concepts',   type=int, default=10)
    parser.add_argument('--num_labels',     type=int, required=True)
    parser.add_argument('--device',        type=str, default='cpu')
    parser.add_argument('--output',        type=str, default=None)
    args = parser.parse_args()

    # load model
    model = HybridECGModel(
        in_channels=12,
        d_model=args.d_model,
        num_prototypes=args.num_prototypes,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_concepts=args.num_concepts,
        num_labels=args.num_labels
    ).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    plot_cf_twinscale(
        cf_path=args.cf_path,
        meta_csv=args.meta_csv,
        data_dir=args.data_dir,
        orig_index=args.orig_index,
        target_label=args.target_label,
        model=model,
        device=args.device,
        output_path=args.output
    )
