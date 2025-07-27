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
    각 리드별로 4×3 그리드:
      - lightgreen Δ (linewidth=0.3) 축을 제일 아래(zorder 낮게)
      - blue CF (linewidth=1, zorder 중간)
      - orange Orig (linewidth=1, zorder 가장 높게)
    figsize=(21,12)
    """

    # 1) Load signals
    ckpt = torch.load(cf_path, map_location='cpu')
    x_cf = ckpt['x_cf'][0].numpy()   # (12, T)
    ds = ECGDataset(meta_csv, data_dir, use_lowres=False)
    x_orig, _ = ds[orig_index]
    x_orig = x_orig.numpy()

    # 2) Compute logits for display
    model.to(device).eval()
    with torch.no_grad():
        p_o = model(torch.from_numpy(x_orig)[None].to(device))['logits'][0, target_label].item()
        p_c = model(torch.from_numpy(x_cf)[None].to(device))  ['logits'][0, target_label].item()

    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

    # 3) Plot setup
    fig, axes = plt.subplots(4, 3,
                             figsize=(21, 12),
                             constrained_layout=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # 3.1) Δ on ax2, 아래로
        ax2 = ax.twinx()
        delta = x_cf[i] - x_orig[i]
        ax2.set_zorder(ax.get_zorder() - 1)  # ax 아래로
        ax2.patch.set_alpha(0)               # 투명 배경
        ax2.plot(delta, color='lightgreen', linewidth=0.3, label='Δ')

        # 3.2) CF
        ax.plot(x_cf[i],   color='blue',   linewidth=1.0, zorder=2, label='CF')
        # 3.3) Orig (맨 위)
        ax.plot(x_orig[i], color='orange', linewidth=1.0, zorder=3, label='Orig')

        # titles & labels
        ax.set_title(f'Lead {lead_names[i]}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('mV')
        ax2.set_ylabel('Δ mV')

        # legend only once
        if i == 0:
            lines, labs = ax.get_legend_handles_labels()
            l2, lab2 = ax2.get_legend_handles_labels()
            ax.legend(lines + l2, labs + lab2, loc='upper right')

    # 4) super-title
    fig.suptitle(
        f"Idx={orig_index}, Label={target_label}, prob {p_o:.4f} → {p_c:.4f}",
        fontsize=18, y=1.02
    )

    # 5) save/show
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
