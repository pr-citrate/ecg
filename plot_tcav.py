import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_tcav(scores_dict, labels, concepts, out_path, title):
    """
    scores_dict: dict(concept -> dict(label->score))
    labels: list of label indices
    concepts: list of concept names
    """
    # 행: concepts, 열: labels
    M = np.zeros((len(concepts), len(labels)))
    for i,c in enumerate(concepts):
        for j,l in enumerate(labels):
            M[i, j] = scores_dict[c].get(l, np.nan)

    fig, ax = plt.subplots(figsize=(12,8))
    im = ax.pcolormesh(M, cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label='TCAV score')
    ax.set_xticks(np.arange(0.5, len(labels), 1))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(np.arange(0.5, len(concepts), 1))
    ax.set_yticklabels(concepts)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tcav_path',  required=True)
    p.add_argument('--labels',     type=int, nargs='+', required=True)
    p.add_argument('--concepts',   nargs='+', required=True)
    p.add_argument('--output',     required=True)
    p.add_argument('--title',      default=None)
    args = p.parse_args()

    tcav_scores = torch.load(args.tcav_path)
    title = args.title or args.tcav_path
    plot_tcav(tcav_scores, args.labels, args.concepts, args.output, title)
    print(f"[INFO] Saved TCAV heatmap to {args.output}")

if __name__ == '__main__':
    main()
