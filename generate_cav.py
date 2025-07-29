import os
import argparse
import pandas as pd
import torch

from data import ECGDataset
from models.hybrid import HybridECGModel
from explain import learn_cavs

def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple CAVs and save to a single .pth file (resumeable)"
    )
    parser.add_argument('--concepts',       nargs='+', required=True,
                        help='List of SCP-ECG diagnostic_subclass codes')
    parser.add_argument('--meta_csv',       required=True,
                        help='Path to PTB-XL metadata CSV')
    parser.add_argument('--scp_csv',        required=True,
                        help='Path to scp_statements.csv')
    parser.add_argument('--data_dir',       required=True,
                        help='Directory containing ECG WFDB files')
    parser.add_argument('--checkpoint',     required=True,
                        help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--output',         required=True,
                        help='Where to save/load the combined CAVs (.pth)')
    parser.add_argument('--d_model',        type=int, default=128)
    parser.add_argument('--n_heads',        type=int, default=8)
    parser.add_argument('--n_layers',       type=int, default=4)
    parser.add_argument('--num_prototypes', type=int, default=64)
    parser.add_argument('--num_concepts',   type=int, default=20)
    parser.add_argument('--num_labels',     type=int, required=True)
    parser.add_argument('--device',         type=str, default='cuda')
    args = parser.parse_args()

    # load metadata & SCP list
    meta   = pd.read_csv(args.meta_csv)
    scp_df = pd.read_csv(args.scp_csv, index_col=0)
    available = set(scp_df.index)

    # prepare dataset & model
    ds = ECGDataset(args.meta_csv, args.data_dir, use_lowres=False)
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
    model.eval()

    # load existing CAVs if any
    if os.path.exists(args.output):
        cav_dict = torch.load(args.output)
        print(f"[INFO] Loaded existing CAV file with {len(cav_dict)} concepts", flush=True)
    else:
        cav_dict = {}

    # iterate concepts
    for concept in args.concepts:
        if concept in cav_dict:
            print(f"[SKIP] '{concept}' already computed", flush=True)
            continue
        if concept not in available:
            print(f"[WARN] '{concept}' not in SCP list, skipping", flush=True)
            continue

        # special chars safely match
        mask = meta['scp_codes'].fillna('').str.contains(concept, regex=False)
        pos_idx = meta.index[mask].tolist()
        neg_idx = meta.index[~mask].tolist()

        if len(pos_idx) < 2:
            print(f"[WARN] '{concept}' has only {len(pos_idx)} positive samples, skipping", flush=True)
            continue
        if len(neg_idx) < 2:
            print(f"[WARN] '{concept}' has only {len(neg_idx)} negative samples, skipping", flush=True)
            continue

        pos_sigs = [ds[i][0] for i in pos_idx]
        neg_sigs = [ds[i][0] for i in neg_idx]
        print(f"[INFO] Training CAV for '{concept}': pos={len(pos_sigs)}, neg={len(neg_sigs)}", flush=True)

        v_c = learn_cavs(pos_sigs, neg_sigs, model, args.device)
        cav_tensor = torch.from_numpy(v_c).float().cpu()
        cav_dict[concept] = cav_tensor

        # save incrementally
        torch.save(cav_dict, args.output)
        print(f"[INFO] Saved CAV for '{concept}' → {args.output}", flush=True)

    print(f"[DONE] Total CAVs saved: {len(cav_dict)}", flush=True)

if __name__ == '__main__':
    main()
