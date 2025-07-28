import argparse
import pandas as pd
import torch

from data import ECGDataset
from models.hybrid import HybridECGModel
from explain import learn_cavs

def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple CAVs and save to a single .pth file"
    )
    parser.add_argument('--concepts',      nargs='+', required=True,
                        help='List of SCP-ECG diagnostic_subclass codes')
    parser.add_argument('--meta_csv',      required=True)
    parser.add_argument('--scp_csv',       required=True)
    parser.add_argument('--data_dir',      required=True)
    parser.add_argument('--checkpoint',    required=True)
    parser.add_argument('--output',        required=True)
    parser.add_argument('--d_model',       type=int, default=128)
    parser.add_argument('--n_heads',       type=int, default=8)
    parser.add_argument('--n_layers',      type=int, default=4)
    parser.add_argument('--num_prototypes',type=int, default=64)
    parser.add_argument('--num_concepts',  type=int, default=20)
    parser.add_argument('--num_labels',    type=int, required=True)
    parser.add_argument('--device',        type=str, default='cuda')
    args = parser.parse_args()

    meta = pd.read_csv(args.meta_csv)
    scp_df = pd.read_csv(args.scp_csv, index_col=0)
    available = set(scp_df.index)

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

    cav_dict = {}
    for concept in args.concepts:
        if concept not in available:
            print(f"[WARN] '{concept}' not in SCP list, skipping.", flush=True)
            continue

        mask = meta['scp_codes'].fillna('').str.contains(concept)
        pos_idx = meta.index[mask].tolist()
        neg_idx = meta.index[~mask].tolist()

        pos_sigs = [ds[i][0] for i in pos_idx]
        neg_sigs = [ds[i][0] for i in neg_idx]

        print(f"[INFO] Training CAV for '{concept}': pos={len(pos_sigs)}, neg={len(neg_sigs)}", flush=True)
        cav = learn_cavs(pos_sigs, neg_sigs, model, args.device)
        cav = torch.from_numpy(cav)
        cav_dict[concept] = cav.cpu()

    torch.save(cav_dict, args.output)
    print(f"[INFO] Saved all CAVs to {args.output}", flush=True)

if __name__ == '__main__':
    main()
