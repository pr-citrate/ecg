import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ECGDataset
from models.hybrid import HybridECGModel
from losses import classification_loss, prototype_loss, attention_regularization
from train_utils import (
    get_optimizer_scheduler,
    train_one_epoch,
    evaluate,
    find_optimal_thresholds
)
from explain import generate_counterfactual, learn_cavs, compute_tcav_scores
from plot_utils import plot_counterfactual


def train_mode(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Datasets
    train_ds = ECGDataset(args.meta_csv, args.data_dir, use_lowres=False)
    val_ds = ECGDataset(args.meta_csv, args.data_dir, use_lowres=False)

    # class-balanced sampling
    all_labels = np.vstack([lbl for _, lbl in train_ds])
    sample_weights = 1.0 / (all_labels.sum(axis=1) + 1)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # pos_weight for Focal/BCE loss
    pos_counts = all_labels.sum(axis=0)
    neg_counts = all_labels.shape[0] - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float, device=device)
    print("[Init] pos_weight_tensor computed", flush=True)

    # initial thresholds & warmup
    thresholds = np.full(args.num_labels, 0.5)
    print(f"[Init] thresholds set to 0.5", flush=True)
    base_lr = args.lr
    warmup_epochs = args.warmup_epochs
    print(f"[Init] warmup_epochs = {warmup_epochs}", flush=True)

    # Model
    model = HybridECGModel(
        in_channels=12,
        d_model=args.d_model,
        num_prototypes=args.num_prototypes,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_concepts=args.num_concepts,
        num_labels=args.num_labels
    ).to(device)

    # Optimizer & Scheduler
    optimizer, scheduler = get_optimizer_scheduler(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        scheduler_step_size=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor
    )

    best_f1 = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # linear warm-up for first warmup_epochs
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warm_lr = base_lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warm_lr

        loss_fn = lambda p, t: classification_loss(
            p, t,
            loss_type='focal',
            gamma=2.0,
            pos_weight=pos_weight_tensor
        ) + prototype_loss(model) + attention_regularization(p)

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        val_loss, metrics, probs, targets = evaluate(
            model, val_loader, loss_fn, device, threshold=thresholds
        )

        # scheduler step
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # threshold tuning
        if epoch % args.threshold_freq == 0:
            thresholds = find_optimal_thresholds(probs, targets)
            print(f"[Epoch {epoch}] Updated thresholds", flush=True)

        # logging
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Metrics: {metrics}", flush=True)

        # save best
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            path = os.path.join(args.output_dir, f"best_epoch{epoch}.pt")
            torch.save(model.state_dict(), path)
            print(f"Saved best model to {path}", flush=True)


def cf_mode(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Load model
    model = HybridECGModel(
        in_channels=12,
        d_model=args.d_model,
        num_prototypes=args.num_prototypes,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_concepts=args.num_concepts,
        num_labels=args.num_labels
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    # support multiple

    for idx in args.indices:
        for tgt in args.target_labels:
            ds = ECGDataset(args.meta_csv, args.data_dir, use_lowres=False)
            x, _ = ds[idx]
            x = x.unsqueeze(0).to(device)

            mask = None

            if args.mask:
                mask = torch.load(args.mask).to(device)

            # generate CF
            x_cf, delta = generate_counterfactual(
                model, x, tgt,
                lambda_coeff = args.lambda_coeff,
                steps = args.steps,
                lr = args.lr_cf,
                mask = mask,
                device = device
            )
            cf_file = os.path.join(args.output_dir, f"cf_idx{idx}_lbl{tgt}.pth")
            torch.save({'x_cf': x_cf, 'delta': delta}, cf_file)
            print(f"Saved counterfactual to {cf_file}", flush=True)

            # plot comparison
            plot_file = cf_file.replace('.pth', '.png')
            plot_counterfactual(
                cf_file,
                args.meta_csv,
                args.data_dir,
                orig_index=idx,  # 원본 ECG 인덱스
                target_label=tgt,  # CF 생성 타겟 레이블
                model=model,
                device=args.device,
                output_path=plot_file
            )
            print(f"Saved plot to {plot_file}", flush=True)

def cav_mode(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = HybridECGModel(
        in_channels=12,
        d_model=args.d_model,
        num_prototypes=args.num_prototypes,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_concepts=args.num_concepts,
        num_labels=args.num_labels
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Collect examples
    def load_examples(dir_path):
        files = os.listdir(dir_path)
        ds = ECGDataset(args.meta_csv, args.data_dir, use_lowres=False)
        examples = []
        for fname in files:
            sig, _ = ds[int(fname)]
            examples.append(sig)
        return examples

    pos = load_examples(args.pos_dir)
    neg = load_examples(args.neg_dir)
    cav = learn_cavs(pos, neg, model, device)
    torch.save(cav, args.output)
    print(f"Saved CAV vector to {args.output}", flush=True)


def tcav_mode(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = HybridECGModel(
        in_channels=12,
        d_model=args.d_model,
        num_prototypes=args.num_prototypes,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_concepts=args.num_concepts,
        num_labels=args.num_labels
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    cavs = torch.load(args.cav_path)

    ds = ECGDataset(args.meta_csv, args.data_dir, use_lowres=False)
    loader = DataLoader(ds, batch_size=args.batch_size)

    scores = compute_tcav_scores(model, cavs, loader, args.labels, device)
    torch.save(scores, args.output)
    print(f"Saved TCAV scores to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)

    # Train
    p_train = sub.add_parser('train')
    p_train.add_argument('--meta_csv',      required=True)
    p_train.add_argument('--data_dir',      required=True)
    p_train.add_argument('--num_labels',    type=int,   required=True)
    p_train.add_argument('--epochs',        type=int,   default=50)
    p_train.add_argument('--batch_size',    type=int,   default=32)
    p_train.add_argument('--lr',            type=float, default=1e-4)
    p_train.add_argument('--weight_decay',  type=float, default=1e-5)

    p_train.add_argument('--scheduler',         type=str,   choices=['step','cosine','plateau'], default='step')
    p_train.add_argument('--scheduler_step',    type=int,   default=10)
    p_train.add_argument('--scheduler_gamma',   type=float, default=0.1)
    p_train.add_argument('--scheduler_patience',type=int,   default=5)
    p_train.add_argument('--scheduler_factor',  type=float, default=0.5)

    p_train.add_argument('--warmup_epochs',     type=int,   default=0,
                         help='number of epochs to linearly warm up LR')
    p_train.add_argument('--threshold_freq',    type=int,   default=20,
                         help='epochs between threshold recalculations')

    p_train.add_argument('--d_model',        type=int, default=64)
    p_train.add_argument('--n_heads',        type=int, default=4)
    p_train.add_argument('--n_layers',       type=int, default=2)
    p_train.add_argument('--num_prototypes', type=int, default=32)
    p_train.add_argument('--num_concepts',   type=int, default=10)

    p_train.add_argument('--device',     default='cuda')
    p_train.add_argument('--output_dir', default='chkpts')
    p_train.add_argument('--log_dir',    default='runs')

    # Counterfactual
    p_cf = sub.add_parser('cf')
    p_cf.add_argument('--checkpoint',     required=True)
    p_cf.add_argument('--meta_csv',       required=True)
    p_cf.add_argument('--data_dir',       required=True)
    p_cf.add_argument('--indices',       type=int,   nargs='+', required=True)
    p_cf.add_argument('--target_labels', type=int,   nargs='+', required=True)
    p_cf.add_argument('--output_dir',    type=str,   required=True,
                        help="directory to save cf .pth and plot images")
    p_cf.add_argument('--lambda_coeff',   type=float, default=0.1)
    p_cf.add_argument('--steps',          type=int,   default=100)
    p_cf.add_argument('--lr_cf',          type=float, default=1e-2)
    p_cf.add_argument('--mask',           default=None)
    p_cf.add_argument('--device',         default='cuda')

    # Model hyperparams for CF/CAV/TCAV
    for p in (p_cf,):
        p.add_argument('--d_model',        type=int, default=64)
        p.add_argument('--n_heads',        type=int, default=4)
        p.add_argument('--n_layers',       type=int, default=2)
        p.add_argument('--num_prototypes', type=int, default=32)
        p.add_argument('--num_concepts',   type=int, default=10)
        p.add_argument('--num_labels',     type=int, required=True)

    # Learn CAV
    p_cav = sub.add_parser('cav')
    p_cav.add_argument('--checkpoint', required=True)
    p_cav.add_argument('--meta_csv',   required=True)
    p_cav.add_argument('--data_dir',   required=True)
    p_cav.add_argument('--pos_dir',    required=True)
    p_cav.add_argument('--neg_dir',    required=True)
    p_cav.add_argument('--output',     required=True)
    p_cav.add_argument('--device',     default='cuda')
    for p in (p_cav,):
        p.add_argument('--d_model',        type=int, default=64)
        p.add_argument('--n_heads',        type=int, default=4)
        p.add_argument('--n_layers',       type=int, default=2)
        p.add_argument('--num_prototypes', type=int, default=32)
        p.add_argument('--num_concepts',   type=int, default=10)
        p.add_argument('--num_labels',     type=int, required=True)

    # TCAV
    p_tcav = sub.add_parser('tcav')
    p_tcav.add_argument('--checkpoint', required=True)
    p_tcav.add_argument('--cav_path',   required=True)
    p_tcav.add_argument('--meta_csv',   required=True)
    p_tcav.add_argument('--data_dir',   required=True)
    p_tcav.add_argument('--labels',     type=int, nargs='+', required=True)
    p_tcav.add_argument('--batch_size', type=int, default=32)
    p_tcav.add_argument('--output',     required=True)
    p_tcav.add_argument('--device',     default='cuda')
    for p in (p_tcav,):
        p.add_argument('--d_model',        type=int, default=64)
        p.add_argument('--n_heads',        type=int, default=4)
        p.add_argument('--n_layers',       type=int, default=2)
        p.add_argument('--num_prototypes', type=int, default=32)
        p.add_argument('--num_concepts',   type=int, default=10)
        p.add_argument('--num_labels',     type=int, required=True)

    args = parser.parse_args()
    print(args, flush=True)

    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'cf':
        cf_mode(args)
    elif args.mode == 'cav':
        cav_mode(args)
    elif args.mode == 'tcav':
        tcav_mode(args)


if __name__ == '__main__':
    main()
