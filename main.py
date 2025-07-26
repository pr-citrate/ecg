import os
import argparse
import torch
from torch.utils.data import DataLoader

from data import ECGDataset
from models.hybrid import HybridECGModel
from losses import classification_loss, prototype_loss, attention_regularization
from train_utils import get_optimizer_scheduler, train_one_epoch, evaluate
from explain import generate_counterfactual, learn_cavs, compute_tcav_scores


def train_mode(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Datasets
    train_ds = ECGDataset(args.meta_csv, args.data_dir, use_lowres=False)
    val_ds = ECGDataset(args.meta_csv, args.data_dir, use_lowres=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    # Model
    model = HybridECGModel(
        in_channels=12, d_model=args.d_model,
        num_prototypes=args.num_prototypes,
        n_heads=args.n_heads, n_layers=args.n_layers,
        num_concepts=args.num_concepts, num_labels=args.num_labels
    ).to(device)
    # Optimizer & Scheduler
    optimizer, scheduler = get_optimizer_scheduler(
        model,
        lr = args.lr,
        weight_decay = args.weight_decay,
        scheduler_type = args.scheduler,
        scheduler_step_size = args.scheduler_step,
        scheduler_gamma = args.scheduler_gamma,
        scheduler_patience = args.scheduler_patience,
        scheduler_factor = args.scheduler_factor
    )
    best_f1 = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            lambda p,t: classification_loss(p,t) + prototype_loss(model) + attention_regularization(p),
            device
        )
        val_loss, metrics, probs, targets = evaluate(
            model, val_loader,
            lambda p,t: classification_loss(p,t) + prototype_loss(model) + attention_regularization(p),
            device
        )

        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Metrics: {metrics}", flush=True)
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            path = os.path.join(args.output_dir, f"best_epoch{epoch}.pt")
            torch.save(model.state_dict(), path)
            print(f"Saved best model to {path}", flush=True)


def cf_mode(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Load model
    model = HybridECGModel(
        in_channels=12, d_model=args.d_model,
        num_prototypes=args.num_prototypes,
        n_heads=args.n_heads, n_layers=args.n_layers,
        num_concepts=args.num_concepts, num_labels=args.num_labels
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    # Load single example
    ds = ECGDataset(args.meta_csv, args.data_dir, use_lowres=False)
    x, _ = ds[args.index]
    x = x.unsqueeze(0)
    mask = None
    if args.mask:
        mask = torch.load(args.mask).to(device)
    x_cf, delta = generate_counterfactual(
        model, x, args.target_label,
        lambda_coeff=args.lambda_coeff,
        steps=args.steps, lr=args.lr_cf, mask=mask,
        device=device
    )
    torch.save({'x_cf': x_cf, 'delta': delta}, args.output)
    print(f"Saved counterfactual to {args.output}", flush=True)


def cav_mode(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = HybridECGModel(
        in_channels=12, d_model=args.d_model,
        num_prototypes=args.num_prototypes,
        n_heads=args.n_heads, n_layers=args.n_layers,
        num_concepts=args.num_concepts, num_labels=args.num_labels
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    # Collect examples
    def load_examples(dir_path):
        files = os.listdir(dir_path)
        ds = ECGDataset(args.meta_csv, args.data_dir, use_lowres=False)
        ex = []
        for i in files:
            sig, _ = ds[int(i)]
            ex.append(sig)
        return ex
    pos = load_examples(args.pos_dir)
    neg = load_examples(args.neg_dir)
    cav = learn_cavs(pos, neg, model, device)
    torch.save(cav, args.output)
    print(f"Saved CAV vector to {args.output}", flush=True)


def tcav_mode(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = HybridECGModel(
        in_channels=12, d_model=args.d_model,
        num_prototypes=args.num_prototypes,
        n_heads=args.n_heads, n_layers=args.n_layers,
        num_concepts=args.num_concepts, num_labels=args.num_labels
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    # Load CAVs
    cavs = torch.load(args.cav_path)
    # DataLoader
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
    p_train.add_argument('--scheduler', type=str, choices=['step', 'cosine', 'plateau'], default='step')
    p_train.add_argument('--scheduler_patience', type=int, default=5)
    p_train.add_argument('--scheduler_factor', type=float, default=0.5)
    p_train.add_argument('--log_dir', type=str, default='runs')
    p_train.add_argument('--meta_csv', required=True)
    p_train.add_argument('--data_dir', required=True)
    p_train.add_argument('--epochs', type=int, default=50)
    p_train.add_argument('--batch_size', type=int, default=32)
    p_train.add_argument('--lr', type=float, default=1e-4)
    p_train.add_argument('--weight_decay', type=float, default=1e-5)
    p_train.add_argument('--scheduler_step', type=int, default=10)
    p_train.add_argument('--scheduler_gamma', type=float, default=0.1)
    p_train.add_argument('--num_prototypes', type=int, default=32)
    p_train.add_argument('--num_concepts', type=int, default=10)
    p_train.add_argument('--num_labels', type=int, required=True)
    p_train.add_argument('--device', default='cuda')
    p_train.add_argument('--output_dir', default='chkpts')
    p_train.add_argument('--d_model', type=int, default=64)
    p_train.add_argument('--n_heads', type=int, default=4)
    p_train.add_argument('--n_layers', type=int, default=2)
    p_train.add_argument('--early_stop_patience', type=int, default=10)
    p_train.add_argument('--threshold_freq', type=int, default=10)

    # Counterfactual
    p_cf = sub.add_parser('cf')
    p_cf.add_argument('--checkpoint', required=True)
    p_cf.add_argument('--meta_csv', required=True)
    p_cf.add_argument('--data_dir', required=True)
    p_cf.add_argument('--index', type=int, required=True)
    p_cf.add_argument('--target_label', type=int, required=True)
    p_cf.add_argument('--output', required=True)
    p_cf.add_argument('--lambda_coeff', type=float, default=0.1)
    p_cf.add_argument('--steps', type=int, default=100)
    p_cf.add_argument('--lr_cf', type=float, default=1e-2)
    p_cf.add_argument('--mask', default=None)
    p_cf.add_argument('--device', default='cuda')

    # Learn CAV
    p_cav = sub.add_parser('cav')
    p_cav.add_argument('--checkpoint', required=True)
    p_cav.add_argument('--meta_csv', required=True)
    p_cav.add_argument('--data_dir', required=True)
    p_cav.add_argument('--pos_dir', required=True)
    p_cav.add_argument('--neg_dir', required=True)
    p_cav.add_argument('--output', required=True)
    p_cav.add_argument('--num_prototypes', type=int, default=32)
    p_cav.add_argument('--num_concepts', type=int, default=10)
    p_cav.add_argument('--num_labels', type=int, required=True)
    p_cav.add_argument('--d_model', type=int, default=64)
    p_cav.add_argument('--n_heads', type=int, default=4)
    p_cav.add_argument('--n_layers', type=int, default=2)
    p_cav.add_argument('--device', default='cuda')

    # TCAV
    p_tcav = sub.add_parser('tcav')
    p_tcav.add_argument('--checkpoint', required=True)
    p_tcav.add_argument('--cav_path', required=True)
    p_tcav.add_argument('--meta_csv', required=True)
    p_tcav.add_argument('--data_dir', required=True)
    p_tcav.add_argument('--labels', type=int, nargs='+', required=True)
    p_tcav.add_argument('--batch_size', type=int, default=32)
    p_tcav.add_argument('--output', required=True)
    p_tcav.add_argument('--num_prototypes', type=int, default=32)
    p_tcav.add_argument('--num_concepts', type=int, default=10)
    p_tcav.add_argument('--num_labels', type=int, required=True)
    p_tcav.add_argument('--d_model', type=int, default=64)
    p_tcav.add_argument('--n_heads', type=int, default=4)
    p_tcav.add_argument('--n_layers', type=int, default=2)
    p_tcav.add_argument('--device', default='cuda')

    args = parser.parse_args()
    print(args)

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
