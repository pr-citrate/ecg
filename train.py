import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import ECGDataset
from models.hybrid import HybridECGModel
from losses import classification_loss, prototype_loss, attention_regularization
from train_utils import get_optimizer_scheduler, train_one_epoch, evaluate, find_optimal_thresholds


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid ECG Model")
    parser.add_argument('--meta_csv', type=str, required=True, help='Path to metadata CSV')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with WFDB files')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--scheduler_step', type=int, default=10)
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)
    parser.add_argument('--num_prototypes', type=int, default=32)
    parser.add_argument('--num_concepts', type=int, default=10)
    parser.add_argument('--num_labels', type=int, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--scheduler', type=str, choices=['step', 'cosine', 'plateau'], default='step')
    parser.add_argument('--log_dir', type=str, default='runs')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Dataset & DataLoader
    train_dataset = ECGDataset(
        metadata_csv=args.meta_csv,
        data_dir=args.data_dir,
        use_lowres=False
    )
    val_dataset = ECGDataset(
        metadata_csv=args.meta_csv,
        data_dir=args.data_dir,
        use_lowres=False
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ——— Compute pos_weight tensor for imbalanced BCE/focal loss ———
    # gather all training labels
    all_labels = np.vstack([label.numpy() for _, label in train_loader.dataset])
    pos_counts = all_labels.sum(axis=0)
    neg_counts = all_labels.shape[0] - pos_counts
    # avoid division by zero
    pos_weight = neg_counts / (pos_counts + 1e-6)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float, device=device)
    print(f"[Init] pos_weight_tensor: {pos_weight_tensor}", flush=True)

    # Model
    model = HybridECGModel(
        in_channels=12,
        d_model=64,
        num_prototypes=args.num_prototypes,
        n_heads=4,
        n_layers=2,
        num_concepts=args.num_concepts,
        num_labels=args.num_labels
    ).to(device)

    # Optimizer & Scheduler
    optimizer, scheduler = get_optimizer_scheduler(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_step_size=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_type = args.scheduler
    )

    writer = SummaryWriter(log_dir=args.log_dir)
    thresholds = np.full(args.num_labels, 0.5)

    # ——— Resume from checkpoint if provided ———
    start_epoch = 1
    best_val_f1 = 0.0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        scheduler.load_state_dict(ckpt['sched_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_f1 = ckpt.get('best_f1', 0.0)
        print(f"[Resume] Starting from epoch {start_epoch}, best_val_f1={best_val_f1:.4f}")

    # Training loop
    best_val_f1 = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     lambda p, t: classification_loss(p, t, loss_type='focal', gamma=2.0, pos_weight=pos_weight_tensor)
                                                  + prototype_loss(model)
                                                  + attention_regularization(p), device)
        val_loss, val_metrics, probs, targets = evaluate(model, val_loader,
                                         lambda p, t: classification_loss(p, t, loss_type='focal', gamma=2.0, pos_weight=pos_weight_tensor)
                                                  + prototype_loss(model)
                                                  + attention_regularization(p), device)
        # Scheduler step
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Threshold tuning every 2 epochs

        if epoch % 2 == 0:
            thresholds = find_optimal_thresholds(probs, targets)
            print(f"[Epoch {epoch}] Updated thresholds: {thresholds}", flush=True)

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/micro_f1', val_metrics['micro_f1'], epoch)
        writer.add_scalar('Metrics/macro_f1', val_metrics['macro_f1'], epoch)
        writer.add_scalar('Metrics/mean_auroc', val_metrics['mean_auroc'], epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        print(f"[Epoch {epoch}] Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Metrics: {val_metrics}",
              flush=True)

        # 1) Save checkpoint for every epoch
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict(),
            'best_f1': best_val_f1,
        }, ckpt_path)

        # 2) Update best model
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved to {best_path}")


if __name__ == '__main__':
    main()
