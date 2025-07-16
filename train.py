import os
import argparse
import torch
from torch.utils.data import DataLoader

from data import ECGDataset
from models.hybrid import HybridECGModel
from losses import classification_loss, prototype_loss, attention_regularization
from train_utils import get_optimizer_scheduler, train_one_epoch, evaluate


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
        scheduler_gamma=args.scheduler_gamma
    )

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
                                     lambda p, t: classification_loss(p, t) + prototype_loss(
                                         model) + attention_regularization(p), device)
        val_loss, val_metrics = evaluate(model, val_loader,
                                         lambda p, t: classification_loss(p, t) + prototype_loss(
                                             model) + attention_regularization(p),
                                         device)
        scheduler.step()

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Metrics {val_metrics}")

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
