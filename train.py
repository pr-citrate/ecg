import os
import argparse
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import ECGDataset, ECGContrastiveDataset
from models.hybrid import HybridECGModel
from losses import (
    classification_loss,
    prototype_loss,
    attention_regularization,
    contrastive_loss,
    compute_jaccard,
    prototype_contrastive_loss
)
from train_utils import get_optimizer_scheduler, evaluate, find_optimal_thresholds


def load_label_matrix(meta_csv: str, data_dir: str) -> np.ndarray:
    """
    ECGDataset을 순회하며 (N, C) 형태의 레이블 행렬을 반환합니다.
    """
    base_ds = ECGDataset(meta_csv, data_dir)
    labels = []
    for item in base_ds:
        # ECGDataset은 (signal, label) 반환
        _, label = item
        # label이 Tensor일 경우 Numpy로 변환
        labels.append(label.cpu().numpy() if isinstance(label, torch.Tensor) else label)
    return np.stack(labels, axis=0)  # shape: (N, C)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid ECG Model")
    parser.add_argument('--meta_csv',    type=str,   required=True)
    parser.add_argument('--data_dir',    type=str,   required=True)
    parser.add_argument('--resume',      type=str,   default=None)
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--weight_decay',type=float, default=1e-5)

    parser.add_argument('--scheduler',          choices=['step', 'cosine', 'plateau'], default='step')
    parser.add_argument('--scheduler_step',     type=int,   default=10)
    parser.add_argument('--scheduler_gamma',    type=float, default=0.1)
    parser.add_argument('--scheduler_patience', type=int,   default=5)
    parser.add_argument('--scheduler_factor',   type=float, default=0.5)

    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--threshold_freq',      type=int, default=10)

    parser.add_argument('--num_prototypes', type=int, default=32)
    parser.add_argument('--num_concepts',   type=int, default=10)
    parser.add_argument('--num_labels',     type=int, required=True)

    parser.add_argument('--d_model',   type=int, default=64)
    parser.add_argument('--n_heads',   type=int, default=4)
    parser.add_argument('--n_layers',  type=int, default=2)

    parser.add_argument('--device',    type=str, default='cuda')
    parser.add_argument('--output_dir',type=str, default='checkpoints')
    parser.add_argument('--log_dir',   type=str, default='runs')

    parser.add_argument('--contrastive',        action='store_true', help='use InfoNCE contrastive loss')
    parser.add_argument('--alpha',              type=float, default=1.0, help='weight for InfoNCE loss')
    parser.add_argument('--temperature',        type=float, default=0.5, help='InfoNCE temperature')
    parser.add_argument('--use_proto_contrast', action='store_true', help='use prototype-level contrastive loss')
    parser.add_argument('--alpha_contrast',     type=float, default=1.0, help='weight for prototype contrastive loss')

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 1) 전체 레이블 행렬 구성 (N, C)
    all_labels = load_label_matrix(args.meta_csv, args.data_dir)

    # 2) pos_weight 계산 (for focal/BCE)
    pos_counts = all_labels.sum(axis=0)
    neg_counts = all_labels.shape[0] - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)
    pos_weight_tensor = torch.tensor(pos_weight, device=device, dtype=torch.float)

    # 3) Jaccard 행렬 계산 (prototype-level contrastive)
    if args.use_proto_contrast:
        Y_tensor = torch.from_numpy(all_labels)
        J = compute_jaccard(Y_tensor)  # shape: (C, C)

    # 4) Dataset & DataLoader
    if args.contrastive:
        train_ds = ECGContrastiveDataset(args.meta_csv, args.data_dir)
    else:
        train_ds = ECGDataset(args.meta_csv, args.data_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_ds     = ECGDataset(args.meta_csv, args.data_dir, use_lowres=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 5) Model 생성
    model = HybridECGModel(
        in_channels=12,
        d_model=args.d_model,
        num_prototypes=args.num_prototypes,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_concepts=args.num_concepts,
        num_labels=args.num_labels
    ).to(device)

    # 6) W_proto 구성
    if args.use_proto_contrast:
        proto_classes = model.prototype_layer.class_assign  # Tensor length P
        W_proto = J[proto_classes][:, proto_classes].to(device)  # shape: (P, P)

    # 7) Optimizer & Scheduler
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

    # 8) TensorBoard Writer
    run_dir = os.path.join(args.log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=run_dir)
    thresholds = np.full(args.num_labels, 0.5)

    # 9) Resume 설정
    start_epoch   = 1
    best_val_f1   = 0.0
    best_val_loss = float('inf')
    no_improve    = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        scheduler.load_state_dict(ckpt['sched_state'])
        start_epoch   = ckpt.get('epoch', 0) + 1
        best_val_f1   = ckpt.get('best_f1', 0.0)
        best_val_loss = ckpt.get('best_loss', best_val_loss)
        print(f"[Resume] from epoch {start_epoch}, best_val_f1={best_val_f1:.4f}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # 10) Training & Validation Loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss       = 0.0
        total_clf_loss   = 0.0
        total_con_loss   = 0.0
        total_proto_con  = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # a) Contrastive vs. standard 분기
            if args.contrastive:
                v1, v2, labels = batch
                v1, v2, labels = v1.to(device), v2.to(device), labels.to(device)

                out1 = model(v1)
                out2 = model(v2)

                # 분류 손실
                logits    = out1['logits']
                loss_clf  = classification_loss(
                    logits, labels,
                    loss_type='focal',
                    gamma=2.0,
                    pos_weight=pos_weight_tensor
                )
                # InfoNCE 손실
                loss_con  = contrastive_loss(
                    out1['concept_scores'],
                    out2['concept_scores'],
                    temperature=args.temperature
                )
                loss = loss_clf + args.alpha * loss_con
            else:
                x, labels = batch
                x, labels = x.to(device), labels.to(device)
                out       = model(x)
                logits    = out['logits']
                loss      = classification_loss(
                    logits, labels,
                    loss_type='focal',
                    gamma=2.0,
                    pos_weight=pos_weight_tensor
                )
                loss_clf  = loss
                loss_con  = torch.tensor(0.0, device=device)

            # b) Prototype & Attention Regularization
            loss_proto    = prototype_loss(model)
            attn_feats    = out1['attn_feats'] if args.contrastive else out['attn_feats']
            loss_attn     = attention_regularization(attn_feats)
            loss         += loss_proto + loss_attn

            # c) Prototype-level 대조 손실
            if args.use_proto_contrast:
                protos         = model.prototype_layer.prototype_vectors
                loss_p_contrast = prototype_contrastive_loss(protos, W_proto)
                loss          += args.alpha_contrast * loss_p_contrast
            else:
                loss_p_contrast = torch.tensor(0.0, device=device)

            # d) Backprop & step
            loss.backward()
            optimizer.step()

            # e) Accumulate for logging
            bsz = v1.size(0) if args.contrastive else x.size(0)
            total_loss      += loss.item() * bsz
            total_clf_loss  += loss_clf.item() * bsz
            total_con_loss  += loss_con.item() * bsz
            total_proto_con += loss_p_contrast.item() * bsz

            # f) Step-level TensorBoard 로그
            global_step = (epoch - 1) * len(train_loader) + step
            writer.add_scalar("Loss/train_step", loss.item(), global_step)

        # Epoch-level 평균 손실 계산
        dataset_size = len(train_loader.dataset)
        train_loss       = total_loss / dataset_size
        train_clf_avg    = total_clf_loss / dataset_size
        train_con_avg    = total_con_loss / dataset_size if args.contrastive else 0.0
        train_proto_avg  = total_proto_con / dataset_size if args.use_proto_contrast else 0.0

        # 11) TensorBoard Epoch-level 로그
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/train_classification", train_clf_avg, epoch)
        if args.contrastive:
            writer.add_scalar("Loss/train_contrastive", train_con_avg, epoch)
        if args.use_proto_contrast:
            writer.add_scalar("Loss/train_proto_contrast", train_proto_avg, epoch)

        # 파라미터 & 그래디언트 분포
        for name, p in model.named_parameters():
            writer.add_histogram(f"Weights/{name}", p.data, epoch)
            if p.grad is not None:
                writer.add_histogram(f"Grads/{name}", p.grad, epoch)

        # 임베딩 시각화 (contrastive인 경우)
        if args.contrastive:
            embeddings, metadata = [], []
            model.eval()
            with torch.no_grad():
                for xb in train_loader:
                    v1, _, lbls = xb
                    emb = model(v1.to(device))['concept_scores'].cpu()
                    embeddings.append(emb)
                    metadata += [str(int(x)) for x in lbls]
                embeddings = torch.cat(embeddings, dim=0)
                writer.add_embedding(embeddings, metadata=metadata, global_step=epoch, tag="Embeddings")

        # 모델 그래프 (첫 에포크에만)
        if epoch == start_epoch:
            sample = next(iter(train_loader))
            sample_x = sample[0] if args.contrastive else sample[0]
            writer.add_graph(model, sample_x.to(device))

        # 하이퍼파라미터 비교
        writer.add_hparams(
            {
                "lr": args.lr,
                "batch_size": args.batch_size,
                "alpha": args.alpha,
                "alpha_contrast": args.alpha_contrast
            },
            {"hparam/train_loss": train_loss}
        )

        # 12) Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_metrics, probs, targets = evaluate(
                model, val_loader,
                lambda p, t: (
                    classification_loss(p, t, loss_type='focal', gamma=2.0, pos_weight=pos_weight_tensor)
                    + prototype_loss(model)
                    + attention_regularization(p)
                ),
                device
            )

        # Scheduler step
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Threshold tuning
        if epoch % args.threshold_freq == 0:
            thresholds = find_optimal_thresholds(probs, targets)
            print(f"[Epoch {epoch}] thresholds updated", flush=True)

        # TensorBoard Validation 로그
        writer.add_scalar('Loss/val',      val_loss,           epoch)
        writer.add_scalar('Metrics/micro_f1', val_metrics['micro_f1'], epoch)
        writer.add_scalar('Metrics/macro_f1', val_metrics['macro_f1'], epoch)
        writer.add_scalar('Metrics/mean_auroc', val_metrics['mean_auroc'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Progress 출력
        print(
            f"[Epoch {epoch}] "
            f"Train {train_loss:.4f}  Val {val_loss:.4f}  "
            f"Micro-F1 {val_metrics['micro_f1']:.3f}  "
            f"Macro-F1 {val_metrics['macro_f1']:.3f}  "
            f"AUROC {val_metrics['mean_auroc']:.3f}",
            flush=True
        )

        # 체크포인트 저장
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict(),
            'best_f1': best_val_f1,
            'best_loss': best_val_loss
        }
        torch.save(ckpt, os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt"))

        # 베스트 갱신
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1   = val_metrics['macro_f1']
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"New best macro-F1: {best_val_f1:.3f}", flush=True)
            no_improve = 0
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= args.early_stop_patience:
            print(f"No improvement for {args.early_stop_patience} epochs, stopping.", flush=True)
            break

    writer.close()


if __name__ == '__main__':
    main()
