"""
Training pipeline for the deepfake detection model.

Usage:
    python -m training.train --data_dir data --epochs 60 --backbone efficientnet_b4

Expects dataset layout:
    data/
      train/  real/  fake/
      val/    real/  fake/
      test/   real/  fake/

Recommended datasets:
    - FaceForensics++ (https://github.com/ondyari/FaceForensics)
    - DFDC (Deepfake Detection Challenge)
    - Celeb-DF v2
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from training.dataset import DeepfakeDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(backbone="efficientnet_b4", pretrained=True, dropout=0.4):
    import timm
    from backend.services.detector import EfficientNetDetector
    model = EfficientNetDetector(backbone=backbone, dropout=dropout)
    if pretrained:
        # Load pretrained backbone weights
        pretrained_backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        model.backbone.load_state_dict(pretrained_backbone.state_dict())
    return model


def build_transforms(size=224):
    train_tf = transforms.Compose([
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
        transforms.GaussianBlur(3, (0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(0.2),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def compute_metrics(labels, preds, probs=None):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    m = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    if probs is not None:
        try:
            m["auc"] = float(roc_auc_score(labels, probs))
        except ValueError:
            m["auc"] = 0.0
    return m


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    bar = tqdm(loader, desc="Train" if training else "Val", leave=False)
    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            logits = model(images)
            loss = criterion(logits, labels)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        bar.set_postfix(loss=f"{loss.item():.4f}")

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    metrics["loss"] = total_loss / len(all_labels)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--backbone", default="efficientnet_b4")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path("models").mkdir(exist_ok=True)

    train_tf, eval_tf = build_transforms()
    train_ds = DeepfakeDataset(args.data_dir, "train", train_tf)
    val_ds = DeepfakeDataset(args.data_dir, "val", eval_tf)

    train_loader = DataLoader(train_ds, args.batch_size, sampler=train_ds.make_weighted_sampler(), num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0)

    model = create_model(args.backbone, pretrained=True).to(device)

    # Freeze backbone initially
    for p in model.backbone.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.1)

    best_auc = 0.0
    patience_counter = 0

    print(f"\n{'='*60}\n  Training — {args.backbone} on {device}\n{'='*60}")

    for epoch in range(args.epochs):
        # Progressive unfreezing
        if epoch == 5:
            print(f"\n  [Epoch {epoch+1}] Unfreezing top backbone blocks")
            for child in list(model.backbone.children())[-3:]:
                for p in child.parameters():
                    p.requires_grad = True
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr / 5, weight_decay=1e-4)
            scheduler = OneCycleLR(optimizer, max_lr=args.lr / 5, steps_per_epoch=len(train_loader), epochs=args.epochs - epoch, pct_start=0.05)
        elif epoch == 10:
            print(f"\n  [Epoch {epoch+1}] Full fine-tuning")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=args.lr / 10, weight_decay=1e-4)
            scheduler = OneCycleLR(optimizer, max_lr=args.lr / 10, steps_per_epoch=len(train_loader), epochs=args.epochs - epoch, pct_start=0.05)

        train_m = run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
        val_m = run_epoch(model, val_loader, criterion, device)

        print(f"  Epoch {epoch+1}/{args.epochs}  Train: loss={train_m['loss']:.4f} acc={train_m['accuracy']:.4f} auc={train_m.get('auc',0):.4f}  Val: loss={val_m['loss']:.4f} acc={val_m['accuracy']:.4f} auc={val_m.get('auc',0):.4f}")

        val_auc = val_m.get("auc", val_m["accuracy"])
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1, "val_auc": val_auc, "backbone": args.backbone}, "models/best_model.pt")
            print(f"  ✓ Saved best model (val_auc={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("  Early stopping.")
                break

    print(f"\n{'='*60}\n  Done. Best AUC: {best_auc:.4f}\n{'='*60}")


if __name__ == "__main__":
    main()
