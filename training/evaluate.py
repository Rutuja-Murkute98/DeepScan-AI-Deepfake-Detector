"""
Model evaluation with comprehensive metrics and visualizations.

Usage:
    python -m training.evaluate --data_root data --checkpoint models/best_model.pt
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from training.dataset import DeepfakeDataset
from training.train import create_model, compute_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--checkpoint", default="models/best_model.pt")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--backbone", default="efficientnet_b4")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = DeepfakeDataset(args.data_root, "test", tf)
    loader = DataLoader(ds, args.batch_size, shuffle=False, num_workers=0)

    model = create_model(args.backbone, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Evaluating"):
            probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(lbls.numpy())

    probs_arr = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    preds_arr = (probs_arr >= args.threshold).astype(int)

    report = classification_report(labels_arr, preds_arr, target_names=["Real", "Fake"])
    metrics = compute_metrics(labels_arr, preds_arr, probs_arr)
    auc = roc_auc_score(labels_arr, probs_arr)
    ap = average_precision_score(labels_arr, probs_arr)

    summary = {"auc_roc": round(float(auc), 4), "avg_precision": round(float(ap), 4), "metrics": {k: round(v, 4) for k, v in metrics.items()}, "report": report}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}\n{report}\nAUC: {auc:.4f}  AP: {ap:.4f}\n{'='*60}")

    # Plots
    plt.style.use("dark_background")

    cm = confusion_matrix(labels_arr, preds_arr)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cmn, annot=True, fmt=".2%", cmap="RdYlGn", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(labels_arr, probs_arr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#7c6af7", lw=2, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
    plt.tight_layout(); plt.savefig(out / "roc_curve.png", dpi=150); plt.close()

    print(f"  Results saved to {out.resolve()}")


if __name__ == "__main__":
    main()
