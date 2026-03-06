"""ResNet50 Baseline — Training, Evaluation, and Logging Pipeline.

Usage
-----
    python src/train_resnet_baseline.py --config configs/resnet_baseline.yaml

Runs on CUDA if available, otherwise CPU.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Resolve project root so imports work regardless of cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.mri_dataset import MRIDataset  # noqa: E402
from src.models.resnet50_baseline import build_resnet50_baseline  # noqa: E402


# ===== helpers =============================================================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== training & validation loops =========================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    preds = (np.array(all_probs) >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.0
    return epoch_loss, acc, auroc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)

        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    preds = (np.array(all_probs) >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.0
    return epoch_loss, acc, auroc, np.array(all_labels), np.array(all_probs)


# ===== full evaluation with all metrics ====================================

def compute_full_metrics(labels, probs, dataset: MRIDataset):
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }

    # Per-source accuracy
    source_correct = defaultdict(int)
    source_total = defaultdict(int)
    for i, (pred, label) in enumerate(zip(preds, labels)):
        src = dataset.get_source_for_index(i)
        source_total[src] += 1
        if pred == label:
            source_correct[src] += 1

    metrics["per_source_accuracy"] = {
        src: round(source_correct[src] / source_total[src], 4)
        for src in sorted(source_total)
    }
    return metrics


# ===== plotting ============================================================

def save_loss_curves(log_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(log_df["epoch"], log_df["train_loss"], label="train")
    axes[0].plot(log_df["epoch"], log_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(log_df["epoch"], log_df["accuracy"], label="val accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(log_df["epoch"], log_df["auroc"], label="val AUROC")
    axes[2].set_title("Validation AUROC")
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc_curve(labels, probs, out_path: Path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_val = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUROC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — ResNet50 Baseline")
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(labels, preds, out_path: Path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=16)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — ResNet50 Baseline")
    fig.colorbar(im)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===== main ================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 baseline")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Resolve dataset path relative to project root
    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    experiment_dir = PROJECT_ROOT / cfg.get("experiment_dir",
                                             "experiments/resnet_baseline")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    # ------ datasets & loaders ------
    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 0)

    train_ds = MRIDataset(str(dataset_path), split="train")
    val_ds = MRIDataset(str(dataset_path), split="val")
    test_ds = MRIDataset(str(dataset_path), split="test")

    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, "
          f"test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # ------ model ------
    model = build_resnet50_baseline(pretrained=True).to(device)

    # ------ loss (weighted BCE) ------
    pos_weight = train_ds.get_class_weights().to(device)
    print(f"Class pos_weight (n_real/n_fake): {pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ------ optimizer & scheduler ------
    lr = cfg.get("learning_rate", 3e-4)
    weight_decay = cfg.get("weight_decay", 1e-4)
    epochs = cfg.get("epochs", 25)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=epochs)

    # ------ training loop ------
    best_auroc = 0.0
    log_rows = []

    print(f"\nTraining for {epochs} epochs …\n")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_auroc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)

        val_loss, val_acc, val_auroc, _, _ = evaluate(
            model, val_loader, criterion, device)

        scheduler.step()

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "accuracy": round(val_acc, 5),
            "auroc": round(val_auroc, 5),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
        })

        improved = ""
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(),
                       experiment_dir / "best_model.pth")
            improved = " ★"

        print(f"Epoch {epoch:>3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_auroc={val_auroc:.4f}{improved}")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed / 60:.1f} min. "
          f"Best val AUROC: {best_auroc:.4f}")

    # ------ save training log ------
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(experiment_dir / "training_log.csv", index=False)

    # ------ training plots ------
    save_loss_curves(log_df, experiment_dir / "loss_curve.png")

    # ------ test set evaluation ------
    print("\nEvaluating on test set …")
    model.load_state_dict(torch.load(experiment_dir / "best_model.pth",
                                     map_location=device, weights_only=True))
    _, test_acc, test_auroc, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device)

    metrics = compute_full_metrics(test_labels, test_probs, test_ds)
    metrics["best_val_auroc"] = round(best_auroc, 5)
    metrics["training_time_min"] = round(elapsed / 60, 2)

    # Save metrics
    with open(experiment_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save test-set plots
    save_roc_curve(test_labels, test_probs, experiment_dir / "roc_curve.png")
    test_preds = (test_probs >= 0.5).astype(int)
    save_confusion_matrix(test_labels, test_preds,
                          experiment_dir / "confusion_matrix.png")

    # ------ summary ------
    print(f"\n{'='*50}")
    print("TEST RESULTS — ResNet50 Baseline")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  AUROC     : {metrics['auroc']:.4f}")
    print(f"  AUPRC     : {metrics['auprc']:.4f}")
    print(f"\nPer-source accuracy:")
    for src, acc in metrics["per_source_accuracy"].items():
        print(f"    {src:20s}: {acc:.4f}")
    print(f"\nArtifacts saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
