"""Cross-Generator Generalization Experiment — ResNet50 (Leave-One-Generator-Out).

Usage
-----
    python src/train_resnet_cross_generator.py --config configs/resnet_cross_generator.yaml

For each synthetic generator, trains on remaining generators + all real data,
then evaluates on the held-out generator to test generalization.
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
from torch.utils.data import DataLoader, ConcatDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.mri_dataset import MRIDataset, get_transforms  # noqa: E402
from src.models.resnet50_baseline import build_resnet50_baseline  # noqa: E402


# ===== helpers ==============================================================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== training & evaluation (reused from baseline) ========================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels, all_probs = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
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
        images, labels = images.to(device), labels.float().to(device)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(loader.dataset)
    epoch_loss = running_loss / n if n > 0 else 0.0
    preds = (np.array(all_probs) >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds) if n > 0 else 0.0
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.0
    return epoch_loss, acc, auroc, np.array(all_labels), np.array(all_probs)


def compute_metrics(labels, probs):
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auroc"] = 0.0
    try:
        metrics["auprc"] = float(average_precision_score(labels, probs))
    except ValueError:
        metrics["auprc"] = 0.0
    metrics["confusion_matrix"] = confusion_matrix(labels, preds).tolist()
    return metrics


def compute_per_generator_accuracy(labels, probs, dataset):
    preds = (probs >= 0.5).astype(int)
    source_correct = defaultdict(int)
    source_total = defaultdict(int)
    for i, (pred, label) in enumerate(zip(preds, labels)):
        src = dataset.get_source_for_index(i)
        source_total[src] += 1
        if pred == label:
            source_correct[src] += 1
    return {
        src: round(source_correct[src] / source_total[src], 4)
        for src in sorted(source_total)
    }


# ===== plotting =============================================================

def save_roc_curve(labels, probs, title, out_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_val = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUROC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(labels, preds, title, out_path):
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
    ax.set_title(title)
    fig.colorbar(im)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===== single leave-one-out run =============================================

def run_single_experiment(held_out_gen, all_synthetic, dataset_path,
                          cfg, experiment_dir, device):
    """Train excluding *held_out_gen*, evaluate on it."""

    fold_dir = experiment_dir / f"{held_out_gen}_test"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Held-out generator: {held_out_gen}")
    print(f"  Training generators: {[g for g in all_synthetic if g != held_out_gen]}")
    print(f"{'='*60}")

    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 0)
    epochs = cfg.get("epochs", 15)
    lr = cfg.get("learning_rate", 3e-4)
    weight_decay = cfg.get("weight_decay", 1e-4)

    # --- datasets ---
    # Training: all splits combined, excluding held-out generator
    train_ds = MRIDataset(str(dataset_path), split="train",
                          exclude_sources=[held_out_gen])
    val_ds = MRIDataset(str(dataset_path), split="val",
                        exclude_sources=[held_out_gen])

    # Test: only the held-out generator (fake) + all real
    test_ds = MRIDataset(str(dataset_path), split="test",
                         include_sources=[held_out_gen, "cermep", "tcga", "upenn"])

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    if len(train_ds) == 0 or len(test_ds) == 0:
        print(f"  SKIP — empty dataset for {held_out_gen}")
        return None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # --- model ---
    model = build_resnet50_baseline(pretrained=True).to(device)

    pos_weight = train_ds.get_class_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=epochs)

    # --- training loop ---
    best_auroc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_auroc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)

        val_loss, val_acc, val_auroc, _, _ = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        improved = ""
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), fold_dir / "best_model.pth")
            improved = " ★"

        print(f"  Epoch {epoch:>2d}/{epochs}  "
              f"t_loss={train_loss:.4f}  v_loss={val_loss:.4f}  "
              f"v_acc={val_acc:.4f}  v_auroc={val_auroc:.4f}{improved}")

    elapsed = time.time() - t0
    print(f"  Training done ({elapsed / 60:.1f} min). Best val AUROC: {best_auroc:.4f}")

    # --- evaluate on held-out generator ---
    model.load_state_dict(torch.load(fold_dir / "best_model.pth",
                                     map_location=device, weights_only=True))
    _, test_acc, test_auroc, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device)

    metrics = compute_metrics(test_labels, test_probs)
    metrics["held_out_generator"] = held_out_gen
    metrics["best_val_auroc"] = round(best_auroc, 5)
    metrics["training_time_min"] = round(elapsed / 60, 2)
    metrics["per_generator_accuracy"] = compute_per_generator_accuracy(
        test_labels, test_probs, test_ds)

    # --- save artifacts ---
    with open(fold_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        save_roc_curve(test_labels, test_probs,
                       f"ROC — held-out {held_out_gen}",
                       fold_dir / "roc_curve.png")
    except ValueError:
        pass

    test_preds = (test_probs >= 0.5).astype(int)
    save_confusion_matrix(test_labels, test_preds,
                          f"CM — held-out {held_out_gen}",
                          fold_dir / "confusion_matrix.png")

    print(f"  Test Accuracy={metrics['accuracy']:.4f}  "
          f"AUROC={metrics['auroc']:.4f}  F1={metrics['f1']:.4f}")

    return metrics


# ===== main =================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Generator Generalization Experiment (ResNet50)")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    experiment_dir = PROJECT_ROOT / cfg.get(
        "experiment_dir", "experiments/resnet_cross_generator")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    synthetic_generators = cfg["synthetic_generators"]
    print(f"Synthetic generators: {synthetic_generators}")
    print(f"Running {len(synthetic_generators)} leave-one-out experiments\n")

    summary = {}
    for gen in synthetic_generators:
        metrics = run_single_experiment(
            held_out_gen=gen,
            all_synthetic=synthetic_generators,
            dataset_path=dataset_path,
            cfg=cfg,
            experiment_dir=experiment_dir,
            device=device,
        )
        if metrics:
            summary[f"{gen}_test"] = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "per_generator_accuracy": metrics["per_generator_accuracy"],
            }

    # Save summary
    with open(experiment_dir / "summary_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print("CROSS-GENERATOR GENERALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Held-out':<15s} {'Acc':>8s} {'Prec':>8s} {'Rec':>8s} "
          f"{'F1':>8s} {'AUROC':>8s} {'AUPRC':>8s}")
    print("-" * 70)
    for key, m in summary.items():
        gen = key.replace("_test", "")
        print(f"{gen:<15s} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['auroc']:>8.4f} "
              f"{m['auprc']:>8.4f}")
    print(f"\nResults saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
