"""CLIP Linear Probe — Frozen CLIP encoder + trainable linear classifier.

Usage
-----
    python scripts/run_clip_linear_probe.py
    python scripts/run_clip_linear_probe.py --config configs/clip_linear_probe.yaml

Trains a single Linear(512→1) layer on L2-normalised CLIP image embeddings.
The CLIP encoder remains completely frozen throughout.
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
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
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.mri_dataset import MRIDataset  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── embedding extraction ────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(clip_model, processor, dataset, batch_size, device):
    """Extract L2-normalised CLIP image embeddings for an entire dataset.

    Returns
    -------
    embeddings : torch.Tensor  — (N, 512)
    labels     : torch.Tensor  — (N,)
    sources    : list[str]
    """
    clip_model.eval()
    all_embs = []
    all_labels = []
    all_sources = []

    n = len(dataset)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        pil_images = []
        batch_labels = []
        batch_sources = []
        for idx in range(start, end):
            path, label = dataset.samples[idx]
            img = Image.open(path).convert("RGB")
            pil_images.append(img)
            batch_labels.append(label)
            batch_sources.append(dataset.sources[idx])

        inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        features = clip_model.get_image_features(**inputs)  # (B, 512)
        features = features / features.norm(dim=-1, keepdim=True)

        all_embs.append(features.cpu())
        all_labels.extend(batch_labels)
        all_sources.extend(batch_sources)

        if (start // batch_size) % 20 == 0:
            print(f"    {end}/{n} ...")

    embeddings = torch.cat(all_embs, dim=0)        # (N, 512)
    labels = torch.tensor(all_labels, dtype=torch.long)
    return embeddings, labels, all_sources


# ── metrics (matching ResNet format) ─────────────────────────────────────────

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


def compute_per_source_accuracy(labels, probs, sources):
    preds = (probs >= 0.5).astype(int)
    src_correct = defaultdict(int)
    src_total = defaultdict(int)
    for pred, label, src in zip(preds, labels, sources):
        src_total[src] += 1
        if pred == label:
            src_correct[src] += 1
    return {
        src: round(src_correct[src] / src_total[src], 4)
        for src in sorted(src_total)
    }


# ── plotting ─────────────────────────────────────────────────────────────────

def save_loss_curves(log_rows, out_path):
    epochs = [r["epoch"] for r in log_rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, [r["train_loss"] for r in log_rows], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in log_rows], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [r["val_acc"] for r in log_rows], label="val accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, [r["val_auroc"] for r in log_rows], label="val AUROC")
    axes[2].set_title("Validation AUROC")
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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


# ── training loop ────────────────────────────────────────────────────────────

def train_one_epoch(classifier, embeddings, labels, criterion, optimizer,
                    batch_size, device):
    classifier.train()
    n = embeddings.size(0)
    perm = torch.randperm(n)
    running_loss = 0.0
    all_probs, all_labels = [], []

    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        emb = embeddings[idx].to(device)
        lbl = labels[idx].float().to(device)

        optimizer.zero_grad()
        logits = classifier(emb).squeeze(1)
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * emb.size(0)
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())

    epoch_loss = running_loss / n
    preds = (np.array(all_probs) >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.0
    return epoch_loss, acc, auroc


@torch.no_grad()
def evaluate_embeddings(classifier, embeddings, labels, criterion,
                        batch_size, device):
    classifier.eval()
    n = embeddings.size(0)
    running_loss = 0.0
    all_probs, all_labels = [], []

    for start in range(0, n, batch_size):
        emb = embeddings[start:start + batch_size].to(device)
        lbl = labels[start:start + batch_size].float().to(device)

        logits = classifier(emb).squeeze(1)
        loss = criterion(logits, lbl)

        running_loss += loss.item() * emb.size(0)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())

    epoch_loss = running_loss / n if n > 0 else 0.0
    preds = (np.array(all_probs) >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds) if n > 0 else 0.0
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.0
    return epoch_loss, acc, auroc, np.array(all_labels), np.array(all_probs)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CLIP Linear Probe — MRI Real vs Synthetic")
    parser.add_argument("--config", type=str,
                        default="configs/clip_linear_probe.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    experiment_dir = PROJECT_ROOT / cfg.get(
        "experiment_dir", "experiments/clip_linear_probe")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    clip_model_name = cfg.get("clip_model_name", "openai/clip-vit-base-patch32")
    batch_size = cfg.get("batch_size", 64)
    epochs = cfg.get("epochs", 20)
    lr = cfg.get("learning_rate", 1e-3)
    weight_decay = cfg.get("weight_decay", 1e-4)
    num_workers = cfg.get("num_workers", 0)

    print(f"Device          : {device}")
    print(f"CLIP model      : {clip_model_name}")
    print(f"Dataset         : {dataset_path}")
    print(f"Epochs          : {epochs}")
    print(f"LR              : {lr}")
    print(f"Output          : {experiment_dir}")

    # ── load CLIP (frozen) ───────────────────────────────────────────────
    print("\nLoading CLIP model ...")
    clip_model = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    print("  CLIP loaded and frozen.")

    # ── load datasets ────────────────────────────────────────────────────
    train_ds = MRIDataset(str(dataset_path), split="train", transform=None)
    val_ds = MRIDataset(str(dataset_path), split="val", transform=None)
    test_ds = MRIDataset(str(dataset_path), split="test", transform=None)
    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, "
          f"test: {len(test_ds)}")

    # ── extract embeddings (one-time cost) ───────────────────────────────
    print("\nExtracting CLIP embeddings ...")
    t0 = time.time()

    print("  Train split:")
    train_emb, train_lbl, train_src = extract_embeddings(
        clip_model, processor, train_ds, batch_size, device)

    print("  Val split:")
    val_emb, val_lbl, val_src = extract_embeddings(
        clip_model, processor, val_ds, batch_size, device)

    print("  Test split:")
    test_emb, test_lbl, test_src = extract_embeddings(
        clip_model, processor, test_ds, batch_size, device)

    emb_time = time.time() - t0
    print(f"  Embedding extraction done ({emb_time:.1f}s)")
    print(f"  Embedding dim: {train_emb.shape[1]}")

    # Free CLIP from GPU after extraction
    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── build linear classifier ──────────────────────────────────────────
    embed_dim = train_emb.shape[1]  # 512
    classifier = nn.Linear(embed_dim, 1).to(device)

    # Class weighting
    n_real = (train_lbl == 0).sum().item()
    n_fake = (train_lbl == 1).sum().item()
    pos_weight = torch.tensor([n_real / n_fake]).to(device) if n_fake > 0 \
        else torch.tensor([1.0]).to(device)
    print(f"  pos_weight: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=epochs)

    # ── training loop ────────────────────────────────────────────────────
    print(f"\nTraining linear probe ({epochs} epochs) ...")
    best_auroc = 0.0
    log_rows = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_auroc = train_one_epoch(
            classifier, train_emb, train_lbl, criterion, optimizer,
            batch_size, device)

        val_loss, val_acc, val_auroc, _, _ = evaluate_embeddings(
            classifier, val_emb, val_lbl, criterion, batch_size, device)

        scheduler.step()

        improved = ""
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(classifier.state_dict(),
                       experiment_dir / "best_model.pth")
            improved = " *"

        print(f"  Epoch {epoch:>2d}/{epochs}  "
              f"t_loss={train_loss:.4f}  v_loss={val_loss:.4f}  "
              f"v_acc={val_acc:.4f}  v_auroc={val_auroc:.4f}{improved}")

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "train_auroc": round(train_auroc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "val_auroc": round(val_auroc, 5),
        })

    elapsed = time.time() - t0
    print(f"  Training done ({elapsed:.1f}s). Best val AUROC: {best_auroc:.4f}")

    # Save training log
    log_path = experiment_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    # Save loss curves
    save_loss_curves(log_rows, experiment_dir / "loss_curve.png")

    # ── evaluate on test set ─────────────────────────────────────────────
    print("\nEvaluating on test set ...")
    classifier.load_state_dict(
        torch.load(experiment_dir / "best_model.pth",
                   map_location=device, weights_only=True))

    _, test_acc, test_auroc, test_labels, test_probs = evaluate_embeddings(
        classifier, test_emb, test_lbl, criterion, batch_size, device)

    metrics = compute_metrics(test_labels, test_probs)
    metrics["model"] = clip_model_name
    metrics["classifier"] = "Linear(512, 1)"
    metrics["split"] = "test"
    metrics["n_samples"] = int(len(test_labels))
    metrics["best_val_auroc"] = round(best_auroc, 5)
    metrics["training_time_sec"] = round(elapsed, 2)
    metrics["embedding_time_sec"] = round(emb_time, 2)
    metrics["per_source_accuracy"] = compute_per_source_accuracy(
        test_labels, test_probs, test_src)

    # ── save artifacts ───────────────────────────────────────────────────
    with open(experiment_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        save_roc_curve(test_labels, test_probs,
                       "ROC — CLIP Linear Probe",
                       experiment_dir / "roc_curve.png")
    except ValueError:
        pass

    preds = (test_probs >= 0.5).astype(int)
    save_confusion_matrix(test_labels, preds,
                          "CM — CLIP Linear Probe",
                          experiment_dir / "confusion_matrix.png")

    # ── print summary ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("CLIP LINEAR PROBE RESULTS")
    print(f"{'='*55}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  AUROC     : {metrics['auroc']:.4f}")
    print(f"  AUPRC     : {metrics['auprc']:.4f}")
    print(f"\nPer-source accuracy:")
    for src, acc in metrics["per_source_accuracy"].items():
        print(f"    {src:<15s} {acc:.4f}")
    print(f"\nResults saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
