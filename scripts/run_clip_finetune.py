"""CLIP Fine-Tuning — Contrastive learning on MRI Real vs Synthetic.

Usage
-----
    python scripts/run_clip_finetune.py
    python scripts/run_clip_finetune.py --config configs/clip_finetune.yaml

Fine-tunes both CLIP encoders using contrastive loss so that image and text
embeddings align for real / synthetic MRI classes.
"""

import argparse
import csv
import json
import math
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
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.mri_dataset import LABEL_MAP, MRIDataset  # noqa: E402

# ── text prompts ─────────────────────────────────────────────────────────────

PROMPT_MAP = {
    0: "a real brain MRI image",      # Real
    1: "a synthetic brain MRI image",  # Fake
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── dataset wrapper ──────────────────────────────────────────────────────────

class CLIPFinetuneDataset(Dataset):
    """Wraps MRIDataset to yield (PIL image, text prompt, label, source)."""

    def __init__(self, mri_dataset: MRIDataset):
        self.mri_dataset = mri_dataset

    def __len__(self):
        return len(self.mri_dataset)

    def __getitem__(self, idx):
        path, label = self.mri_dataset.samples[idx]
        image = Image.open(path).convert("RGB")
        text = PROMPT_MAP[label]
        source = self.mri_dataset.sources[idx]
        return image, text, label, source


def collate_fn(batch):
    """Custom collate — keeps PIL images as a list (CLIPProcessor handles them)."""
    images, texts, labels, sources = zip(*batch)
    return list(images), list(texts), torch.tensor(labels, dtype=torch.long), list(sources)


# ── contrastive loss ─────────────────────────────────────────────────────────

def clip_contrastive_loss(image_embeds, text_embeds, logit_scale):
    """Symmetric CLIP contrastive loss.

    Parameters
    ----------
    image_embeds : (B, D) — L2-normalised
    text_embeds  : (B, D) — L2-normalised
    logit_scale  : scalar (learnable log-scale from CLIPModel)

    Returns
    -------
    loss : scalar tensor
    """
    # logit_scale is stored as log; exponentiate
    scale = logit_scale.exp()
    logits_per_image = scale * image_embeds @ text_embeds.T  # (B, B)
    logits_per_text = logits_per_image.T                     # (B, B)

    targets = torch.arange(len(image_embeds), device=image_embeds.device)

    loss_i2t = nn.functional.cross_entropy(logits_per_image, targets)
    loss_t2i = nn.functional.cross_entropy(logits_per_text, targets)
    return (loss_i2t + loss_t2i) / 2.0


# ── optimizer with per-component LR ─────────────────────────────────────────

def build_optimizer(model, cfg):
    """Create AdamW with separate learning rates for image, text, projection."""
    lr_image = cfg.get("lr_image", 1e-5)
    lr_text = cfg.get("lr_text", 1e-5)
    lr_proj = cfg.get("lr_projection", 1e-4)
    wd = cfg.get("weight_decay", 1e-4)

    image_params = []
    text_params = []
    proj_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "visual_projection" in name or "text_projection" in name:
            proj_params.append(param)
        elif "vision_model" in name:
            image_params.append(param)
        elif "text_model" in name:
            text_params.append(param)
        else:
            # logit_scale and any remaining
            other_params.append(param)

    param_groups = [
        {"params": image_params, "lr": lr_image},
        {"params": text_params, "lr": lr_text},
        {"params": proj_params, "lr": lr_proj},
        {"params": other_params, "lr": lr_proj},
    ]
    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    return torch.optim.AdamW(param_groups, weight_decay=wd)


# ── cosine scheduler ─────────────────────────────────────────────────────────

def build_scheduler(optimizer, epochs, steps_per_epoch):
    total_steps = epochs * steps_per_epoch
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)


# ── metrics ──────────────────────────────────────────────────────────────────

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


# ── zero-shot validation ────────────────────────────────────────────────────

@torch.no_grad()
def validate_zero_shot(model, processor, dataset, batch_size, device):
    """Zero-shot classification using the two class prompts.

    Returns labels, probs (P(fake)), sources.
    """
    model.eval()

    # Build class embeddings from the two prompts
    prompts = [PROMPT_MAP[0], PROMPT_MAP[1]]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    # text_features: (2, D) — row 0=real, row 1=fake

    all_labels = []
    all_probs = []
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

        img_inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}

        image_features = model.get_image_features(**img_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        sims = image_features @ text_features.T  # (B, 2)
        probs_fake = torch.softmax(sims, dim=-1)[:, 1].cpu().numpy()

        all_labels.extend(batch_labels)
        all_probs.extend(probs_fake.tolist())
        all_sources.extend(batch_sources)

    return np.array(all_labels), np.array(all_probs), all_sources


# ── plotting ─────────────────────────────────────────────────────────────────

def save_loss_curves(log_rows, out_path):
    epochs = [r["epoch"] for r in log_rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, [r["train_loss"] for r in log_rows], label="train")
    axes[0].set_title("Contrastive Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [r["val_acc"] for r in log_rows], label="val accuracy")
    axes[1].set_title("Validation Accuracy (Zero-Shot)")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, [r["val_auroc"] for r in log_rows], label="val AUROC")
    axes[2].set_title("Validation AUROC (Zero-Shot)")
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


# ── training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, processor, dataloader, optimizer, scheduler,
                    scaler, device):
    """Train one epoch with CLIP contrastive loss + mixed precision."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, texts, labels, sources in dataloader:
        # Process inputs through CLIPProcessor
        img_inputs = processor(images=images, return_tensors="pt", padding=True)
        txt_inputs = processor(text=texts, return_tensors="pt", padding=True,
                               truncation=True)
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            image_embeds = model.get_image_features(**img_inputs)
            text_embeds = model.get_text_features(**txt_inputs)

            # L2 normalise
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            loss = clip_contrastive_loss(
                image_embeds, text_embeds, model.logit_scale)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CLIP Fine-Tuning — MRI Real vs Synthetic")
    parser.add_argument("--config", type=str,
                        default="configs/clip_finetune.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    experiment_dir = PROJECT_ROOT / cfg.get(
        "experiment_dir", "experiments/clip_finetune")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = get_device()
    clip_model_name = cfg.get("clip_model_name", "openai/clip-vit-base-patch32")
    batch_size = cfg.get("batch_size", 32)
    epochs = cfg.get("epochs", 15)
    num_workers = cfg.get("num_workers", 0)

    print(f"Device          : {device}")
    print(f"CLIP model      : {clip_model_name}")
    print(f"Dataset         : {dataset_path}")
    print(f"Batch size      : {batch_size}")
    print(f"Epochs          : {epochs}")
    print(f"LR (image)      : {cfg.get('lr_image', 1e-5)}")
    print(f"LR (text)       : {cfg.get('lr_text', 1e-5)}")
    print(f"LR (projection) : {cfg.get('lr_projection', 1e-4)}")
    print(f"Output          : {experiment_dir}")

    # ── load CLIP (trainable) ────────────────────────────────────────────
    print("\nLoading CLIP model ...")
    model = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    # Ensure all parameters are trainable
    for p in model.parameters():
        p.requires_grad = True
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── load datasets ────────────────────────────────────────────────────
    train_mri = MRIDataset(str(dataset_path), split="train", transform=None)
    val_mri = MRIDataset(str(dataset_path), split="val", transform=None)
    test_mri = MRIDataset(str(dataset_path), split="test", transform=None)

    train_ds = CLIPFinetuneDataset(train_mri)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=False,
    )

    print(f"Samples — train: {len(train_mri)}, val: {len(val_mri)}, "
          f"test: {len(test_mri)}")

    # ── optimizer + scheduler ────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    scheduler = build_scheduler(optimizer, epochs, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── training loop ────────────────────────────────────────────────────
    print(f"\nTraining ({epochs} epochs) ...")
    best_auroc = 0.0
    log_rows = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        train_loss = train_one_epoch(
            model, processor, train_loader, optimizer, scheduler,
            scaler, device)

        # Zero-shot validation
        val_labels, val_probs, val_sources = validate_zero_shot(
            model, processor, val_mri, batch_size, device)
        val_metrics = compute_metrics(val_labels, val_probs)
        val_acc = val_metrics["accuracy"]
        val_auroc = val_metrics["auroc"]

        epoch_time = time.time() - epoch_t0

        improved = ""
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), experiment_dir / "best_model.pth")
            improved = " *"

        print(f"  Epoch {epoch:>2d}/{epochs}  "
              f"loss={train_loss:.4f}  "
              f"v_acc={val_acc:.4f}  v_auroc={val_auroc:.4f}  "
              f"({epoch_time:.1f}s){improved}")

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_acc": round(val_acc, 5),
            "val_auroc": round(val_auroc, 5),
            "val_f1": round(val_metrics["f1"], 5),
            "epoch_time_sec": round(epoch_time, 1),
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

    # ── evaluate on test set with best checkpoint ────────────────────────
    print("\nEvaluating best checkpoint on test set ...")
    model.load_state_dict(
        torch.load(experiment_dir / "best_model.pth",
                   map_location=device, weights_only=True))

    test_labels, test_probs, test_sources = validate_zero_shot(
        model, processor, test_mri, batch_size, device)

    metrics = compute_metrics(test_labels, test_probs)
    metrics["model"] = clip_model_name
    metrics["method"] = "clip_finetune_contrastive"
    metrics["split"] = "test"
    metrics["n_samples"] = int(len(test_labels))
    metrics["best_val_auroc"] = round(best_auroc, 5)
    metrics["training_time_sec"] = round(elapsed, 2)
    metrics["epochs"] = epochs
    metrics["batch_size"] = batch_size
    metrics["lr_image"] = cfg.get("lr_image", 1e-5)
    metrics["lr_text"] = cfg.get("lr_text", 1e-5)
    metrics["lr_projection"] = cfg.get("lr_projection", 1e-4)
    metrics["per_source_accuracy"] = compute_per_source_accuracy(
        test_labels, test_probs, test_sources)

    # ── save artifacts ───────────────────────────────────────────────────
    with open(experiment_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        save_roc_curve(test_labels, test_probs,
                       "ROC — CLIP Fine-Tuned",
                       experiment_dir / "roc_curve.png")
    except ValueError:
        pass

    preds = (test_probs >= 0.5).astype(int)
    save_confusion_matrix(test_labels, preds,
                          "CM — CLIP Fine-Tuned",
                          experiment_dir / "confusion_matrix.png")

    # ── print summary ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("CLIP FINE-TUNING RESULTS (TEST SET)")
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
