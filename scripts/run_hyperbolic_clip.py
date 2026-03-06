"""Hyperbolic CLIP — Poincaré Ball contrastive learning on MRI data.

Usage
-----
    python scripts/run_hyperbolic_clip.py
    python scripts/run_hyperbolic_clip.py --config configs/hyperbolic_clip.yaml

Projects CLIP embeddings into a Poincaré Ball manifold via learned projection
heads and trains with hyperbolic contrastive loss.  Inspired by the MERU
framework.
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

import geoopt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.mri_dataset import LABEL_MAP, MRIDataset  # noqa: E402


def _to_tensor(output):
    """Extract tensor from CLIP output (handles both raw tensor and model output)."""
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output if hasattr(output, "pooler_output") else output[0]


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


# ── Poincaré ball utilities ──────────────────────────────────────────────────

class HyperbolicProjection(nn.Module):
    """Projection head: Euclidean CLIP embedding → Poincaré ball.

    Pipeline:
        z_e = MLP(clip_embedding)
        z_e = L2_normalize(z_e) * scale
        z_h = exp_map_0(z_e)
    """

    def __init__(self, embed_dim: int = 512, curvature: float = 1.0,
                 scale: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ball = geoopt.PoincareBall(c=curvature)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map Euclidean vectors to the Poincaré ball.

        Parameters
        ----------
        x : (B, D) — Euclidean CLIP embeddings

        Returns
        -------
        h : (B, D) — points on the Poincaré ball
        """
        z_e = self.mlp(x)
        # Normalise and scale to keep vectors comfortably inside the ball
        z_e = nn.functional.normalize(z_e, dim=-1) * self.scale
        # Exponential map at the origin
        z_h = self.ball.expmap0(z_e)
        return z_h


def poincare_distance(x, y, ball):
    """Pairwise hyperbolic distance matrix.

    Parameters
    ----------
    x : (N, D) — points on the Poincaré ball
    y : (M, D) — points on the Poincaré ball
    ball : geoopt.PoincareBall

    Returns
    -------
    dist : (N, M) — pairwise distances
    """
    # Expand for pairwise computation:  (N, 1, D) vs (1, M, D)
    x_exp = x.unsqueeze(1)  # (N, 1, D)
    y_exp = y.unsqueeze(0)  # (1, M, D)
    return ball.dist(x_exp, y_exp)  # (N, M)


# ── hyperbolic contrastive loss ──────────────────────────────────────────────

def hyperbolic_contrastive_loss(image_h, text_h, ball, temperature=0.07):
    """Symmetric contrastive loss using hyperbolic distances.

    Parameters
    ----------
    image_h : (B, D) — image embeddings on Poincaré ball
    text_h  : (B, D) — text embeddings on Poincaré ball
    ball    : geoopt.PoincareBall
    temperature : float — scaling for logits

    Returns
    -------
    loss : scalar tensor
    """
    # Pairwise distances → similarities (negated distance)
    dist_matrix = poincare_distance(image_h, text_h, ball)  # (B, B)
    sim_matrix = -dist_matrix / temperature

    targets = torch.arange(image_h.size(0), device=image_h.device)

    loss_i2t = nn.functional.cross_entropy(sim_matrix, targets)
    loss_t2i = nn.functional.cross_entropy(sim_matrix.T, targets)
    return (loss_i2t + loss_t2i) / 2.0


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
    """Custom collate — keeps PIL images as a list."""
    images, texts, labels, sources = zip(*batch)
    return list(images), list(texts), torch.tensor(labels, dtype=torch.long), list(sources)


# ── optimizer (RiemannianAdam) ───────────────────────────────────────────────

def build_optimizer(clip_model, img_proj, txt_proj, cfg):
    """Create RiemannianAdam with per-component learning rates."""
    lr_image = cfg.get("lr_image", 1e-5)
    lr_text = cfg.get("lr_text", 1e-5)
    lr_proj = cfg.get("lr_projection", 1e-4)
    wd = cfg.get("weight_decay", 1e-4)

    image_params = []
    text_params = []
    clip_proj_params = []
    other_params = []

    for name, param in clip_model.named_parameters():
        if not param.requires_grad:
            continue
        if "visual_projection" in name or "text_projection" in name:
            clip_proj_params.append(param)
        elif "vision_model" in name:
            image_params.append(param)
        elif "text_model" in name:
            text_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": image_params, "lr": lr_image},
        {"params": text_params, "lr": lr_text},
        {"params": clip_proj_params, "lr": lr_proj},
        {"params": other_params, "lr": lr_proj},
        {"params": list(img_proj.parameters()), "lr": lr_proj},
        {"params": list(txt_proj.parameters()), "lr": lr_proj},
    ]
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    return geoopt.optim.RiemannianAdam(param_groups, weight_decay=wd)


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


# ── hyperbolic validation ───────────────────────────────────────────────────

@torch.no_grad()
def validate_hyperbolic(clip_model, processor, img_proj, txt_proj,
                        dataset, batch_size, device):
    """Classify using hyperbolic distance to class prompt embeddings.

    Returns labels, probs (P(fake)), sources.
    """
    clip_model.eval()
    img_proj.eval()
    txt_proj.eval()

    # Build class text embeddings in hyperbolic space
    prompts = [PROMPT_MAP[0], PROMPT_MAP[1]]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_features = _to_tensor(clip_model.get_text_features(**text_inputs))
    class_h = txt_proj(text_features)  # (2, D) on Poincaré ball

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

        image_features = _to_tensor(clip_model.get_image_features(**img_inputs))
        image_h = img_proj(image_features)  # (B, D) on Poincaré ball

        # Distance to each class prompt: (B, 2)
        dists = poincare_distance(image_h, class_h, img_proj.ball)

        # Similarity = -distance → softmax → P(fake)
        sims = -dists
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
    axes[0].set_title("Hyperbolic Contrastive Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [r["val_acc"] for r in log_rows], label="val accuracy")
    axes[1].set_title("Validation Accuracy (Hyperbolic)")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, [r["val_auroc"] for r in log_rows], label="val AUROC")
    axes[2].set_title("Validation AUROC (Hyperbolic)")
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

def train_one_epoch(clip_model, processor, img_proj, txt_proj,
                    dataloader, optimizer, scheduler, scaler, device):
    """Train one epoch with hyperbolic contrastive loss + mixed precision."""
    clip_model.train()
    img_proj.train()
    txt_proj.train()
    running_loss = 0.0
    n_batches = 0

    for images, texts, labels, sources in dataloader:
        img_inputs = processor(images=images, return_tensors="pt", padding=True)
        txt_inputs = processor(text=texts, return_tensors="pt", padding=True,
                               truncation=True)
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            # CLIP embeddings (Euclidean)
            image_e = _to_tensor(clip_model.get_image_features(**img_inputs))
            text_e = _to_tensor(clip_model.get_text_features(**txt_inputs))

            # Project to Poincaré ball
            image_h = img_proj(image_e)
            text_h = txt_proj(text_e)

            loss = hyperbolic_contrastive_loss(
                image_h, text_h, img_proj.ball)

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
        description="Hyperbolic CLIP — MRI Real vs Synthetic")
    parser.add_argument("--config", type=str,
                        default="configs/hyperbolic_clip.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    experiment_dir = PROJECT_ROOT / cfg.get(
        "experiment_dir", "experiments/hyperbolic_clip")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = get_device()
    clip_model_name = cfg.get("clip_model_name", "openai/clip-vit-base-patch32")
    batch_size = cfg.get("batch_size", 32)
    epochs = cfg.get("epochs", 25)
    num_workers = cfg.get("num_workers", 0)
    curvature = cfg.get("manifold_curvature", 1.0)
    scale = cfg.get("scale", 0.1)

    print(f"Device          : {device}")
    print(f"CLIP model      : {clip_model_name}")
    print(f"Dataset         : {dataset_path}")
    print(f"Batch size      : {batch_size}")
    print(f"Epochs          : {epochs}")
    print(f"LR (image)      : {cfg.get('lr_image', 1e-5)}")
    print(f"LR (text)       : {cfg.get('lr_text', 1e-5)}")
    print(f"LR (projection) : {cfg.get('lr_projection', 1e-4)}")
    print(f"Curvature (c)   : {curvature}")
    print(f"Scale           : {scale}")
    print(f"Output          : {experiment_dir}")

    # ── load CLIP (trainable) ────────────────────────────────────────────
    print("\nLoading CLIP model ...")
    clip_model = CLIPModel.from_pretrained(
        clip_model_name, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    for p in clip_model.parameters():
        p.requires_grad = True

    # ── build hyperbolic projection heads ────────────────────────────────
    img_proj = HyperbolicProjection(
        embed_dim=512, curvature=curvature, scale=scale).to(device)
    txt_proj = HyperbolicProjection(
        embed_dim=512, curvature=curvature, scale=scale).to(device)

    n_clip = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    n_proj = sum(p.numel() for p in img_proj.parameters()) + \
             sum(p.numel() for p in txt_proj.parameters())
    print(f"  CLIP params   : {n_clip:,}")
    print(f"  Proj params   : {n_proj:,}")
    print(f"  Total         : {n_clip + n_proj:,}")

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
    optimizer = build_optimizer(clip_model, img_proj, txt_proj, cfg)
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
            clip_model, processor, img_proj, txt_proj,
            train_loader, optimizer, scheduler, scaler, device)

        # Hyperbolic validation
        val_labels, val_probs, val_sources = validate_hyperbolic(
            clip_model, processor, img_proj, txt_proj,
            val_mri, batch_size, device)
        val_metrics = compute_metrics(val_labels, val_probs)
        val_acc = val_metrics["accuracy"]
        val_auroc = val_metrics["auroc"]

        epoch_time = time.time() - epoch_t0

        improved = ""
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            # Save all trainable state: CLIP + projection heads
            torch.save({
                "clip_model": clip_model.state_dict(),
                "img_proj": img_proj.state_dict(),
                "txt_proj": txt_proj.state_dict(),
            }, experiment_dir / "best_model.pth")
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
    ckpt = torch.load(experiment_dir / "best_model.pth",
                      map_location=device, weights_only=True)
    clip_model.load_state_dict(ckpt["clip_model"])
    img_proj.load_state_dict(ckpt["img_proj"])
    txt_proj.load_state_dict(ckpt["txt_proj"])

    test_labels, test_probs, test_sources = validate_hyperbolic(
        clip_model, processor, img_proj, txt_proj,
        test_mri, batch_size, device)

    metrics = compute_metrics(test_labels, test_probs)
    metrics["model"] = clip_model_name
    metrics["method"] = "hyperbolic_clip_poincare"
    metrics["split"] = "test"
    metrics["n_samples"] = int(len(test_labels))
    metrics["best_val_auroc"] = round(best_auroc, 5)
    metrics["training_time_sec"] = round(elapsed, 2)
    metrics["epochs"] = epochs
    metrics["batch_size"] = batch_size
    metrics["lr_image"] = cfg.get("lr_image", 1e-5)
    metrics["lr_text"] = cfg.get("lr_text", 1e-5)
    metrics["lr_projection"] = cfg.get("lr_projection", 1e-4)
    metrics["manifold_curvature"] = curvature
    metrics["scale"] = scale
    metrics["per_source_accuracy"] = compute_per_source_accuracy(
        test_labels, test_probs, test_sources)

    # ── save artifacts ───────────────────────────────────────────────────
    with open(experiment_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        save_roc_curve(test_labels, test_probs,
                       "ROC — Hyperbolic CLIP",
                       experiment_dir / "roc_curve.png")
    except ValueError:
        pass

    preds = (test_probs >= 0.5).astype(int)
    save_confusion_matrix(test_labels, preds,
                          "CM — Hyperbolic CLIP",
                          experiment_dir / "confusion_matrix.png")

    # ── print summary ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("HYPERBOLIC CLIP RESULTS (TEST SET)")
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
