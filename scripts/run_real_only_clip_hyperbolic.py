"""Real-Only Hyperbolic CLIP — Hyperbolic OOD Detection (Deep SVDD style).

Usage
-----
    python scripts/run_real_only_clip_hyperbolic.py
    python scripts/run_real_only_clip_hyperbolic.py --config configs/real_only_clip_hyperbolic.yaml

Trains the CLIP image encoder + hyperbolic projection head on real MRI images only.
The model learns a compact representation centered around the Fréchet mean of real
embeddings in the Poincaré ball.  At test time, synthetic images produce larger
hyperbolic anomaly scores and are detected as out-of-distribution.
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
    """Extract tensor from CLIP output."""
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output if hasattr(output, "pooler_output") else output[0]


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


# ── hyperbolic projection head ───────────────────────────────────────────────

class HyperbolicProjectionHead(nn.Module):
    """Projection head: CLIP embedding → Poincaré ball.

    Architecture: Linear(512 → 512) → ReLU → Linear(512 → projection_dim)
    Then: normalize → scale → expmap0

    This maps Euclidean CLIP embeddings to hyperbolic space.
    """

    def __init__(self, input_dim: int = 512, projection_dim: int = 256,
                 curvature: float = 1.0, scale: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim),
        )
        # Xavier initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.ball = geoopt.PoincareBall(c=curvature)
        self.scale = scale
        self.curvature = curvature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map Euclidean vectors to the Poincaré ball.

        Parameters
        ----------
        x : (B, D) — Euclidean CLIP embeddings

        Returns
        -------
        h : (B, projection_dim) — points on the Poincaré ball
        """
        z_e = self.net(x)
        # Normalize and scale to keep vectors comfortably inside the ball
        z_e = nn.functional.normalize(z_e, dim=-1) * self.scale
        # Exponential map at the origin
        z_h = self.ball.expmap0(z_e)
        return z_h


# ── hyperbolic distance utilities ────────────────────────────────────────────

def hyperbolic_distance(x, center, ball):
    """Compute hyperbolic distance from each point to the center.

    Parameters
    ----------
    x : (N, D) — points on the Poincaré ball
    center : (D,) — center point on the Poincaré ball
    ball : geoopt.PoincareBall

    Returns
    -------
    dist : (N,) — distances to center
    """
    # Expand center to match batch: (1, D)
    center_exp = center.unsqueeze(0)
    # Compute distance: ball.dist returns (N,) when shapes are (N, D) and (1, D)
    distances = ball.dist(x, center_exp)
    # Ensure output is 1D (handle various geoopt return shapes)
    return distances.view(-1)


# ── Fréchet mean (Riemannian mean) ───────────────────────────────────────────

def frechet_mean_iterative(points, curvature, max_iter=100, tol=1e-7):
    """Compute Fréchet mean on Poincaré ball via iterative algorithm.

    Uses gradient descent in the tangent space to find the point that
    minimizes the sum of squared hyperbolic distances.

    Parameters
    ----------
    points : (N, D) — points on the Poincaré ball (any device)
    curvature : float — Poincaré ball curvature
    max_iter : int — maximum iterations
    tol : float — convergence tolerance

    Returns
    -------
    mean : (D,) — Fréchet mean point (same device as input)
    """
    # Create ball on the same device as points
    ball = geoopt.PoincareBall(c=curvature)
    
    # Initialize at the Euclidean mean projected onto the ball
    mean = points.mean(dim=0)
    # Project to ensure it's inside the ball (with some margin)
    mean_norm = mean.norm()
    max_norm = 1.0 / math.sqrt(curvature) - 1e-5
    if mean_norm > max_norm:
        mean = mean * (max_norm / mean_norm)

    for _ in range(max_iter):
        # Compute logarithmic map from current mean to all points
        # log_mean(points) gives tangent vectors at mean pointing to points
        tangent_vectors = ball.logmap(mean.unsqueeze(0), points)  # (N, D)

        # The gradient is the mean of tangent vectors (pointing to the Fréchet mean)
        gradient = tangent_vectors.mean(dim=0)  # (D,)

        # Check convergence
        grad_norm = gradient.norm().item()
        if grad_norm < tol:
            break

        # Move in the direction of the gradient using exponential map
        # Use a small step size for stability
        step_size = min(1.0, 0.5 / (grad_norm + 1e-8))
        mean = ball.expmap(mean, gradient * step_size)

        # Ensure mean stays inside the ball
        mean_norm = mean.norm()
        if mean_norm > max_norm:
            mean = mean * (max_norm / mean_norm)

    return mean


# ── real-only dataset ────────────────────────────────────────────────────────

class RealOnlyDataset(Dataset):
    """Dataset wrapper that yields only real images (label=0)."""

    def __init__(self, mri_dataset: MRIDataset):
        self.samples = []
        self.sources = []
        for i, (path, label) in enumerate(mri_dataset.samples):
            if label == 0:  # Real only
                self.samples.append((path, label))
                self.sources.append(mri_dataset.sources[i])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return image, label, self.sources[idx]


def collate_fn(batch):
    """Custom collate — keeps PIL images as a list."""
    images, labels, sources = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(sources)


# ── full dataset for evaluation ──────────────────────────────────────────────

class FullDataset(Dataset):
    """Dataset wrapper for evaluation (real + synthetic)."""

    def __init__(self, mri_dataset: MRIDataset):
        self.mri_dataset = mri_dataset

    def __len__(self):
        return len(self.mri_dataset)

    def __getitem__(self, idx):
        path, label = self.mri_dataset.samples[idx]
        image = Image.open(path).convert("RGB")
        source = self.mri_dataset.sources[idx]
        return image, label, source


# ── center computation (Fréchet mean) ────────────────────────────────────────

@torch.no_grad()
def compute_hyperbolic_center(clip_model, processor, projection_head, dataset,
                               batch_size, device):
    """Compute Fréchet mean of real image embeddings in hyperbolic space."""
    clip_model.eval()
    projection_head.eval()

    embeddings = []
    n = len(dataset)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pil_images = []
        for idx in range(start, end):
            img, _, _ = dataset[idx]
            pil_images.append(img)

        inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        image_features = _to_tensor(clip_model.get_image_features(**inputs))
        projected = projection_head(image_features)
        embeddings.append(projected.cpu())

    all_embeddings = torch.cat(embeddings, dim=0)

    # Compute Fréchet mean in hyperbolic space (on CPU to avoid memory issues)
    center = frechet_mean_iterative(all_embeddings, projection_head.curvature)
    return center


# ── Hyperbolic SVDD loss ─────────────────────────────────────────────────────

def hyperbolic_svdd_loss(embeddings, center, ball):
    """Hyperbolic center-distance loss: d_h(f(x), c)^2."""
    distances = hyperbolic_distance(embeddings, center, ball)
    distances_sq = distances ** 2
    return distances_sq.mean()


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(labels, scores, threshold):
    """Compute metrics using anomaly scores.

    Note: higher score = more anomalous = predicted synthetic (label=1)
    """
    preds = (scores > threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }

    # For AUROC/AUPRC: higher score should indicate positive class (synthetic=1)
    try:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        metrics["auroc"] = 0.0
    try:
        metrics["auprc"] = float(average_precision_score(labels, scores))
    except ValueError:
        metrics["auprc"] = 0.0

    # PPV and NPV from confusion matrix
    cm = confusion_matrix(labels, preds)
    metrics["confusion_matrix"] = cm.tolist()

    TN, FP = cm[0]
    FN, TP = cm[1]
    metrics["PPV"] = round(TP / (TP + FP), 4) if (TP + FP) > 0 else 0.0
    metrics["NPV"] = round(TN / (TN + FN), 4) if (TN + FN) > 0 else 0.0

    return metrics


def compute_per_source_accuracy(labels, scores, sources, threshold):
    preds = (scores > threshold).astype(int)
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


# ── anomaly score computation ────────────────────────────────────────────────

@torch.no_grad()
def compute_anomaly_scores(clip_model, processor, projection_head, center,
                           dataset, batch_size, device):
    """Compute hyperbolic anomaly scores (distance to center) for all images."""
    clip_model.eval()
    projection_head.eval()
    center = center.to(device)

    all_scores = []
    all_labels = []
    all_sources = []

    n = len(dataset)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pil_images = []
        batch_labels = []
        batch_sources = []

        for idx in range(start, end):
            img, label, source = dataset[idx]
            pil_images.append(img)
            batch_labels.append(label)
            batch_sources.append(source)

        inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        image_features = _to_tensor(clip_model.get_image_features(**inputs))
        projected = projection_head(image_features)

        # Anomaly score = hyperbolic distance to center
        distances = hyperbolic_distance(projected, center, projection_head.ball)

        all_scores.extend(distances.cpu().numpy().tolist())
        all_labels.extend(batch_labels)
        all_sources.extend(batch_sources)

    return np.array(all_labels), np.array(all_scores), all_sources


# ── plotting ─────────────────────────────────────────────────────────────────

def save_loss_curve(log_rows, out_path):
    epochs = [r["epoch"] for r in log_rows]
    losses = [r["train_loss"] for r in log_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Hyperbolic SVDD Loss", fontsize=12)
    ax.set_title("Real-Only Hyperbolic CLIP Training Loss", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc_curve(labels, scores, title, out_path):
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_val = roc_auc_score(labels, scores)
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
    ax.set_xticklabels(["Real", "Synthetic"])
    ax.set_yticklabels(["Real", "Synthetic"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(im)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_anomaly_distribution(real_scores, synth_scores, threshold, out_path):
    """Plot histogram of hyperbolic anomaly scores for real vs synthetic images."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(
        min(real_scores.min(), synth_scores.min()),
        max(real_scores.max(), synth_scores.max()),
        50
    )

    ax.hist(real_scores, bins=bins, alpha=0.7, label="Real", color="#2196F3", edgecolor="white")
    ax.hist(synth_scores, bins=bins, alpha=0.7, label="Synthetic", color="#FF5722", edgecolor="white")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.4f}")

    ax.set_xlabel("Hyperbolic Anomaly Score (distance to center)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Hyperbolic Anomaly Score Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── training ─────────────────────────────────────────────────────────────────

def train_one_epoch(clip_model, processor, projection_head, center,
                    dataloader, optimizer, scheduler, scaler, device):
    """Train one epoch with hyperbolic SVDD center-distance loss."""
    clip_model.train()
    projection_head.train()
    center = center.to(device)

    running_loss = 0.0
    n_batches = 0

    for images, labels, sources in dataloader:
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            image_features = _to_tensor(clip_model.get_image_features(**inputs))
            projected = projection_head(image_features)
            loss = hyperbolic_svdd_loss(projected, center, projection_head.ball)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-Only Hyperbolic CLIP — Hyperbolic OOD Detection")
    parser.add_argument("--config", type=str,
                        default="configs/real_only_clip_hyperbolic.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    experiment_dir = PROJECT_ROOT / cfg.get(
        "experiment_dir", "experiments/real_only_clip_hyperbolic")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = get_device()
    clip_model_name = cfg.get("clip_model_name", "openai/clip-vit-base-patch32")
    batch_size = cfg.get("batch_size", 32)
    epochs = cfg.get("epochs", 15)
    lr = float(cfg.get("learning_rate", 1e-5))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    num_workers = cfg.get("num_workers", 0)
    projection_dim = cfg.get("projection_dim", 256)
    curvature = float(cfg.get("curvature", 1.0))
    scale = float(cfg.get("scale", 0.1))
    threshold_percentile = cfg.get("threshold_percentile", 95)

    print(f"Device            : {device}")
    print(f"CLIP model        : {clip_model_name}")
    print(f"Dataset           : {dataset_path}")
    print(f"Batch size        : {batch_size}")
    print(f"Epochs            : {epochs}")
    print(f"Learning rate     : {lr}")
    print(f"Projection dim    : {projection_dim}")
    print(f"Curvature (c)     : {curvature}")
    print(f"Scale             : {scale}")
    print(f"Threshold %ile    : {threshold_percentile}")
    print(f"Output            : {experiment_dir}")

    # ── load CLIP (image encoder only) ───────────────────────────────────
    print("\nLoading CLIP model ...")
    clip_model = CLIPModel.from_pretrained(
        clip_model_name, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    # Freeze text encoder, keep image encoder trainable
    for param in clip_model.text_model.parameters():
        param.requires_grad = False
    for param in clip_model.text_projection.parameters():
        param.requires_grad = False
    for param in clip_model.vision_model.parameters():
        param.requires_grad = True
    for param in clip_model.visual_projection.parameters():
        param.requires_grad = True

    # ── build hyperbolic projection head ─────────────────────────────────
    projection_head = HyperbolicProjectionHead(
        input_dim=512, projection_dim=projection_dim,
        curvature=curvature, scale=scale).to(device)

    n_clip = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    n_proj = sum(p.numel() for p in projection_head.parameters())
    print(f"  CLIP trainable  : {n_clip:,}")
    print(f"  Projection      : {n_proj:,}")
    print(f"  Total           : {n_clip + n_proj:,}")

    # ── load datasets ────────────────────────────────────────────────────
    train_mri = MRIDataset(str(dataset_path), split="train", transform=None)
    val_mri = MRIDataset(str(dataset_path), split="val", transform=None)
    test_mri = MRIDataset(str(dataset_path), split="test", transform=None)

    # Real-only training set
    train_real = RealOnlyDataset(train_mri)
    val_real = RealOnlyDataset(val_mri)

    train_loader = DataLoader(
        train_real, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=False,
    )

    # Full datasets for evaluation
    val_full = FullDataset(val_mri)
    test_full = FullDataset(test_mri)

    print(f"Training (real only): {len(train_real)}")
    print(f"Validation (real)   : {len(val_real)}")
    print(f"Validation (full)   : {len(val_full)}")
    print(f"Test (full)         : {len(test_full)}")

    # ── compute initial hyperbolic center (Fréchet mean) ─────────────────
    print("\nComputing initial hyperbolic center (Fréchet mean) ...")
    center = compute_hyperbolic_center(
        clip_model, processor, projection_head, train_real, batch_size, device)
    print(f"  Center shape    : {center.shape}")
    print(f"  Center norm     : {center.norm().item():.4f}")
    print(f"  Max ball radius : {1.0 / math.sqrt(curvature):.4f}")

    # ── optimizer + scheduler ────────────────────────────────────────────
    param_groups = [
        {"params": [p for p in clip_model.parameters() if p.requires_grad], "lr": lr},
        {"params": projection_head.parameters(), "lr": lr * 10},  # Higher LR for projection
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    steps_per_epoch = math.ceil(len(train_real) / batch_size)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── training loop ────────────────────────────────────────────────────
    print(f"\nTraining ({epochs} epochs) ...")
    log_rows = []
    t0 = time.time()
    best_auroc = 0.0

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        train_loss = train_one_epoch(
            clip_model, processor, projection_head, center,
            train_loader, optimizer, scheduler, scaler, device)

        # Evaluate on validation set
        val_labels, val_scores, val_sources = compute_anomaly_scores(
            clip_model, processor, projection_head, center,
            val_full, batch_size, device)

        # Determine threshold from validation real scores
        val_real_scores = val_scores[val_labels == 0]
        threshold = np.percentile(val_real_scores, threshold_percentile)

        val_metrics = compute_metrics(val_labels, val_scores, threshold)
        val_auroc = val_metrics["auroc"]

        epoch_time = time.time() - epoch_t0

        improved = ""
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({
                "clip_model": clip_model.state_dict(),
                "projection_head": projection_head.state_dict(),
                "center": center,
                "threshold": threshold,
                "curvature": curvature,
                "scale": scale,
            }, experiment_dir / "best_model.pth")
            improved = " *"

        print(f"  Epoch {epoch:>2d}/{epochs}  "
              f"loss={train_loss:.4f}  "
              f"v_auroc={val_auroc:.4f}  v_acc={val_metrics['accuracy']:.4f}  "
              f"thresh={threshold:.4f}  ({epoch_time:.1f}s){improved}")

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_auroc": round(val_auroc, 5),
            "val_accuracy": round(val_metrics["accuracy"], 5),
            "threshold": round(threshold, 5),
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

    # Save loss curve
    save_loss_curve(log_rows, experiment_dir / "loss_curve.png")

    # ── evaluate on test set with best checkpoint ────────────────────────
    print("\nEvaluating best checkpoint on test set ...")
    ckpt = torch.load(experiment_dir / "best_model.pth",
                      map_location=device, weights_only=False)
    clip_model.load_state_dict(ckpt["clip_model"])
    projection_head.load_state_dict(ckpt["projection_head"])
    center = ckpt["center"]
    threshold = ckpt["threshold"]

    # Save center embedding
    torch.save(center, experiment_dir / "center_embedding.pt")

    test_labels, test_scores, test_sources = compute_anomaly_scores(
        clip_model, processor, projection_head, center,
        test_full, batch_size, device)

    # Separate scores for visualization
    real_test_scores = test_scores[test_labels == 0]
    synth_test_scores = test_scores[test_labels == 1]

    # Save anomaly distribution plot
    save_anomaly_distribution(
        real_test_scores, synth_test_scores, threshold,
        experiment_dir / "anomaly_score_distribution.png")

    # Compute metrics
    metrics = compute_metrics(test_labels, test_scores, threshold)
    metrics["model"] = clip_model_name
    metrics["method"] = "real_only_clip_hyperbolic_svdd"
    metrics["split"] = "test"
    metrics["n_samples"] = int(len(test_labels))
    metrics["n_real"] = int((test_labels == 0).sum())
    metrics["n_synthetic"] = int((test_labels == 1).sum())
    metrics["best_val_auroc"] = round(best_auroc, 5)
    metrics["training_time_sec"] = round(elapsed, 2)
    metrics["epochs"] = epochs
    metrics["batch_size"] = batch_size
    metrics["learning_rate"] = lr
    metrics["projection_dim"] = projection_dim
    metrics["curvature"] = curvature
    metrics["scale"] = scale
    metrics["threshold"] = round(threshold, 5)
    metrics["threshold_percentile"] = threshold_percentile
    metrics["per_source_accuracy"] = compute_per_source_accuracy(
        test_labels, test_scores, test_sources, threshold)

    # Score statistics
    metrics["real_score_mean"] = round(float(real_test_scores.mean()), 5)
    metrics["real_score_std"] = round(float(real_test_scores.std()), 5)
    metrics["synth_score_mean"] = round(float(synth_test_scores.mean()), 5)
    metrics["synth_score_std"] = round(float(synth_test_scores.std()), 5)

    # ── save artifacts ───────────────────────────────────────────────────
    with open(experiment_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        save_roc_curve(test_labels, test_scores,
                       "ROC — Real-Only Hyperbolic CLIP",
                       experiment_dir / "roc_curve.png")
    except ValueError:
        pass

    preds = (test_scores > threshold).astype(int)
    save_confusion_matrix(test_labels, preds,
                          "CM — Real-Only Hyperbolic CLIP",
                          experiment_dir / "confusion_matrix.png")

    # ── print summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("REAL-ONLY HYPERBOLIC CLIP (HYPERBOLIC OOD) RESULTS — TEST SET")
    print(f"{'='*60}")
    print(f"  Curvature     : {curvature}")
    print(f"  Scale         : {scale}")
    print(f"  Threshold     : {threshold:.4f} (p{threshold_percentile})")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  F1            : {metrics['f1']:.4f}")
    print(f"  AUROC         : {metrics['auroc']:.4f}")
    print(f"  AUPRC         : {metrics['auprc']:.4f}")
    print(f"  PPV           : {metrics['PPV']:.4f}")
    print(f"  NPV           : {metrics['NPV']:.4f}")
    print(f"\nHyperbolic score statistics:")
    print(f"  Real mean±std : {metrics['real_score_mean']:.4f} ± {metrics['real_score_std']:.4f}")
    print(f"  Synth mean±std: {metrics['synth_score_mean']:.4f} ± {metrics['synth_score_std']:.4f}")
    print(f"\nPer-source accuracy:")
    for src, acc in metrics["per_source_accuracy"].items():
        print(f"    {src:<15s} {acc:.4f}")
    print(f"\nResults saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
