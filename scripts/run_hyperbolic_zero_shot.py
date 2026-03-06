"""Hyperbolic Zero-Shot CLIP — Poincaré Ball inference without training.

Usage
-----
    python scripts/run_hyperbolic_zero_shot.py
    python scripts/run_hyperbolic_zero_shot.py --config configs/hyperbolic_zero_shot.yaml

Evaluates a frozen CLIP model with randomly initialised (frozen) projection
heads that map Euclidean embeddings into a Poincaré Ball.  Classification is
based on hyperbolic distance to class prompt embeddings.  No parameters are
trained.
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

import geoopt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.mri_dataset import MRIDataset  # noqa: E402
from src.utils.prompt_templates import build_text_embeddings  # noqa: E402


def _to_tensor(output):
    """Extract tensor from CLIP output (handles both raw tensor and model output)."""
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output if hasattr(output, "pooler_output") else output[0]


# ── text prompts (same as CLIP zero-shot) ────────────────────────────────────

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


# ── Poincaré ball projection head ────────────────────────────────────────────

class HyperbolicProjection(nn.Module):
    """Projection head: Euclidean CLIP embedding -> Poincaré ball.

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
        # Xavier uniform initialization
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.ball = geoopt.PoincareBall(c=curvature)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.mlp(x)
        z_e = nn.functional.normalize(z_e, dim=-1) * self.scale
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
    x_exp = x.unsqueeze(1)  # (N, 1, D)
    y_exp = y.unsqueeze(0)  # (1, M, D)
    return ball.dist(x_exp, y_exp)  # (N, M)


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


# ── plotting ─────────────────────────────────────────────────────────────────

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


# ── hyperbolic zero-shot inference ───────────────────────────────────────────

@torch.no_grad()
def run_hyperbolic_zero_shot(clip_model, processor, img_proj, txt_proj,
                             dataset, batch_size, device):
    """Classify using hyperbolic distance to class prompt embeddings.

    Both CLIP encoders and projection heads are frozen.

    Returns
    -------
    all_labels  : np.ndarray  — ground truth (0 = real, 1 = fake)
    all_probs   : np.ndarray  — P(fake) in [0, 1]
    all_sources : list[str]
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

        # Similarity = -distance -> softmax -> P(fake)
        sims = -dists
        probs_fake = torch.softmax(sims, dim=-1)[:, 1].cpu().numpy()

        all_labels.extend(batch_labels)
        all_probs.extend(probs_fake.tolist())
        all_sources.extend(batch_sources)

        if (start // batch_size) % 10 == 0:
            print(f"  Processed {end}/{n} images ...")

    return np.array(all_labels), np.array(all_probs), all_sources


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hyperbolic Zero-Shot CLIP — MRI Real vs Synthetic")
    parser.add_argument("--config", type=str,
                        default="configs/hyperbolic_zero_shot.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    experiment_dir = PROJECT_ROOT / cfg.get(
        "experiment_dir", "experiments/hyperbolic_zero_shot")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = get_device()
    clip_model_name = cfg.get("clip_model_name", "openai/clip-vit-base-patch32")
    batch_size = cfg.get("batch_size", 64)
    curvature = cfg.get("manifold_curvature", 1.0)
    scale = cfg.get("scale", 0.1)

    print(f"Device          : {device}")
    print(f"CLIP model      : {clip_model_name}")
    print(f"Dataset         : {dataset_path}")
    print(f"Curvature (c)   : {curvature}")
    print(f"Scale           : {scale}")
    print(f"Output          : {experiment_dir}")

    # ── load CLIP (frozen) ───────────────────────────────────────────────
    print("\nLoading CLIP model ...")
    clip_model = CLIPModel.from_pretrained(
        clip_model_name, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    print("  CLIP model loaded and frozen.")

    # ── build hyperbolic projection heads (random, frozen) ───────────────
    img_proj = HyperbolicProjection(
        embed_dim=512, curvature=curvature, scale=scale).to(device)
    txt_proj = HyperbolicProjection(
        embed_dim=512, curvature=curvature, scale=scale).to(device)

    img_proj.eval()
    txt_proj.eval()
    for param in img_proj.parameters():
        param.requires_grad = False
    for param in txt_proj.parameters():
        param.requires_grad = False

    n_proj = sum(p.numel() for p in img_proj.parameters()) + \
             sum(p.numel() for p in txt_proj.parameters())
    print(f"  Projection params (frozen): {n_proj:,}")

    # ── load test dataset ────────────────────────────────────────────────
    test_ds = MRIDataset(str(dataset_path), split="test", transform=None)
    print(f"Test samples    : {len(test_ds)}")

    # ── run inference ────────────────────────────────────────────────────
    print("\nRunning hyperbolic zero-shot inference ...")
    t0 = time.time()
    labels, probs, sources = run_hyperbolic_zero_shot(
        clip_model, processor, img_proj, txt_proj, test_ds, batch_size, device)
    elapsed = time.time() - t0
    print(f"  Inference done ({elapsed:.1f}s)")

    # ── compute metrics ──────────────────────────────────────────────────
    metrics = compute_metrics(labels, probs)
    metrics["model"] = clip_model_name
    metrics["method"] = "hyperbolic_zero_shot_poincare"
    metrics["split"] = "test"
    metrics["n_samples"] = int(len(labels))
    metrics["inference_time_sec"] = round(elapsed, 2)
    metrics["manifold_curvature"] = curvature
    metrics["scale"] = scale
    metrics["per_source_accuracy"] = compute_per_source_accuracy(
        labels, probs, sources)

    # ── save artifacts ───────────────────────────────────────────────────
    with open(experiment_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        save_roc_curve(labels, probs,
                       "ROC — Hyperbolic Zero-Shot CLIP",
                       experiment_dir / "roc_curve.png")
    except ValueError:
        pass

    preds = (probs >= 0.5).astype(int)
    save_confusion_matrix(labels, preds,
                          "CM — Hyperbolic Zero-Shot CLIP",
                          experiment_dir / "confusion_matrix.png")

    # ── print summary ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("HYPERBOLIC ZERO-SHOT CLIP RESULTS (TEST SET)")
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
