"""CLIP Zero-Shot Baseline — Real vs Synthetic MRI classification.

Usage
-----
    python scripts/run_clip_zero_shot.py
    python scripts/run_clip_zero_shot.py --config configs/clip_zero_shot.yaml

Evaluates a frozen CLIP model on the test split using prompt-ensembled
text embeddings.  No parameters are trained.
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

from src.datasets.mri_dataset import MRIDataset          # noqa: E402
from src.utils.prompt_templates import build_text_embeddings  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── metrics (same format as ResNet experiments) ──────────────────────────────

def compute_metrics(labels, probs):
    """Return dict matching ResNet output format."""
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


# ── inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_zero_shot(model, processor, real_emb, fake_emb, dataset,
                  batch_size, device):
    """Run batched CLIP zero-shot inference.

    Returns
    -------
    all_labels  : np.ndarray  — ground truth (0 = real, 1 = fake)
    all_probs   : np.ndarray  — P(fake) in [0, 1]
    all_sources : list[str]
    """
    model.eval()
    all_labels = []
    all_probs = []
    all_sources = []

    # Stack class embeddings: (2, D)  — row 0 = real, row 1 = fake
    class_embs = torch.stack([real_emb, fake_emb], dim=0)  # (2, D)

    n = len(dataset)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        # Collect raw PIL images and metadata for this batch
        pil_images = []
        batch_labels = []
        batch_sources = []
        for idx in range(start, end):
            path, label = dataset.samples[idx]
            img = Image.open(path).convert("RGB")
            pil_images.append(img)
            batch_labels.append(label)
            batch_sources.append(dataset.sources[idx])

        # CLIP preprocessing (handles resize, centre-crop, normalisation)
        inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        image_features = model.get_image_features(**inputs)  # (B, D)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity with each class
        sims = image_features @ class_embs.T  # (B, 2) — col0=real, col1=fake

        # P(fake) = softmax over [sim_real, sim_fake], take fake column
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
        description="CLIP Zero-Shot Baseline — MRI Real vs Synthetic")
    parser.add_argument("--config", type=str,
                        default="configs/clip_zero_shot.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    experiment_dir = PROJECT_ROOT / cfg.get(
        "experiment_dir", "experiments/clip_zero_shot")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    clip_model_name = cfg.get("clip_model_name", "openai/clip-vit-base-patch32")
    batch_size = cfg.get("batch_size", 64)

    print(f"Device          : {device}")
    print(f"CLIP model      : {clip_model_name}")
    print(f"Dataset         : {dataset_path}")
    print(f"Output          : {experiment_dir}")

    # ── load CLIP (frozen) ───────────────────────────────────────────────
    print("\nLoading CLIP model ...")
    model = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("  Model loaded and frozen.")

    # ── build text embeddings ────────────────────────────────────────────
    print("Building prompt-ensembled text embeddings ...")
    real_emb, fake_emb = build_text_embeddings(model, processor, device)
    print(f"  Embedding dim: {real_emb.shape[0]}")

    # ── load test dataset ────────────────────────────────────────────────
    # We pass transform=None placeholder — we bypass __getitem__ transforms
    # and use the CLIPProcessor directly on raw PIL images inside run_zero_shot
    test_ds = MRIDataset(str(dataset_path), split="test", transform=None)
    print(f"Test samples    : {len(test_ds)}")

    # ── run inference ────────────────────────────────────────────────────
    print("\nRunning zero-shot inference ...")
    t0 = time.time()
    labels, probs, sources = run_zero_shot(
        model, processor, real_emb, fake_emb, test_ds, batch_size, device)
    elapsed = time.time() - t0
    print(f"  Inference done ({elapsed:.1f}s)")

    # ── compute metrics ──────────────────────────────────────────────────
    metrics = compute_metrics(labels, probs)
    metrics["model"] = clip_model_name
    metrics["split"] = "test"
    metrics["n_samples"] = int(len(labels))
    metrics["inference_time_sec"] = round(elapsed, 2)
    metrics["per_source_accuracy"] = compute_per_source_accuracy(
        labels, probs, sources)

    # ── save artifacts ───────────────────────────────────────────────────
    with open(experiment_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        save_roc_curve(labels, probs,
                       "ROC — CLIP Zero-Shot",
                       experiment_dir / "roc_curve.png")
    except ValueError:
        pass

    preds = (probs >= 0.5).astype(int)
    save_confusion_matrix(labels, preds,
                          "CM — CLIP Zero-Shot",
                          experiment_dir / "confusion_matrix.png")

    # ── print summary ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("CLIP ZERO-SHOT RESULTS")
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
