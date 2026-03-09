"""Cross-Generator × Backbone Sweep — CLIP (ViT) & ResNet Architectures.

Usage
-----
    # Run full sweep (all backbones × all generator splits)
    python scripts/run_cross_generator_sweep.py --config configs/cross_generator_backbone_sweep.yaml

    # Run a single backbone
    python scripts/run_cross_generator_sweep.py --config configs/cross_generator_backbone_sweep.yaml --backbone ViT-B-32

    # Run a single generator split
    python scripts/run_cross_generator_sweep.py --config configs/cross_generator_backbone_sweep.yaml --generator GAN

    # Run a specific combination
    python scripts/run_cross_generator_sweep.py --config configs/cross_generator_backbone_sweep.yaml --backbone ViT-B-32 --generator GAN

Train on ONE synthetic generator group (+ all real images), then test on the
UNSEEN generators (+ all real images).  Repeat across multiple CLIP ViT and
ResNet backbones, producing a full results matrix.
"""

import argparse
import csv
import gc
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.mri_dataset import LABEL_MAP, MRIDataset, get_transforms  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

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


PROMPT_MAP = {
    0: "a real brain MRI image",      # Real
    1: "a synthetic brain MRI image",  # Fake
}

REAL_SOURCES = ["cermep", "tcga", "upenn"]


def compute_metrics(labels, probs):
    """Compute standard binary classification metrics."""
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


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

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


def save_loss_curves(log_rows, out_path):
    epochs = [r["epoch"] for r in log_rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in log_rows], label="train")
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


# ═══════════════════════════════════════════════════════════════════════════
# Dataset helpers for cross-generator splits
# ═══════════════════════════════════════════════════════════════════════════

def build_datasets(dataset_path, train_gen_sources, real_sources):
    """Build train/val/test datasets for a cross-generator split.

    Train & Val: only ``train_gen_sources`` fakes + all real
    Test: everything EXCEPT ``train_gen_sources`` fakes (i.e. unseen generators + all real)
    """
    include_train = train_gen_sources + real_sources

    train_ds = MRIDataset(str(dataset_path), split="train",
                          transform=None,
                          include_sources=include_train)
    val_ds = MRIDataset(str(dataset_path), split="val",
                        transform=None,
                        include_sources=include_train)
    test_ds = MRIDataset(str(dataset_path), split="test",
                         transform=None,
                         exclude_sources=train_gen_sources)

    return train_ds, val_ds, test_ds


# ═══════════════════════════════════════════════════════════════════════════
# CLIP contrastive fine-tuning (ViT backbones)
# ═══════════════════════════════════════════════════════════════════════════

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


def clip_collate_fn(batch):
    images, texts, labels, sources = zip(*batch)
    return list(images), list(texts), torch.tensor(labels, dtype=torch.long), list(sources)


def _to_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output if hasattr(output, "pooler_output") else output[0]


def clip_contrastive_loss(image_embeds, text_embeds, logit_scale):
    scale = logit_scale.exp()
    logits_per_image = scale * image_embeds @ text_embeds.T
    logits_per_text = logits_per_image.T
    targets = torch.arange(len(image_embeds), device=image_embeds.device)
    loss_i2t = nn.functional.cross_entropy(logits_per_image, targets)
    loss_t2i = nn.functional.cross_entropy(logits_per_text, targets)
    return (loss_i2t + loss_t2i) / 2.0


def build_clip_optimizer(model, cfg):
    lr_image = float(cfg.get("lr_image", 1e-5))
    lr_text = float(cfg.get("lr_text", 1e-5))
    lr_proj = float(cfg.get("lr_projection", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-4))

    image_params, text_params, proj_params, other_params = [], [], [], []
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
            other_params.append(param)

    param_groups = [
        {"params": image_params, "lr": lr_image},
        {"params": text_params, "lr": lr_text},
        {"params": proj_params, "lr": lr_proj},
        {"params": other_params, "lr": lr_proj},
    ]
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    return torch.optim.AdamW(param_groups, weight_decay=wd)


@torch.no_grad()
def clip_zero_shot_eval(model, processor, mri_dataset, batch_size, device):
    """Zero-shot classification using the two class prompts."""
    model.eval()

    prompts = [PROMPT_MAP[0], PROMPT_MAP[1]]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_features = _to_tensor(model.get_text_features(**text_inputs))
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    all_labels, all_probs, all_sources = [], [], []
    n = len(mri_dataset)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pil_images, batch_labels, batch_sources = [], [], []
        for idx in range(start, end):
            path, label = mri_dataset.samples[idx]
            pil_images.append(Image.open(path).convert("RGB"))
            batch_labels.append(label)
            batch_sources.append(mri_dataset.sources[idx])

        img_inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        image_features = _to_tensor(model.get_image_features(**img_inputs))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        sims = image_features @ text_features.T
        probs_fake = torch.softmax(sims, dim=-1)[:, 1].cpu().numpy()

        all_labels.extend(batch_labels)
        all_probs.extend(probs_fake.tolist())
        all_sources.extend(batch_sources)

    return np.array(all_labels), np.array(all_probs), all_sources


def train_clip_experiment(model_id, train_mri, val_mri, test_mri,
                          clip_cfg, num_workers, fold_dir, device):
    """Full CLIP contrastive fine-tuning experiment."""
    from transformers import CLIPModel, CLIPProcessor

    batch_size = int(clip_cfg.get("batch_size", 32))
    epochs = int(clip_cfg.get("epochs", 15))

    print(f"    Loading CLIP: {model_id}")
    model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    for p in model.parameters():
        p.requires_grad = True
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable params: {n_params:,}")

    train_ds = CLIPFinetuneDataset(train_mri)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=clip_collate_fn, pin_memory=False,
    )

    optimizer = build_clip_optimizer(model, clip_cfg)
    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── training loop ────────────────────────────────────────────────
    best_auroc = 0.0
    log_rows = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0

        for images, texts, labels, sources in train_loader:
            img_inputs = processor(images=images, return_tensors="pt", padding=True)
            txt_inputs = processor(text=texts, return_tensors="pt", padding=True,
                                   truncation=True)
            img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
            txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                image_embeds = _to_tensor(model.get_image_features(**img_inputs))
                text_embeds = _to_tensor(model.get_text_features(**txt_inputs))
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                loss = clip_contrastive_loss(image_embeds, text_embeds,
                                             model.logit_scale)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        # validation (zero-shot on val set — same generator as train)
        val_labels, val_probs, _ = clip_zero_shot_eval(
            model, processor, val_mri, batch_size, device)
        val_m = compute_metrics(val_labels, val_probs)

        improved = ""
        if val_m["auroc"] > best_auroc:
            best_auroc = val_m["auroc"]
            torch.save(model.state_dict(), fold_dir / "best_model.pth")
            improved = " *"

        print(f"      Epoch {epoch:>2d}/{epochs}  loss={train_loss:.4f}  "
              f"v_acc={val_m['accuracy']:.4f}  v_auroc={val_m['auroc']:.4f}{improved}")

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_acc": round(val_m["accuracy"], 5),
            "val_auroc": round(val_m["auroc"], 5),
            "val_f1": round(val_m["f1"], 5),
        })

    elapsed = time.time() - t0
    print(f"    Training done ({elapsed / 60:.1f} min). Best val AUROC: {best_auroc:.4f}")

    # Save training log + loss curves
    if log_rows:
        with open(fold_dir / "training_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        save_loss_curves(log_rows, fold_dir / "loss_curve.png")

    # ── evaluate on test set (unseen generators) ─────────────────────
    print("    Evaluating on unseen generators ...")
    model.load_state_dict(
        torch.load(fold_dir / "best_model.pth",
                   map_location=device, weights_only=True))

    test_labels, test_probs, test_sources = clip_zero_shot_eval(
        model, processor, test_mri, batch_size, device)

    metrics = compute_metrics(test_labels, test_probs)
    metrics["per_source_accuracy"] = compute_per_source_accuracy(
        test_labels, test_probs, test_sources)
    metrics["best_val_auroc"] = round(best_auroc, 5)
    metrics["training_time_min"] = round(elapsed / 60, 2)

    # cleanup
    del model, processor, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()

    return metrics, test_labels, test_probs


# ═══════════════════════════════════════════════════════════════════════════
# ResNet binary classification
# ═══════════════════════════════════════════════════════════════════════════

def build_resnet_model(model_id: str, pretrained: bool = True):
    """Build a ResNet (50 or 101) with a binary classification head."""
    from torchvision import models

    if model_id == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = 2048
    elif model_id == "resnet101":
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        backbone = models.resnet101(weights=weights)
        in_features = 2048
    else:
        raise ValueError(f"Unsupported ResNet model_id: {model_id}")

    backbone.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
    )
    return backbone


def resnet_train_one_epoch(model, loader, criterion, optimizer, device):
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
def resnet_evaluate(model, loader, criterion, device):
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
    return epoch_loss, np.array(all_labels), np.array(all_probs)


def train_resnet_experiment(model_id, dataset_path, train_gen_sources,
                            real_sources, resnet_cfg, num_workers,
                            fold_dir, device):
    """Full ResNet binary classification experiment."""
    batch_size = int(resnet_cfg.get("batch_size", 32))
    epochs = int(resnet_cfg.get("epochs", 20))
    lr = float(resnet_cfg.get("learning_rate", 3e-4))
    wd = float(resnet_cfg.get("weight_decay", 1e-4))

    include_train = train_gen_sources + real_sources
    train_ds = MRIDataset(str(dataset_path), split="train",
                          transform=get_transforms("train"),
                          include_sources=include_train)
    val_ds = MRIDataset(str(dataset_path), split="val",
                        transform=get_transforms("val"),
                        include_sources=include_train)
    test_ds = MRIDataset(str(dataset_path), split="test",
                         transform=get_transforms("test"),
                         exclude_sources=train_gen_sources)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"    Loading {model_id} ...")
    model = build_resnet_model(model_id, pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable params: {n_params:,}")

    pos_weight = train_ds.get_class_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── training loop ────────────────────────────────────────────────
    best_auroc = 0.0
    log_rows = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_auroc = resnet_train_one_epoch(
            model, train_loader, criterion, optimizer, device)

        val_loss, val_labels, val_probs = resnet_evaluate(
            model, val_loader, criterion, device)
        val_m = compute_metrics(val_labels, val_probs)
        scheduler.step()

        improved = ""
        if val_m["auroc"] > best_auroc:
            best_auroc = val_m["auroc"]
            torch.save(model.state_dict(), fold_dir / "best_model.pth")
            improved = " *"

        print(f"      Epoch {epoch:>2d}/{epochs}  "
              f"t_loss={train_loss:.4f}  v_loss={val_loss:.4f}  "
              f"v_acc={val_m['accuracy']:.4f}  v_auroc={val_m['auroc']:.4f}{improved}")

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_acc": round(val_m["accuracy"], 5),
            "val_auroc": round(val_m["auroc"], 5),
            "val_f1": round(val_m["f1"], 5),
        })

    elapsed = time.time() - t0
    print(f"    Training done ({elapsed / 60:.1f} min). Best val AUROC: {best_auroc:.4f}")

    # Save training log + loss curves
    if log_rows:
        with open(fold_dir / "training_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        save_loss_curves(log_rows, fold_dir / "loss_curve.png")

    # ── evaluate on test set (unseen generators) ─────────────────────
    print("    Evaluating on unseen generators ...")
    model.load_state_dict(
        torch.load(fold_dir / "best_model.pth",
                   map_location=device, weights_only=True))

    _, test_labels, test_probs = resnet_evaluate(
        model, test_loader, criterion, device)

    metrics = compute_metrics(test_labels, test_probs)

    # Per-source accuracy
    test_sources = test_ds.sources
    metrics["per_source_accuracy"] = compute_per_source_accuracy(
        test_labels, test_probs, test_sources)
    metrics["best_val_auroc"] = round(best_auroc, 5)
    metrics["training_time_min"] = round(elapsed / 60, 2)

    # cleanup
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    return metrics, test_labels, test_probs


# ═══════════════════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Generator × Backbone Sweep")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--backbone", type=str, default=None,
                        help="Run only this backbone (e.g. ViT-B-32)")
    parser.add_argument("--generator", type=str, default=None,
                        help="Run only this generator group (e.g. GAN)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device()

    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    experiment_dir = PROJECT_ROOT / cfg.get(
        "experiment_dir", "experiments/cross_generator_sweep")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    generator_groups = cfg["generator_groups"]   # {name: [source_list]}
    real_sources = cfg.get("real_sources", REAL_SOURCES)
    backbones = cfg["backbones"]                 # [{name, type, model_id}]
    num_workers = int(cfg.get("num_workers", 0))

    clip_cfg = cfg.get("clip_training", {})
    resnet_cfg = cfg.get("resnet_training", {})

    # Optional filters
    if args.backbone:
        backbones = [b for b in backbones if b["name"] == args.backbone]
        if not backbones:
            print(f"ERROR: backbone '{args.backbone}' not found in config.")
            sys.exit(1)
    if args.generator:
        if args.generator not in generator_groups:
            print(f"ERROR: generator '{args.generator}' not found in config.")
            sys.exit(1)
        generator_groups = {args.generator: generator_groups[args.generator]}

    total = len(backbones) * len(generator_groups)
    print(f"Device           : {device}")
    print(f"Dataset          : {dataset_path}")
    print(f"Backbones        : {[b['name'] for b in backbones]}")
    print(f"Generator groups : {list(generator_groups.keys())}")
    print(f"Total experiments: {total}")
    print(f"Output           : {experiment_dir}")

    # ── summary collector ────────────────────────────────────────────
    summary_rows = []
    run_idx = 0

    for backbone in backbones:
        bb_name = backbone["name"]
        bb_type = backbone["type"]
        model_id = backbone["model_id"]

        for gen_name, gen_sources in generator_groups.items():
            run_idx += 1
            # Generator sources NOT used for training → they appear in test
            other_gen_sources = []
            for gn, gs in cfg["generator_groups"].items():
                if gn != gen_name:
                    other_gen_sources.extend(gs)

            fold_dir = experiment_dir / bb_name / f"train_{gen_name}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*65}")
            print(f"  [{run_idx}/{total}] Backbone: {bb_name}  |  Train on: {gen_name}")
            print(f"  Train fakes: {gen_sources}")
            print(f"  Test fakes : {other_gen_sources}")
            print(f"{'='*65}")

            # Build datasets
            train_mri, val_mri, test_mri = build_datasets(
                dataset_path, gen_sources, real_sources)

            n_train_real = sum(1 for s in train_mri.sources if s in real_sources)
            n_train_fake = len(train_mri) - n_train_real
            n_test_real = sum(1 for s in test_mri.sources if s in real_sources)
            n_test_fake = len(test_mri) - n_test_real

            print(f"  Train: {len(train_mri)} ({n_train_real} real, {n_train_fake} fake)")
            print(f"  Val  : {len(val_mri)}")
            print(f"  Test : {len(test_mri)} ({n_test_real} real, {n_test_fake} fake)")

            if len(train_mri) == 0 or len(test_mri) == 0:
                print("  SKIP — empty dataset")
                continue

            # Dispatch to appropriate training paradigm
            if bb_type == "clip":
                metrics, test_labels, test_probs = train_clip_experiment(
                    model_id, train_mri, val_mri, test_mri,
                    clip_cfg, num_workers, fold_dir, device)
            elif bb_type == "resnet":
                metrics, test_labels, test_probs = train_resnet_experiment(
                    model_id, dataset_path, gen_sources, real_sources,
                    resnet_cfg, num_workers, fold_dir, device)
            else:
                print(f"  SKIP — unknown backbone type: {bb_type}")
                continue

            # Save experiment results
            result_record = {
                "backbone": bb_name,
                "backbone_type": bb_type,
                "model_id": model_id,
                "train_generator": gen_name,
                "test_generators": [gn for gn in cfg["generator_groups"]
                                    if gn != gen_name],
                **metrics,
            }
            with open(fold_dir / "results.json", "w") as f:
                json.dump(result_record, f, indent=2)

            # Plots
            try:
                save_roc_curve(test_labels, test_probs,
                               f"ROC — {bb_name} (train {gen_name})",
                               fold_dir / "roc_curve.png")
            except ValueError:
                pass
            preds = (test_probs >= 0.5).astype(int)
            save_confusion_matrix(test_labels, preds,
                                  f"CM — {bb_name} (train {gen_name})",
                                  fold_dir / "confusion_matrix.png")

            print(f"\n  Results: Acc={metrics['accuracy']:.4f}  "
                  f"F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}")
            if "per_source_accuracy" in metrics:
                for src, acc in metrics["per_source_accuracy"].items():
                    print(f"    {src:<15s} {acc:.4f}")

            summary_rows.append({
                "backbone": bb_name,
                "train_gen": gen_name,
                "accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
                "auroc": round(metrics["auroc"], 4),
                "auprc": round(metrics["auprc"], 4),
            })

    # ── save summary CSV ─────────────────────────────────────────────
    if summary_rows:
        summary_path = experiment_dir / "summary_results.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

        # Print summary table
        print(f"\n\n{'='*80}")
        print("CROSS-GENERATOR × BACKBONE SWEEP — SUMMARY")
        print(f"{'='*80}")
        header = f"{'Backbone':<14s} {'Train Gen':<10s} {'Acc':>8s} {'F1':>8s} {'AUROC':>8s} {'AUPRC':>8s}"
        print(header)
        print("-" * len(header))
        for row in summary_rows:
            print(f"{row['backbone']:<14s} {row['train_gen']:<10s} "
                  f"{row['accuracy']:>8.4f} {row['f1']:>8.4f} "
                  f"{row['auroc']:>8.4f} {row['auprc']:>8.4f}")
        print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
