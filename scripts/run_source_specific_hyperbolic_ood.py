"""Source-Specific Hyperbolic CLIP OOD experiments.

Trains one-class Hyperbolic SVDD models for domains: Real, GAN, LDM, MLS.
Evaluation uses fixed, reproducible, class-balanced (Real=Fake) val/test subsets
with proportional fake stratification across GAN/LDM/MLS.
"""

import argparse
import csv
import json
import math
import os
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers import CLIPModel, CLIPProcessor

import geoopt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.source_specific_ood import (  # noqa: E402
    ALL_DOMAINS,
    build_loader_generator,
    load_or_create_manifest,
    path_to_label_source,
    set_global_determinism,
    worker_init_fn,
)


def _to_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output if hasattr(output, "pooler_output") else output[0]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HyperbolicProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 512, projection_dim: int = 256, curvature: float = 1.0, scale: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.ball = geoopt.PoincareBall(c=curvature)
        self.curvature = curvature
        # MERU-style learnable scale.
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.net(x)
        z_e = nn.functional.normalize(z_e, dim=-1) * self.scale.abs()
        z_h = self.ball.expmap0(z_e)
        return z_h


class ImagePathDataset(Dataset):
    def __init__(self, dataset_root: Path, rel_paths):
        self.dataset_root = dataset_root
        self.rel_paths = list(rel_paths)

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        rel = self.rel_paths[idx]
        abs_path = self.dataset_root / rel
        image = Image.open(abs_path).convert("RGB")
        label, source = path_to_label_source(rel)
        return image, label, source, rel


def collate_fn(batch):
    images, labels, sources, rel_paths = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(sources), list(rel_paths)


def hyperbolic_distance(x, center, ball):
    center_exp = center.unsqueeze(0)
    return ball.dist(x, center_exp).view(-1)


def frechet_mean_iterative(points, curvature, max_iter=100, tol=1e-7):
    ball = geoopt.PoincareBall(c=curvature)
    mean = points.mean(dim=0)

    mean_norm = mean.norm()
    max_norm = 1.0 / math.sqrt(curvature) - 1e-5
    if mean_norm > max_norm:
        mean = mean * (max_norm / mean_norm)

    for _ in range(max_iter):
        tangent = ball.logmap(mean.unsqueeze(0), points)
        grad = tangent.mean(dim=0)
        grad_norm = grad.norm().item()
        if grad_norm < tol:
            break

        step_size = min(1.0, 0.5 / (grad_norm + 1e-8))
        mean = ball.expmap(mean, grad * step_size)
        mean_norm = mean.norm()
        if mean_norm > max_norm:
            mean = mean * (max_norm / mean_norm)

    return mean


@torch.no_grad()
def compute_hyperbolic_center(clip_model, processor, projection_head, dataset, batch_size, device):
    clip_model.eval()
    projection_head.eval()
    embs = []

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        imgs = [dataset[i][0] for i in range(start, end)]
        inputs = processor(images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = _to_tensor(clip_model.get_image_features(**inputs))
        proj = projection_head(feats)
        embs.append(proj.cpu())

    all_embs = torch.cat(embs, dim=0)
    return frechet_mean_iterative(all_embs, projection_head.curvature)


def hyperbolic_svdd_loss(embeddings, center, ball):
    return (hyperbolic_distance(embeddings, center, ball) ** 2).mean()


@torch.no_grad()
def compute_anomaly_scores(clip_model, processor, projection_head, center, dataset, batch_size, device):
    clip_model.eval()
    projection_head.eval()
    center = center.to(device)

    scores, labels, sources, ids = [], [], [], []
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        imgs, lbs, srcs, rels = [], [], [], []
        for i in range(start, end):
            img, lb, src, rel = dataset[i]
            imgs.append(img)
            lbs.append(lb)
            srcs.append(src)
            rels.append(rel)

        inputs = processor(images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = _to_tensor(clip_model.get_image_features(**inputs))
        proj = projection_head(feats)
        dist = hyperbolic_distance(proj, center, projection_head.ball)

        scores.extend(dist.cpu().numpy().tolist())
        labels.extend(lbs)
        sources.extend(srcs)
        ids.extend(rels)

    return np.array(labels), np.array(scores), sources, ids


def predict_from_threshold(scores: np.ndarray, threshold: float, fake_positive_if_high: bool):
    if fake_positive_if_high:
        return (scores > threshold).astype(int)
    return (scores < threshold).astype(int)


def compute_metrics(labels, scores, threshold, fake_positive_if_high):
    preds = predict_from_threshold(scores, threshold, fake_positive_if_high)
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }

    rank_scores = scores if fake_positive_if_high else -scores
    try:
        out["auroc"] = float(roc_auc_score(labels, rank_scores))
    except ValueError:
        out["auroc"] = 0.0
    try:
        out["auprc"] = float(average_precision_score(labels, rank_scores))
    except ValueError:
        out["auprc"] = 0.0

    cm = confusion_matrix(labels, preds)
    out["confusion_matrix"] = cm.tolist()
    tn, fp = cm[0]
    fn, tp = cm[1]
    out["specificity"] = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
    out["sensitivity"] = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    out["PPV"] = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    out["NPV"] = round(tn / (tn + fn), 4) if (tn + fn) > 0 else 0.0
    return out


def compute_per_source_accuracy(labels, scores, sources, threshold, fake_positive_if_high):
    preds = predict_from_threshold(scores, threshold, fake_positive_if_high)
    src_correct = defaultdict(int)
    src_total = defaultdict(int)
    for p, y, s in zip(preds, labels, sources):
        src_total[s] += 1
        if p == y:
            src_correct[s] += 1
    return {k: round(src_correct[k] / src_total[k], 4) for k in sorted(src_total)}


def calibrate_threshold(labels, scores, fake_positive_if_high):
    uniq = np.unique(scores)
    if len(uniq) < 10:
        candidates = uniq
    else:
        q = np.linspace(0.01, 0.99, 200)
        candidates = np.quantile(scores, q)

    best_f1 = {"threshold": float(candidates[0]), "f1": -1.0}
    best_j = {"threshold": float(candidates[0]), "youden_j": -2.0}

    for th in candidates:
        preds = predict_from_threshold(scores, float(th), fake_positive_if_high)
        f1 = float(f1_score(labels, preds, zero_division=0))
        rec = float(recall_score(labels, preds, zero_division=0))
        spe = float(recall_score(labels, preds, pos_label=0, zero_division=0))
        j = rec + spe - 1.0

        if f1 > best_f1["f1"]:
            best_f1 = {"threshold": float(th), "f1": f1}
        if j > best_j["youden_j"]:
            best_j = {"threshold": float(th), "youden_j": j}

    return best_f1, best_j


def build_optimizer(clip_model, projection_head, cfg):
    lr_image = float(cfg.get("lr_image", 1e-5))
    lr_proj = float(cfg.get("lr_projection", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-4))

    image_params = []
    clip_proj_params = []
    other_params = []

    for name, param in clip_model.named_parameters():
        if not param.requires_grad:
            continue
        if "visual_projection" in name:
            clip_proj_params.append(param)
        elif "vision_model" in name:
            image_params.append(param)
        else:
            other_params.append(param)

    groups = [
        {"params": image_params, "lr": lr_image},
        {"params": clip_proj_params, "lr": lr_proj},
        {"params": other_params, "lr": lr_proj},
        {"params": list(projection_head.parameters()), "lr": lr_proj},
    ]
    groups = [g for g in groups if len(g["params"]) > 0]
    return geoopt.optim.RiemannianAdam(groups, weight_decay=wd)


def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs=2):
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    sched1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
    )
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[sched1, sched2], milestones=[warmup_steps]
    )


def save_loss_curve(rows, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hyperbolic SVDD Loss")
    ax.set_title("Source-Specific Hyperbolic OOD Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc_curve(labels, scores, title, out_path, fake_positive_if_high):
    rank_scores = scores if fake_positive_if_high else -scores
    fpr, tpr, _ = roc_curve(labels, rank_scores)
    auc_val = roc_auc_score(labels, rank_scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUROC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_pr_curve(labels, scores, title, out_path, fake_positive_if_high):
    rank_scores = scores if fake_positive_if_high else -scores
    precision, recall, _ = precision_recall_curve(labels, rank_scores)
    ap = average_precision_score(labels, rank_scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label=f"AUPRC = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
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
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
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


def save_score_distribution(real_scores, fake_scores, default_th, calibrated_th, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(min(real_scores.min(), fake_scores.min()), max(real_scores.max(), fake_scores.max()), 60)
    ax.hist(real_scores, bins=bins, alpha=0.65, label="Real", color="#2196F3", edgecolor="white")
    ax.hist(fake_scores, bins=bins, alpha=0.65, label="Fake", color="#FF5722", edgecolor="white")
    ax.axvline(default_th, color="black", linestyle="--", linewidth=2, label=f"Default th={default_th:.4f}")
    ax.axvline(calibrated_th, color="#2E7D32", linestyle="-.", linewidth=2, label=f"Calibrated th={calibrated_th:.4f}")
    ax.set_xlabel("Hyperbolic anomaly score (distance to center)")
    ax.set_ylabel("Count")
    ax.set_title("Hyperbolic Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_one_epoch(clip_model, processor, projection_head, center, dataloader, optimizer, scheduler, scaler, device):
    clip_model.train()
    projection_head.train()
    center = center.to(device)
    running, n_batches = 0.0, 0

    for images, _, _, _ in dataloader:
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            feats = _to_tensor(clip_model.get_image_features(**inputs))
            proj = projection_head(feats)
            loss = hyperbolic_svdd_loss(proj, center, projection_head.ball)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(clip_model.parameters()) + list(projection_head.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        running += loss.item()
        n_batches += 1

    return running / max(n_batches, 1)


def run_domain(cfg, domain: str):
    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    experiment_dir = PROJECT_ROOT / cfg.get("experiment_dir", "experiments/source_specific_ood_hyperbolic")
    run_dir = experiment_dir / f"domain_{domain}"
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    set_global_determinism(seed)
    device = get_device()

    clip_model_name = cfg.get("clip_model_name", "openai/clip-vit-base-patch16")
    batch_size = int(cfg.get("batch_size", 32))
    epochs = int(cfg.get("epochs", 25))
    projection_dim = int(cfg.get("projection_dim", 256))
    curvature = float(cfg.get("curvature", 1.0))
    scale = float(cfg.get("scale", 0.1))
    warmup_epochs = int(cfg.get("warmup_epochs", 2))
    threshold_percentile = float(cfg.get("threshold_percentile", 95))
    threshold_mode = cfg.get("threshold_mode", "calibrated_f1")
    num_workers = int(cfg.get("num_workers", 0))

    manifest = load_or_create_manifest(run_dir / "split_manifest.json", dataset_path, domain, seed)
    train_ids = manifest["train_ids"]
    val_in_domain_ids = manifest["val_in_domain_ids"]
    val_eval_ids = manifest["val_eval_ids"]
    test_eval_ids = manifest["test_eval_ids"]

    if not train_ids or not val_in_domain_ids or not val_eval_ids or not test_eval_ids:
        raise RuntimeError(f"Domain {domain}: empty split detected in manifest")

    fake_positive_if_high = (domain == "Real")

    print(f"\n{'='*72}")
    print(f"Domain: {domain} | Geometry: Hyperbolic | Device: {device}")
    print(f"Train IDs: {len(train_ids)} | Val(in): {len(val_in_domain_ids)} | Val(eval): {len(val_eval_ids)} | Test(eval): {len(test_eval_ids)}")
    print(f"Split hash: {manifest['hashes']['global_split_hash']}")
    print(f"Fake-positive direction: {'score > threshold' if fake_positive_if_high else 'score < threshold'}")

    clip_model = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    for p in clip_model.text_model.parameters():
        p.requires_grad = False
    for p in clip_model.text_projection.parameters():
        p.requires_grad = False
    for p in clip_model.vision_model.parameters():
        p.requires_grad = True
    for p in clip_model.visual_projection.parameters():
        p.requires_grad = True

    projection_head = HyperbolicProjectionHead(
        input_dim=512,
        projection_dim=projection_dim,
        curvature=curvature,
        scale=scale,
    ).to(device)

    train_ds = ImagePathDataset(dataset_path, train_ids)
    val_in_domain_ds = ImagePathDataset(dataset_path, val_in_domain_ids)
    val_eval_ds = ImagePathDataset(dataset_path, val_eval_ids)
    test_eval_ds = ImagePathDataset(dataset_path, test_eval_ids)

    loader_gen = build_loader_generator(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        worker_init_fn=worker_init_fn(seed),
        generator=loader_gen,
    )

    center = compute_hyperbolic_center(clip_model, processor, projection_head, train_ds, batch_size, device)

    optimizer = build_optimizer(clip_model, projection_head, cfg)
    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    scheduler = build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_auroc = -1.0
    log_rows = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        tr_loss = train_one_epoch(
            clip_model, processor, projection_head, center,
            train_loader, optimizer, scheduler, scaler, device,
        )

        val_labels, val_scores, _, _ = compute_anomaly_scores(
            clip_model, processor, projection_head, center,
            val_eval_ds, batch_size, device,
        )
        _, val_in_scores, _, _ = compute_anomaly_scores(
            clip_model, processor, projection_head, center,
            val_in_domain_ds, batch_size, device,
        )

        if fake_positive_if_high:
            default_th = float(np.percentile(val_in_scores, threshold_percentile))
        else:
            default_th = float(np.percentile(val_in_scores, 100.0 - threshold_percentile))

        best_f1, best_j = calibrate_threshold(val_labels, val_scores, fake_positive_if_high)
        calibrated_th = best_f1["threshold"]
        active_th = calibrated_th if threshold_mode == "calibrated_f1" else default_th

        val_metrics = compute_metrics(val_labels, val_scores, active_th, fake_positive_if_high)

        improved = ""
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            torch.save(
                {
                    "clip_model": clip_model.state_dict(),
                    "projection_head": projection_head.state_dict(),
                    "center": center,
                    "default_threshold": default_th,
                    "calibrated_threshold": calibrated_th,
                    "calibrated_threshold_youden_j": best_j["threshold"],
                    "threshold_mode": threshold_mode,
                    "curvature": curvature,
                },
                run_dir / "best_model.pth",
            )
            improved = " *"

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(tr_loss, 6),
                "val_auroc": round(val_metrics["auroc"], 6),
                "val_f1": round(val_metrics["f1"], 6),
                "default_threshold": round(default_th, 6),
                "calibrated_threshold": round(calibrated_th, 6),
                "learned_scale_abs": round(float(projection_head.scale.detach().abs().item()), 6),
                "epoch_time_sec": round(time.time() - epoch_t0, 2),
            }
        )

        print(
            f"  Epoch {epoch:>2d}/{epochs} loss={tr_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} th={active_th:.4f} scale={projection_head.scale.detach().abs().item():.4f}{improved}"
        )

    elapsed = time.time() - t0

    with open(run_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    save_loss_curve(log_rows, run_dir / "loss_curve.png")

    ckpt = torch.load(run_dir / "best_model.pth", map_location=device, weights_only=False)
    clip_model.load_state_dict(ckpt["clip_model"])
    projection_head.load_state_dict(ckpt["projection_head"])
    center = ckpt["center"]
    default_th = float(ckpt["default_threshold"])
    calibrated_th = float(ckpt["calibrated_threshold"])
    calibrated_th_j = float(ckpt.get("calibrated_threshold_youden_j", calibrated_th))

    test_labels, test_scores, test_sources, _ = compute_anomaly_scores(
        clip_model, processor, projection_head, center,
        test_eval_ds, batch_size, device,
    )

    test_default = compute_metrics(test_labels, test_scores, default_th, fake_positive_if_high)
    test_calibrated = compute_metrics(test_labels, test_scores, calibrated_th, fake_positive_if_high)

    active_th = calibrated_th if threshold_mode == "calibrated_f1" else default_th
    active_metrics = test_calibrated if threshold_mode == "calibrated_f1" else test_default

    real_scores = test_scores[test_labels == 0]
    fake_scores = test_scores[test_labels == 1]

    save_score_distribution(real_scores, fake_scores, default_th, calibrated_th, run_dir / "score_distribution.png")
    save_roc_curve(test_labels, test_scores, f"ROC - {domain} Hyperbolic", run_dir / "roc_curve.png", fake_positive_if_high)
    save_pr_curve(test_labels, test_scores, f"PR - {domain} Hyperbolic", run_dir / "pr_curve.png", fake_positive_if_high)

    preds_default = predict_from_threshold(test_scores, default_th, fake_positive_if_high)
    preds_calibrated = predict_from_threshold(test_scores, calibrated_th, fake_positive_if_high)
    preds_active = predict_from_threshold(test_scores, active_th, fake_positive_if_high)
    save_confusion_matrix(test_labels, preds_default, f"CM - {domain} Hyperbolic (default)", run_dir / "confusion_matrix_default.png")
    save_confusion_matrix(test_labels, preds_calibrated, f"CM - {domain} Hyperbolic (calibrated)", run_dir / "confusion_matrix_calibrated.png")
    save_confusion_matrix(test_labels, preds_active, f"CM - {domain} Hyperbolic ({threshold_mode})", run_dir / "confusion_matrix.png")

    calibration_report = {
        "threshold_mode": threshold_mode,
        "default_threshold": default_th,
        "calibrated_threshold_f1": calibrated_th,
        "calibrated_threshold_youden_j": calibrated_th_j,
        "test_metrics_default": test_default,
        "test_metrics_calibrated": test_calibrated,
    }
    with open(run_dir / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2)

    metrics = {
        **active_metrics,
        "domain": domain,
        "geometry": "hyperbolic",
        "model": clip_model_name,
        "method": "source_specific_hyperbolic_svdd",
        "n_samples": int(len(test_labels)),
        "n_real": int((test_labels == 0).sum()),
        "n_fake": int((test_labels == 1).sum()),
        "training_time_sec": round(elapsed, 2),
        "epochs": epochs,
        "batch_size": batch_size,
        "projection_dim": projection_dim,
        "curvature": curvature,
        "initial_scale": scale,
        "learned_scale_abs": round(float(projection_head.scale.detach().abs().item()), 6),
        "threshold_mode": threshold_mode,
        "default_threshold": round(default_th, 6),
        "calibrated_threshold": round(calibrated_th, 6),
        "fake_positive_if_high": fake_positive_if_high,
        "split_hash": manifest["hashes"]["global_split_hash"],
        "split_manifest": "split_manifest.json",
        "per_source_accuracy": compute_per_source_accuracy(test_labels, test_scores, test_sources, active_th, fake_positive_if_high),
        "real_score_mean": round(float(real_scores.mean()), 6),
        "real_score_std": round(float(real_scores.std()), 6),
        "fake_score_mean": round(float(fake_scores.mean()), 6),
        "fake_score_std": round(float(fake_scores.std()), 6),
    }

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"  TEST ({domain}): acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
        f"auroc={metrics['auroc']:.4f} auprc={metrics['auprc']:.4f} split={metrics['split_hash'][:10]}"
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Source-Specific Hyperbolic OOD CLIP")
    parser.add_argument("--config", type=str, default="configs/source_specific_ood_hyperbolic.yaml")
    parser.add_argument("--domain", type=str, default=None, choices=ALL_DOMAINS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    domains = [args.domain] if args.domain else cfg.get("domains", ALL_DOMAINS)

    all_rows = []
    for domain in domains:
        all_rows.append(run_domain(cfg, domain))

    experiment_dir = PROJECT_ROOT / cfg.get("experiment_dir", "experiments/source_specific_ood_hyperbolic")
    if all_rows:
        out_csv = experiment_dir / "summary_results.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "domain", "geometry", "accuracy", "precision", "recall", "f1", "auroc", "auprc",
                    "threshold_mode", "default_threshold", "calibrated_threshold", "split_hash",
                ],
            )
            writer.writeheader()
            for r in all_rows:
                writer.writerow(
                    {
                        "domain": r["domain"],
                        "geometry": r["geometry"],
                        "accuracy": round(r["accuracy"], 6),
                        "precision": round(r["precision"], 6),
                        "recall": round(r["recall"], 6),
                        "f1": round(r["f1"], 6),
                        "auroc": round(r["auroc"], 6),
                        "auprc": round(r["auprc"], 6),
                        "threshold_mode": r["threshold_mode"],
                        "default_threshold": r["default_threshold"],
                        "calibrated_threshold": r["calibrated_threshold"],
                        "split_hash": r["split_hash"],
                    }
                )

        print(f"\nSaved summary: {out_csv}")


if __name__ == "__main__":
    main()
