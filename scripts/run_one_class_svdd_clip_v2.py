from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    classification_report,
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
try:
    import open_clip
except ImportError:
    open_clip = None
import geoopt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.one_class_svdd_v2 import build_protocol_manifest, save_manifest
from src.utils.source_specific_ood import build_loader_generator, set_global_determinism, worker_init_fn


def _normalize_backbone_key(name: str) -> str:
    return str(name).strip().upper().replace("-", "").replace("_", "")


def _resolve_backbone_registry(cfg: dict) -> Tuple[Dict[str, Dict], str]:
    registry_cfg = cfg.get("backbone_registry", {}) or {}
    resolved: Dict[str, Dict] = {}

    for key, spec in registry_cfg.items():
        norm_key = _normalize_backbone_key(key)
        if not isinstance(spec, dict):
            continue
        model_name = spec.get("model_name")
        if not model_name:
            continue
        resolved[norm_key] = {
            "display_name": spec.get("display_name", norm_key),
            "model_name": model_name,
            "type": spec.get("type", "clip"),
            "open_clip_pretrained": spec.get("open_clip_pretrained", "openai"),
            "vision_encoder_mode": spec.get(
                "vision_encoder_mode",
                cfg.get("backbone", {}).get("vision_encoder_mode", "fine_tune"),
            ),
        }

    if not resolved:
        fallback_model_name = cfg.get("clip_model_name", cfg.get("backbone", {}).get("model_name", "openai/clip-vit-base-patch16"))
        resolved["B16"] = {
            "display_name": "Hyperbolic CLIP ViT B16",
            "model_name": fallback_model_name,
            "type": "clip",
            "open_clip_pretrained": "openai",
            "vision_encoder_mode": cfg.get("backbone", {}).get("vision_encoder_mode", "fine_tune"),
        }

    default_key = _normalize_backbone_key(cfg.get("default_backbone", "B16"))
    if default_key not in resolved:
        default_key = "B16" if "B16" in resolved else sorted(resolved.keys())[0]

    return resolved, default_key


def _parse_requested_backbones(backbones_arg: Sequence[str] | None) -> List[str]:
    if not backbones_arg:
        return []
    requested: List[str] = []
    seen = set()
    for token in backbones_arg:
        for part in str(token).split(","):
            norm = _normalize_backbone_key(part)
            if not norm or norm in seen:
                continue
            requested.append(norm)
            seen.add(norm)
    return requested


def _parse_requested_layers(layers_arg: Sequence[str] | None) -> List[int]:
    if not layers_arg:
        return []

    requested: List[int] = []
    seen = set()
    for token in layers_arg:
        for part in str(token).split(","):
            value = str(part).strip()
            if not value:
                continue
            try:
                layer_count = int(value)
            except ValueError as exc:
                raise ValueError(f"Invalid layer value '{value}'. Layers must be integers.") from exc
            if layer_count <= 0:
                raise ValueError(f"Invalid layer value '{value}'. Layers must be > 0.")
            if layer_count in seen:
                continue
            requested.append(layer_count)
            seen.add(layer_count)

    return requested


def _default_num_vit_layers(cfg: dict) -> int:
    return int(cfg.get("backbone", {}).get("num_vit_layers", 12))


def _apply_backbone_to_cfg(base_cfg: dict, backbone_spec: Dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["clip_model_name"] = backbone_spec["model_name"]
    backbone_cfg = dict(cfg.get("backbone", {}))
    backbone_cfg["model_name"] = backbone_spec["model_name"]
    backbone_cfg["type"] = backbone_spec.get("type", "clip")
    backbone_cfg["open_clip_pretrained"] = backbone_spec.get("open_clip_pretrained", "openai")
    backbone_cfg["vision_encoder_mode"] = backbone_spec.get("vision_encoder_mode", backbone_cfg.get("vision_encoder_mode", "fine_tune"))
    cfg["backbone"] = backbone_cfg
    return cfg


def _apply_layer_to_cfg(base_cfg: dict, num_vit_layers: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    backbone_cfg = dict(cfg.get("backbone", {}))
    backbone_cfg["num_vit_layers"] = int(num_vit_layers)
    cfg["backbone"] = backbone_cfg
    return cfg


def _resolve_optional_protocol_manifest(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_protocol_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    required_keys = {"test_sets", "cv_folds", "real_train"}
    missing = sorted(required_keys - set(manifest.keys()))
    if missing:
        raise RuntimeError(f"Protocol manifest missing required keys: {missing}")

    return manifest


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output if hasattr(output, "pooler_output") else output[0]


def _truncate_clip_vision_layers(clip_model: CLIPModel, num_vit_layers: int) -> None:
    layers = getattr(getattr(getattr(clip_model, "vision_model", None), "encoder", None), "layers", None)
    if layers is None:
        return

    n_total = len(layers)
    if num_vit_layers <= 0 or num_vit_layers >= n_total:
        return

    clip_model.vision_model.encoder.layers = nn.ModuleList(list(layers)[:num_vit_layers])
    if hasattr(clip_model, "config") and hasattr(clip_model.config, "vision_config"):
        clip_model.config.vision_config.num_hidden_layers = num_vit_layers
    if hasattr(clip_model, "vision_model") and hasattr(clip_model.vision_model, "config"):
        clip_model.vision_model.config.num_hidden_layers = num_vit_layers


def _load_image_backbone(
    backbone_name: str,
    backbone_type: str,
    open_clip_pretrained: str,
    device: torch.device,
    num_vit_layers: int | None = None,
):
    if backbone_type == "open_clip":
        if open_clip is None:
            raise ImportError("open_clip_torch is required for open_clip backbones. Install with: pip install open_clip_torch")

        # Prefer *-quickgelu model defs for OpenAI pretrained tags to avoid activation-mismatch warnings.
        candidate_names = [backbone_name]
        if str(open_clip_pretrained).lower() == "openai" and not backbone_name.lower().endswith("-quickgelu"):
            candidate_names = [f"{backbone_name}-quickgelu", backbone_name]

        last_err = None
        for candidate in candidate_names:
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name=candidate,
                    pretrained=open_clip_pretrained,
                )
                return model.to(device), preprocess
            except Exception as err:
                last_err = err

        if last_err is not None:
            raise last_err

        return model.to(device), preprocess

    model = CLIPModel.from_pretrained(backbone_name, use_safetensors=True).to(device)
    if isinstance(num_vit_layers, int):
        _truncate_clip_vision_layers(model, num_vit_layers)
    processor = CLIPProcessor.from_pretrained(backbone_name)
    return model, processor


def _encode_image_features(clip_model, processor, images, device: torch.device, backbone_type: str) -> torch.Tensor:
    if backbone_type == "open_clip":
        pixel_values = torch.stack([processor(img) for img in images], dim=0).to(device)
        return clip_model.encode_image(pixel_values)

    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return _to_tensor(clip_model.get_image_features(**inputs))


def _get_clip_image_feature_dim(clip_model, backbone_type: str) -> int:
    # CLIP backbones can expose different image feature dimensions (e.g., B32=512, L14=768, RN101=512).
    if backbone_type == "open_clip":
        dim = getattr(getattr(clip_model, "visual", None), "output_dim", None)
        if isinstance(dim, int) and dim > 0:
            return dim
        dim = getattr(clip_model, "embed_dim", None)
        if isinstance(dim, int) and dim > 0:
            return dim
        return 512

    dim = getattr(getattr(clip_model, "config", None), "projection_dim", None)
    if isinstance(dim, int) and dim > 0:
        return dim

    out_features = getattr(getattr(clip_model, "visual_projection", None), "out_features", None)
    if isinstance(out_features, int) and out_features > 0:
        return out_features

    return 512


def _rel_path_to_label_source(rel_path: str) -> Tuple[int, str]:
    path = Path(rel_path)
    parts_lower = [p.lower() for p in path.parts]
    is_real = any("real" in p for p in parts_lower) and not any("fake" in p for p in parts_lower)
    label = 0 if is_real else 1

    stem = path.stem
    if "__" in stem:
        source = stem.split("__", 1)[0]
    else:
        source = "unknown"
        for key in ["cermep", "tcga", "upenn", "gan", "ldm", "mls_cermep", "mls_tcga", "mls_upenn", "mls"]:
            if any(key in p for p in parts_lower):
                source = key.upper() if key in {"gan", "ldm"} else key
                break

    return label, source


class ImagePathDataset(Dataset):
    def __init__(self, dataset_root: Path, rel_paths: Sequence[str]):
        self.dataset_root = dataset_root
        self.rel_paths = list(rel_paths)

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        rel = self.rel_paths[idx]
        abs_path = self.dataset_root / rel
        image = Image.open(abs_path).convert("RGB")
        label, source = _rel_path_to_label_source(rel)
        return image, label, source, rel


def collate_fn(batch):
    images, labels, sources, rel_paths = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(sources), list(rel_paths)


class EuclideanProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 512, projection_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim),
        )
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HyperbolicProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 512, projection_dim: int = 256, curvature: float = 1.0, scale: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim),
        )
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        self.ball = geoopt.PoincareBall(c=curvature)
        self.curvature = curvature
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.net(x)
        z_e = nn.functional.normalize(z_e, dim=-1) * self.scale.abs()
        return self.ball.expmap0(z_e)


@torch.no_grad()
def compute_center_euclidean(clip_model, processor, projection_head, dataset, batch_size, device, backbone_type: str):
    clip_model.eval()
    projection_head.eval()
    embs = []
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        imgs = [dataset[i][0] for i in range(start, end)]
        feats = _encode_image_features(clip_model, processor, imgs, device, backbone_type)
        proj = projection_head(feats)
        embs.append(proj.cpu())
    return torch.cat(embs, dim=0).mean(dim=0)


@torch.no_grad()
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
def compute_center_hyperbolic(clip_model, processor, projection_head, dataset, batch_size, device, backbone_type: str):
    clip_model.eval()
    projection_head.eval()
    embs = []
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        imgs = [dataset[i][0] for i in range(start, end)]
        feats = _encode_image_features(clip_model, processor, imgs, device, backbone_type)
        proj = projection_head(feats)
        embs.append(proj.cpu())
    all_embs = torch.cat(embs, dim=0)
    return frechet_mean_iterative(all_embs, projection_head.curvature)


def hyperbolic_distance(x, center, ball):
    center_exp = center.unsqueeze(0)
    return ball.dist(x, center_exp).view(-1)


def svdd_loss_euclidean(embeddings, center):
    return torch.sum((embeddings - center) ** 2, dim=-1).mean()


def svdd_loss_hyperbolic(embeddings, center, ball):
    return (hyperbolic_distance(embeddings, center, ball) ** 2).mean()


@torch.no_grad()
def compute_anomaly_scores(clip_model, processor, projection_head, center, dataset, batch_size, device, geometry: str, backbone_type: str):
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

        feats = _encode_image_features(clip_model, processor, imgs, device, backbone_type)
        proj = projection_head(feats)

        if geometry == "euclidean":
            dist = torch.sqrt(torch.sum((proj - center) ** 2, dim=-1))
        else:
            dist = hyperbolic_distance(proj, center, projection_head.ball)

        scores.extend(dist.detach().cpu().numpy().tolist())
        labels.extend(lbs)
        sources.extend(srcs)
        ids.extend(rels)

    return np.array(labels), np.array(scores), sources, ids


def predict_from_threshold(scores: np.ndarray, threshold: float, fake_positive_if_high: bool = True):
    return (scores > threshold).astype(int) if fake_positive_if_high else (scores < threshold).astype(int)


def compute_metrics(labels, scores, threshold, fake_positive_if_high=True):
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
    out["classification_report"] = classification_report(labels, preds, target_names=["Real", "Fake"], output_dict=True, zero_division=0)
    return out


def calibrate_threshold(labels, scores, fake_positive_if_high=True):
    uniq = np.unique(scores)
    candidates = uniq if len(uniq) < 10 else np.quantile(scores, np.linspace(0.01, 0.99, 200))

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


def compute_per_source_accuracy(labels, scores, sources, threshold, fake_positive_if_high=True):
    preds = predict_from_threshold(scores, threshold, fake_positive_if_high)
    src_correct = defaultdict(int)
    src_total = defaultdict(int)
    for pred, label, source in zip(preds, labels, sources):
        src_total[source] += 1
        if pred == label:
            src_correct[source] += 1
    return {k: round(src_correct[k] / src_total[k], 4) for k in sorted(src_total)}


def save_loss_curve(rows, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SVDD Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc_curve(labels, scores, title, out_path, fake_positive_if_high=True):
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
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_pr_curve(labels, scores, title, out_path, fake_positive_if_high=True):
    rank_scores = scores if fake_positive_if_high else -scores
    precision, recall, _ = precision_recall_curve(labels, rank_scores)
    auc_val = average_precision_score(labels, rank_scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label=f"AUPRC = {auc_val:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_score_distribution(real_scores, fake_scores, thresholds: Dict[str, float], out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(real_scores, bins=40, alpha=0.5, label="Real")
    ax.hist(fake_scores, bins=40, alpha=0.5, label="Fake")
    for name, threshold in thresholds.items():
        ax.axvline(threshold, linestyle="--", label=name)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(labels, preds, title, out_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def configure_clip_trainability(clip_model, vision_mode: str, backbone_type: str):
    train_vision = vision_mode != "frozen"

    if backbone_type == "open_clip":
        for p in clip_model.parameters():
            p.requires_grad = False
        if hasattr(clip_model, "visual"):
            for p in clip_model.visual.parameters():
                p.requires_grad = train_vision
        return

    for p in clip_model.text_model.parameters():
        p.requires_grad = False
    for p in clip_model.text_projection.parameters():
        p.requires_grad = False
    for p in clip_model.vision_model.parameters():
        p.requires_grad = train_vision
    for p in clip_model.visual_projection.parameters():
        p.requires_grad = train_vision


def build_optimizer(clip_model, projection_head, cfg, geometry: str):
    lr_image = float(cfg.get("lr_image", 1e-5))
    lr_proj = float(cfg.get("lr_projection", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-4))

    trainable_clip = [p for p in clip_model.parameters() if p.requires_grad]
    if geometry == "hyperbolic":
        return torch.optim.AdamW(
            [
                {"params": trainable_clip, "lr": lr_image},
                {"params": projection_head.parameters(), "lr": lr_proj},
            ],
            weight_decay=wd,
        )

    return torch.optim.AdamW(
        [
            {"params": trainable_clip, "lr": lr_image},
            {"params": projection_head.parameters(), "lr": lr_proj},
        ],
        weight_decay=wd,
    )


def train_one_epoch(clip_model, processor, projection_head, center, loader, optimizer, scheduler, scaler, device, geometry: str, backbone_type: str):
    clip_model.train()
    projection_head.train()
    running = 0.0
    n_batches = 0
    center = center.to(device)

    for images, _, _, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            feats = _encode_image_features(clip_model, processor, images, device, backbone_type)
            proj = projection_head(feats)
            if geometry == "euclidean":
                loss = svdd_loss_euclidean(proj, center)
            else:
                loss = svdd_loss_hyperbolic(proj, center, projection_head.ball)

        scaler.scale(loss).backward()
        if geometry == "hyperbolic":
            torch.nn.utils.clip_grad_norm_(projection_head.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        running += float(loss.item())
        n_batches += 1

    return running / max(n_batches, 1)


def evaluate_test_set(clip_model, processor, projection_head, center, dataset, batch_size, device, geometry: str, thresholds: Dict[str, float], out_dir: Path, title_prefix: str, backbone_type: str):
    labels, scores, sources, ids = compute_anomaly_scores(
        clip_model, processor, projection_head, center, dataset, batch_size, device, geometry, backbone_type
    )

    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]
    save_score_distribution(real_scores, fake_scores, thresholds, out_dir / "score_distribution.png")
    save_roc_curve(labels, scores, f"ROC - {title_prefix}", out_dir / "roc_curve.png")
    save_pr_curve(labels, scores, f"PR - {title_prefix}", out_dir / "pr_curve.png")

    threshold_results = {}
    for name, threshold in thresholds.items():
        metrics = compute_metrics(labels, scores, threshold)
        preds = predict_from_threshold(scores, threshold)
        save_confusion_matrix(labels, preds, f"CM - {title_prefix} ({name})", out_dir / f"confusion_matrix_{name}.png")
        metrics["per_source_accuracy"] = compute_per_source_accuracy(labels, scores, sources, threshold)
        threshold_results[name] = metrics

    return {
        "labels": labels,
        "scores": scores,
        "sources": sources,
        "ids": ids,
        "threshold_results": threshold_results,
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def run_fold(cfg: dict, dataset_root: Path, geometry: str, fold: Dict, fold_dir: Path) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model_name = cfg.get("clip_model_name", cfg.get("backbone", {}).get("model_name", "openai/clip-vit-base-patch16"))
    backbone_type = cfg.get("backbone", {}).get("type", "clip")
    open_clip_pretrained = cfg.get("backbone", {}).get("open_clip_pretrained", "openai")
    batch_size = int(cfg.get("batch_size", 32))
    epochs = int(cfg.get("epochs", 10))
    projection_dim = int(cfg.get("projection_dim", 256))
    threshold_percentile = float(cfg.get("threshold_percentile", 95))
    num_workers = int(cfg.get("num_workers", 0))
    curvature = float(cfg.get("curvature", 1.0))
    scale = float(cfg.get("scale", 0.1))
    seed = int(cfg.get("seed", 42)) + int(fold["fold_index"]) * 100
    vision_mode = cfg.get("backbone", {}).get("vision_encoder_mode", "fine_tune")
    num_vit_layers = int(cfg.get("backbone", {}).get("num_vit_layers", 12))

    set_global_determinism(seed)

    clip_model, processor = _load_image_backbone(
        clip_model_name,
        backbone_type,
        open_clip_pretrained,
        device,
        num_vit_layers=num_vit_layers,
    )
    configure_clip_trainability(clip_model, vision_mode, backbone_type)
    feature_dim = _get_clip_image_feature_dim(clip_model, backbone_type)

    if geometry == "euclidean":
        projection_head = EuclideanProjectionHead(input_dim=feature_dim, projection_dim=projection_dim).to(device)
    else:
        projection_head = HyperbolicProjectionHead(
            input_dim=feature_dim,
            projection_dim=projection_dim,
            curvature=curvature,
            scale=scale,
        ).to(device)

    train_ds = ImagePathDataset(dataset_root, fold["train_ids"])
    val_real_ds = ImagePathDataset(dataset_root, fold["val_ids"])
    val_eval_ds = ImagePathDataset(dataset_root, fold["val_eval_ids"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        worker_init_fn=worker_init_fn(seed),
        generator=build_loader_generator(seed),
    )

    optimizer = build_optimizer(clip_model, projection_head, cfg, geometry)
    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps_per_epoch * epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_auroc = -1.0
    best_payload = None
    log_rows = []
    best_val_labels = None
    best_val_scores = None
    best_val_thresholds = None

    for epoch in range(1, epochs + 1):
        if geometry == "euclidean":
            center = compute_center_euclidean(clip_model, processor, projection_head, train_ds, batch_size, device, backbone_type)
        else:
            center = compute_center_hyperbolic(clip_model, processor, projection_head, train_ds, batch_size, device, backbone_type)

        train_loss = train_one_epoch(
            clip_model,
            processor,
            projection_head,
            center,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            geometry,
            backbone_type,
        )

        if geometry == "euclidean":
            center = compute_center_euclidean(clip_model, processor, projection_head, train_ds, batch_size, device, backbone_type)
        else:
            center = compute_center_hyperbolic(clip_model, processor, projection_head, train_ds, batch_size, device, backbone_type)

        val_in_labels, val_in_scores, _, _ = compute_anomaly_scores(
            clip_model, processor, projection_head, center, val_real_ds, batch_size, device, geometry, backbone_type
        )
        _ = val_in_labels
        val_eval_labels, val_eval_scores, val_eval_sources, _ = compute_anomaly_scores(
            clip_model, processor, projection_head, center, val_eval_ds, batch_size, device, geometry, backbone_type
        )
        _ = val_eval_sources

        default_threshold = float(np.percentile(val_in_scores, threshold_percentile))
        best_f1, best_j = calibrate_threshold(val_eval_labels, val_eval_scores)
        val_metrics = {
            "default": compute_metrics(val_eval_labels, val_eval_scores, default_threshold),
            "f1": compute_metrics(val_eval_labels, val_eval_scores, best_f1["threshold"]),
            "youden_j": compute_metrics(val_eval_labels, val_eval_scores, best_j["threshold"]),
        }

        auroc = val_metrics["f1"]["auroc"]
        improved = auroc > best_auroc
        if improved:
            best_auroc = auroc
            best_payload = {
                "clip_model": clip_model.state_dict(),
                "projection_head": projection_head.state_dict(),
                "center": center.detach().cpu(),
                "default_threshold": default_threshold,
                "calibrated_threshold_f1": best_f1["threshold"],
                "calibrated_threshold_youden_j": best_j["threshold"],
                "geometry": geometry,
                "fold_index": fold["fold_index"],
                "val_metrics": val_metrics,
            }
            best_val_labels = val_eval_labels.copy()
            best_val_scores = val_eval_scores.copy()
            best_val_thresholds = {
                "default": float(default_threshold),
                "f1": float(best_f1["threshold"]),
                "youden_j": float(best_j["threshold"]),
            }
            torch.save(best_payload, fold_dir / "best_model.pth")

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_auroc": round(val_metrics["f1"]["auroc"], 6),
                "val_auprc": round(val_metrics["f1"]["auprc"], 6),
                "default_threshold": round(default_threshold, 6),
                "calibrated_threshold_f1": round(float(best_f1["threshold"]), 6),
                "calibrated_threshold_youden_j": round(float(best_j["threshold"]), 6),
            }
        )

        print(
            f"  [{geometry}] fold={fold['fold_index']} epoch={epoch}/{epochs} loss={train_loss:.4f} "
            f"val_auroc={val_metrics['f1']['auroc']:.4f}{' *' if improved else ''}",
            flush=True,
        )

    with (fold_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    save_loss_curve(log_rows, fold_dir / "loss_curve.png")

    # Save fold-level validation interpretation artifacts for all thresholds.
    val_threshold_rows = []
    for threshold_name in ["default", "f1", "youden_j"]:
        th = best_val_thresholds[threshold_name]
        preds = predict_from_threshold(best_val_scores, th)
        cm_title = f"Val CM - {geometry} fold {fold['fold_index']} ({threshold_name})"
        save_confusion_matrix(best_val_labels, preds, cm_title, fold_dir / f"val_confusion_matrix_{threshold_name}.png")

        report = classification_report(
            best_val_labels,
            preds,
            target_names=["Real", "Fake"],
            output_dict=True,
            zero_division=0,
        )
        with (fold_dir / f"val_classification_report_{threshold_name}.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        metrics = compute_metrics(best_val_labels, best_val_scores, th)
        val_threshold_rows.append(
            {
                "threshold_name": threshold_name,
                "threshold_value": th,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "PPV": metrics["PPV"],
                "NPV": metrics["NPV"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
            }
        )

    with (fold_dir / "val_threshold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold_name",
                "threshold_value",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "sensitivity",
                "specificity",
                "PPV",
                "NPV",
                "auroc",
                "auprc",
            ],
        )
        writer.writeheader()
        writer.writerows(val_threshold_rows)

    with (fold_dir / "fold_results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "geometry": geometry,
                "fold_index": fold["fold_index"],
                "train_n_images": fold["train_n_images"],
                "val_n_images": fold["val_n_images"],
                "val_eval_n_images": fold["val_eval_n_images"],
                "calibration_fake_counts": fold.get("calibration_fake_counts", {}),
                "best_checkpoint": "best_model.pth",
                "best_val_auroc": best_payload["val_metrics"]["f1"]["auroc"],
                "best_val_auprc": best_payload["val_metrics"]["f1"]["auprc"],
                "default_threshold": best_payload["default_threshold"],
                "calibrated_threshold_f1": best_payload["calibrated_threshold_f1"],
                "calibrated_threshold_youden_j": best_payload["calibrated_threshold_youden_j"],
                "val_metrics": best_payload["val_metrics"],
            },
            f,
            indent=2,
        )

    return {
        "fold_index": fold["fold_index"],
        "best_val_auroc": best_payload["val_metrics"]["f1"]["auroc"],
        "best_val_auprc": best_payload["val_metrics"]["f1"]["auprc"],
        "checkpoint_path": (fold_dir / "best_model.pth").as_posix(),
        "fold_dir": fold_dir.as_posix(),
    }


def load_best_model(cfg: dict, geometry: str, checkpoint_path: Path, device: torch.device):
    clip_model_name = cfg.get("clip_model_name", cfg.get("backbone", {}).get("model_name", "openai/clip-vit-base-patch16"))
    backbone_type = cfg.get("backbone", {}).get("type", "clip")
    open_clip_pretrained = cfg.get("backbone", {}).get("open_clip_pretrained", "openai")
    projection_dim = int(cfg.get("projection_dim", 256))
    curvature = float(cfg.get("curvature", 1.0))
    scale = float(cfg.get("scale", 0.1))
    vision_mode = cfg.get("backbone", {}).get("vision_encoder_mode", "fine_tune")
    num_vit_layers = int(cfg.get("backbone", {}).get("num_vit_layers", 12))

    clip_model, processor = _load_image_backbone(
        clip_model_name,
        backbone_type,
        open_clip_pretrained,
        device,
        num_vit_layers=num_vit_layers,
    )
    configure_clip_trainability(clip_model, vision_mode, backbone_type)
    feature_dim = _get_clip_image_feature_dim(clip_model, backbone_type)

    if geometry == "euclidean":
        projection_head = EuclideanProjectionHead(input_dim=feature_dim, projection_dim=projection_dim).to(device)
    else:
        projection_head = HyperbolicProjectionHead(
            input_dim=feature_dim,
            projection_dim=projection_dim,
            curvature=curvature,
            scale=scale,
        ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    clip_model.load_state_dict(ckpt["clip_model"])
    projection_head.load_state_dict(ckpt["projection_head"])
    center = ckpt["center"].to(device)
    thresholds = {
        "f1": float(ckpt["calibrated_threshold_f1"]),
        "youden_j": float(ckpt["calibrated_threshold_youden_j"]),
        "default": float(ckpt["default_threshold"]),
    }
    return clip_model, processor, projection_head, center, thresholds


def write_summary_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "geometry",
        "test_set",
        "n_real",
        "n_fake",
        "auroc",
        "auprc",
        "accuracy_default",
        "f1_default",
        "sensitivity_default",
        "specificity_default",
        "accuracy_f1",
        "f1_f1",
        "sensitivity_f1",
        "specificity_f1",
        "accuracy_youden_j",
        "f1_youden_j",
        "sensitivity_youden_j",
        "specificity_youden_j",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_geometry(cfg: dict, manifest: dict, dataset_root: Path, geometry: str, geometry_dir: Path) -> Dict:
    fold_summaries = []
    for fold in manifest["cv_folds"]:
        fold_dir = geometry_dir / f"fold_{fold['fold_index']}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_summaries.append(run_fold(cfg, dataset_root, geometry, fold, fold_dir))

    best_fold = max(fold_summaries, key=lambda x: (x["best_val_auroc"], x["best_val_auprc"]))
    with (geometry_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"folds": fold_summaries, "best_fold": best_fold}, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, processor, projection_head, center, thresholds = load_best_model(
        cfg,
        geometry,
        Path(best_fold["checkpoint_path"]),
        device,
    )
    backbone_type = cfg.get("backbone", {}).get("type", "clip")
    batch_size = int(cfg.get("batch_size", 32))

    test_results = {}
    summary_rows = []

    for test_name, test_spec in manifest["test_sets"].items():
        test_dir = geometry_dir / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        rel_paths = sorted(test_spec["real_ids"] + test_spec["fake_ids"])
        dataset = ImagePathDataset(dataset_root, rel_paths)

        result = evaluate_test_set(
            clip_model,
            processor,
            projection_head,
            center,
            dataset,
            batch_size,
            device,
            geometry,
            thresholds,
            test_dir,
            f"{geometry} {test_name}",
            backbone_type,
        )

        payload = {
            "geometry": geometry,
            "test_set": test_name,
            "n_real": test_spec["n_real"],
            "n_fake": test_spec["n_fake"],
            "auroc": result["auroc"],
            "auprc": result["auprc"],
            "threshold_results": result["threshold_results"],
            "best_fold_index": best_fold["fold_index"],
            "thresholds": thresholds,
        }

        threshold_metrics_rows = []
        for threshold_name in ["default", "f1", "youden_j"]:
            m = result["threshold_results"][threshold_name]
            threshold_metrics_rows.append(
                {
                    "threshold_name": threshold_name,
                    "threshold_value": thresholds[threshold_name],
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"],
                    "PPV": m["PPV"],
                    "NPV": m["NPV"],
                }
            )
            with (test_dir / f"classification_report_{threshold_name}.json").open("w", encoding="utf-8") as f:
                json.dump(m["classification_report"], f, indent=2)

        with (test_dir / "threshold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "threshold_name",
                    "threshold_value",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "sensitivity",
                    "specificity",
                    "PPV",
                    "NPV",
                ],
            )
            writer.writeheader()
            writer.writerows(threshold_metrics_rows)

        if test_name == "test_mls":
            labels = result["labels"]
            scores = result["scores"]
            sources = result["sources"]
            mls_breakdown = {}
            for threshold_name, threshold in thresholds.items():
                preds = predict_from_threshold(scores, threshold)
                per_source = {}
                for source in sorted(set(sources)):
                    idxs = [i for i, s in enumerate(sources) if s == source and labels[i] == 1]
                    if not idxs:
                        continue
                    correct = sum(int(preds[i] == labels[i]) for i in idxs)
                    per_source[source] = {
                        "n_samples": len(idxs),
                        "accuracy": round(correct / len(idxs), 4),
                    }
                mls_breakdown[threshold_name] = per_source
            payload["mls_subsource_breakdown"] = mls_breakdown
            with (test_dir / "mls_subsource_breakdown.json").open("w", encoding="utf-8") as f:
                json.dump(mls_breakdown, f, indent=2)

        with (test_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        summary_rows.append(
            {
                "geometry": geometry,
                "test_set": test_name,
                "n_real": test_spec["n_real"],
                "n_fake": test_spec["n_fake"],
                "auroc": round(result["auroc"], 6),
                "auprc": round(result["auprc"], 6),
                "accuracy_default": round(result["threshold_results"]["default"]["accuracy"], 6),
                "f1_default": round(result["threshold_results"]["default"]["f1"], 6),
                "sensitivity_default": round(result["threshold_results"]["default"]["sensitivity"], 6),
                "specificity_default": round(result["threshold_results"]["default"]["specificity"], 6),
                "accuracy_f1": round(result["threshold_results"]["f1"]["accuracy"], 6),
                "f1_f1": round(result["threshold_results"]["f1"]["f1"], 6),
                "sensitivity_f1": round(result["threshold_results"]["f1"]["sensitivity"], 6),
                "specificity_f1": round(result["threshold_results"]["f1"]["specificity"], 6),
                "accuracy_youden_j": round(result["threshold_results"]["youden_j"]["accuracy"], 6),
                "f1_youden_j": round(result["threshold_results"]["youden_j"]["f1"], 6),
                "sensitivity_youden_j": round(result["threshold_results"]["youden_j"]["sensitivity"], 6),
                "specificity_youden_j": round(result["threshold_results"]["youden_j"]["specificity"], 6),
            }
        )

        test_results[test_name] = payload
        print(
            f"[TEST] geometry={geometry} set={test_name} auroc={result['auroc']:.4f} auprc={result['auprc']:.4f} "
            f"f1@f1={result['threshold_results']['f1']['f1']:.4f} f1@j={result['threshold_results']['youden_j']['f1']:.4f}",
            flush=True,
        )

    write_summary_csv(summary_rows, geometry_dir / "summary_8run_slice.csv")
    with (geometry_dir / "test_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"best_fold": best_fold, "tests": test_results}, f, indent=2)

    return {"geometry": geometry, "best_fold": best_fold, "summary_rows": summary_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="One-Class SVDD CLIP v2 orchestrator")
    parser.add_argument("--config", type=str, default="configs/one_class_svdd_clip_v2.yaml")
    parser.add_argument("--build-only", action="store_true", help="Only build protocol manifests, skip training/eval")
    parser.add_argument(
        "--backbones",
        nargs="*",
        default=None,
        help=(
            "Backbone keys to run sequentially (space/comma separated). "
            "Example: --backbones B32 L16 RN101 or --backbones B32,RN101. "
            "If omitted, the config default backbone is used."
        ),
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        default=None,
        help=(
            "ViT layer counts to run sequentially (space/comma separated). "
            "Example: --layers 2 4 6 8 10 or --layers 4,8,12. "
            "If omitted, the config default num_vit_layers is used."
        ),
    )
    parser.add_argument(
        "--protocol-manifest",
        type=str,
        default=None,
        help=(
            "Optional path to an existing protocol manifest JSON to reuse exact splits/folds. "
            "Overrides protocol_manifest_path in config when provided."
        ),
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    cfg = _load_config(cfg_path)
    backbone_registry, default_backbone_key = _resolve_backbone_registry(cfg)

    requested_backbones = _parse_requested_backbones(args.backbones)
    selected_backbones = requested_backbones or [default_backbone_key]
    unknown_backbones = [b for b in selected_backbones if b not in backbone_registry]
    if unknown_backbones:
        print("[ERROR] Unknown backbone key(s):", ", ".join(unknown_backbones), flush=True)
        print("[ERROR] Available backbone key(s):", ", ".join(sorted(backbone_registry.keys())), flush=True)
        return 2

    try:
        requested_layers = _parse_requested_layers(args.layers)
    except ValueError as exc:
        print(f"[ERROR] {exc}", flush=True)
        return 2

    default_layers = _default_num_vit_layers(cfg)
    selected_layers = requested_layers or [default_layers]

    dataset_root = Path(cfg["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root

    output_root = Path(cfg["output_root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    seed = int(cfg.get("seed", 42))
    target_real_train = int(cfg.get("target_real_train_images", 500))
    target_per_generator = int(cfg.get("target_per_generator", 104))
    n_folds = int(cfg.get("n_folds", 5))

    run_name = cfg.get("run_name", f"one_class_svdd_clip_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "backbone_selection.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "default_backbone": default_backbone_key,
                "requested_backbones": requested_backbones,
                "selected_backbones": selected_backbones,
                "default_num_vit_layers": default_layers,
                "requested_layers": requested_layers,
                "selected_layers": selected_layers,
                "available_backbones": backbone_registry,
            },
            f,
            indent=2,
        )

    manifest_path_cli = _resolve_optional_protocol_manifest(args.protocol_manifest)
    manifest_path_cfg = _resolve_optional_protocol_manifest(cfg.get("protocol_manifest_path"))
    selected_manifest_path = manifest_path_cli or manifest_path_cfg

    if selected_manifest_path is not None:
        if not selected_manifest_path.exists():
            raise FileNotFoundError(f"Protocol manifest not found: {selected_manifest_path}")
        manifest = _load_protocol_manifest(selected_manifest_path)
        print("[INFO] Reusing protocol manifest:", selected_manifest_path, flush=True)
    else:
        manifest = build_protocol_manifest(
            dataset_root=dataset_root,
            seed=seed,
            target_real_train_images=target_real_train,
            target_per_generator=target_per_generator,
            n_folds=n_folds,
        )
        print("[INFO] Generated protocol manifest from dataset and config.", flush=True)

    protocol_manifest_path = run_dir / "protocol_manifest.json"
    save_manifest(manifest, protocol_manifest_path)

    split_summary_path = run_dir / "split_summary.json"
    with split_summary_path.open("w", encoding="utf-8") as f:
        json.dump(manifest["summary"], f, indent=2)

    print("[INFO] Saved protocol manifest:", protocol_manifest_path, flush=True)
    print("[INFO] Real train images:", manifest["summary"]["n_real_train_images"], flush=True)
    print("[INFO] Real test pool images:", manifest["summary"]["n_real_test_pool_images"], flush=True)
    print("[INFO] MLS sampled sub-source counts:", manifest["summary"]["mls_source_sample_counts"], flush=True)
    print("[INFO] Calibration fake pool by generator:", manifest["summary"]["calibration_fake_pool_by_generator"], flush=True)

    if args.build_only:
        print("[INFO] build-only mode enabled. Training/evaluation not started.", flush=True)
        return 0

    geometries = cfg.get("geometries", ["euclidean", "hyperbolic"])
    multi_backbone_rows = []
    multi_backbone_runs = []

    for backbone_key in selected_backbones:
        backbone_spec = backbone_registry[backbone_key]
        cfg_for_backbone = _apply_backbone_to_cfg(cfg, backbone_spec)

        if len(selected_backbones) == 1 and not requested_backbones:
            backbone_base_dir = run_dir
        else:
            backbone_base_dir = run_dir / f"backbone_{backbone_key}"
            backbone_base_dir.mkdir(parents=True, exist_ok=True)

        for num_vit_layers in selected_layers:
            cfg_for_combo = _apply_layer_to_cfg(cfg_for_backbone, num_vit_layers)

            if len(selected_layers) == 1 and not requested_layers:
                combo_run_dir = backbone_base_dir
            else:
                combo_run_dir = backbone_base_dir / f"layer_{num_vit_layers}"
                combo_run_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"[INFO] Running backbone={backbone_key} model={backbone_spec['model_name']} "
                f"layers={num_vit_layers} in {combo_run_dir}",
                flush=True,
            )

            all_summary_rows = []
            geometry_summaries = []
            for geometry in geometries:
                geometry_dir = combo_run_dir / geometry
                geometry_dir.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                summary = run_geometry(cfg_for_combo, manifest, dataset_root, geometry, geometry_dir)
                summary["elapsed_sec"] = round(time.time() - t0, 2)
                summary["backbone_key"] = backbone_key
                summary["clip_model_name"] = backbone_spec["model_name"]
                summary["num_vit_layers"] = num_vit_layers
                geometry_summaries.append(summary)

                for row in summary["summary_rows"]:
                    row_with_context = dict(row)
                    row_with_context["backbone_key"] = backbone_key
                    row_with_context["clip_model_name"] = backbone_spec["model_name"]
                    row_with_context["num_vit_layers"] = num_vit_layers
                    all_summary_rows.append(row_with_context)
                    multi_backbone_rows.append(row_with_context)

            write_summary_csv(
                [
                    {
                        k: v
                        for k, v in row.items()
                        if k
                        in {
                            "geometry",
                            "test_set",
                            "n_real",
                            "n_fake",
                            "auroc",
                            "auprc",
                            "accuracy_default",
                            "f1_default",
                            "sensitivity_default",
                            "specificity_default",
                            "accuracy_f1",
                            "f1_f1",
                            "sensitivity_f1",
                            "specificity_f1",
                            "accuracy_youden_j",
                            "f1_youden_j",
                            "sensitivity_youden_j",
                            "specificity_youden_j",
                        }
                    }
                    for row in all_summary_rows
                ],
                combo_run_dir / "final_8run_summary.csv",
            )
            with (combo_run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "backbone_key": backbone_key,
                        "clip_model_name": backbone_spec["model_name"],
                        "num_vit_layers": num_vit_layers,
                        "geometries": geometry_summaries,
                    },
                    f,
                    indent=2,
                )

            multi_backbone_runs.append(
                {
                    "backbone_key": backbone_key,
                    "clip_model_name": backbone_spec["model_name"],
                    "num_vit_layers": num_vit_layers,
                    "run_dir": combo_run_dir.as_posix(),
                }
            )

    with (run_dir / "run_summary_multi_backbone.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": multi_backbone_runs,
                "selected_backbones": selected_backbones,
            },
            f,
            indent=2,
        )

    if len(selected_backbones) > 1 or len(selected_layers) > 1:
        fieldnames = [
            "backbone_key",
            "clip_model_name",
            "num_vit_layers",
            "geometry",
            "test_set",
            "n_real",
            "n_fake",
            "auroc",
            "auprc",
            "accuracy_default",
            "f1_default",
            "sensitivity_default",
            "specificity_default",
            "accuracy_f1",
            "f1_f1",
            "sensitivity_f1",
            "specificity_f1",
            "accuracy_youden_j",
            "f1_youden_j",
            "sensitivity_youden_j",
            "specificity_youden_j",
        ]
        with (run_dir / "final_multi_backbone_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                [
                    {key: row.get(key) for key in fieldnames}
                    for row in multi_backbone_rows
                ]
            )

    print("[INFO] Backbone run(s) completed under", run_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())