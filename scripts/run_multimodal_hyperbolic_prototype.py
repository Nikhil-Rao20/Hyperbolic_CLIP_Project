from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageEnhance, ImageFilter
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
            "text_model_name": spec.get(
                "text_model_name",
                cfg.get("clip_model_name", cfg.get("backbone", {}).get("model_name", "openai/clip-vit-base-patch16")),
            ),
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
    backbone_cfg = dict(cfg.get("backbone", {}))
    backbone_cfg["model_name"] = backbone_spec["model_name"]
    backbone_cfg["type"] = backbone_spec.get("type", "clip")
    backbone_cfg["open_clip_pretrained"] = backbone_spec.get("open_clip_pretrained", "openai")
    backbone_cfg["vision_encoder_mode"] = backbone_spec.get("vision_encoder_mode", backbone_cfg.get("vision_encoder_mode", "fine_tune"))
    cfg["backbone"] = backbone_cfg
    if backbone_cfg["type"] == "dinov2":
        cfg["clip_text_model_name"] = backbone_spec.get("text_model_name", cfg.get("clip_model_name"))
    else:
        cfg["clip_model_name"] = backbone_spec["model_name"]
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


class DinoWithClipText(nn.Module):
    def __init__(self, image_model: nn.Module, text_model: CLIPModel):
        super().__init__()
        self.image_model = image_model
        self.text_model = text_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.image_model(pixel_values)

    def get_text_features(self, **inputs) -> torch.Tensor:
        return self.text_model.get_text_features(**inputs)


def _load_image_backbone(
    backbone_name: str,
    backbone_type: str,
    open_clip_pretrained: str,
    device: torch.device,
    num_vit_layers: int | None = None,
    text_model_name: str | None = None,
):
    if backbone_type == "dinov2":
        if not text_model_name:
            raise RuntimeError("DINOv2 backbone requires clip_text_model_name for text encoding.")
        import torchvision.transforms as transforms
        image_model = torch.hub.load("facebookresearch/dinov2", backbone_name).to(device)
        text_model = CLIPModel.from_pretrained(text_model_name, use_safetensors=True).to(device)
        preprocess = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        processor = {
            "image": preprocess,
            "text": CLIPProcessor.from_pretrained(text_model_name),
        }
        return DinoWithClipText(image_model, text_model), processor

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
    if backbone_type == "dinov2":
        image_processor = processor["image"] if isinstance(processor, dict) else processor
        pixel_values = torch.stack([image_processor(img) for img in images], dim=0).to(device)
        with torch.no_grad():
            feats = clip_model(pixel_values)
        return feats

    if backbone_type == "open_clip":
        pixel_values = torch.stack([processor(img) for img in images], dim=0).to(device)
        return clip_model.encode_image(pixel_values)

    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return _to_tensor(clip_model.get_image_features(**inputs))


def _get_clip_image_feature_dim(clip_model, backbone_type: str) -> int:
    # CLIP backbones can expose different image feature dimensions (e.g., B32=512, L14=768, RN101=512).
    if backbone_type == "dinov2":
        image_model = getattr(clip_model, "image_model", clip_model)
        dim = getattr(image_model, "embed_dim", None)
        if isinstance(dim, int) and dim > 0:
            return dim
        return 768

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


def _get_clip_text_feature_dim(clip_model, backbone_type: str) -> int:
    text_model = clip_model
    if backbone_type == "dinov2" and hasattr(clip_model, "text_model"):
        text_model = clip_model.text_model

    dim = getattr(getattr(text_model, "config", None), "projection_dim", None)
    if isinstance(dim, int) and dim > 0:
        return dim

    out_features = getattr(getattr(text_model, "text_projection", None), "out_features", None)
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


class MultimodalPromptDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        rel_paths: Sequence[str],
        real_prompts: Sequence[str],
        fake_prompts: Sequence[str],
        image_augmentor=None,
    ):
        if len(rel_paths) != len(real_prompts) or len(rel_paths) != len(fake_prompts):
            raise RuntimeError("Prompt counts must match rel_paths length.")
        self.dataset_root = dataset_root
        self.rel_paths = list(rel_paths)
        self.real_prompts = list(real_prompts)
        self.fake_prompts = list(fake_prompts)
        self.image_augmentor = image_augmentor

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        rel = self.rel_paths[idx]
        abs_path = self.dataset_root / rel
        image = Image.open(abs_path).convert("RGB")
        if self.image_augmentor is not None:
            image = self.image_augmentor(image)
        label, source = _rel_path_to_label_source(rel)
        return image, self.real_prompts[idx], self.fake_prompts[idx], label, source, rel


class GANLikeAugmentor:
    def __init__(self, prob: float = 0.35, min_ops: int = 1, max_ops: int = 3):
        self.prob = float(max(0.0, min(1.0, prob)))
        self.min_ops = max(1, int(min_ops))
        self.max_ops = max(self.min_ops, int(max_ops))

    def _jpeg_recompress(self, image: Image.Image) -> Image.Image:
        quality = random.randint(30, 85)
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def _resample_artifact(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        scale = random.uniform(0.55, 0.85)
        new_w, new_h = max(16, int(w * scale)), max(16, int(h * scale))
        down = image.resize((new_w, new_h), Image.BILINEAR)
        return down.resize((w, h), Image.BICUBIC)

    def _blur_or_sharpen(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            radius = random.uniform(0.5, 1.8)
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        factor = random.uniform(1.2, 2.2)
        return ImageEnhance.Sharpness(image).enhance(factor)

    def _tone_shift(self, image: Image.Image) -> Image.Image:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.25))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.15))
        return image

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return image

        ops = [self._jpeg_recompress, self._resample_artifact, self._blur_or_sharpen, self._tone_shift]
        n_ops = random.randint(self.min_ops, self.max_ops)
        for op in random.sample(ops, k=min(n_ops, len(ops))):
            image = op(image)
        return image


class SpectralAnomalyScorer:
    """
    Computes a frequency-domain anomaly score for each image.
    Fits a mean log-power azimuthal spectrum on real training images.
    At inference, returns L2 deviation of each image's log-spectrum from the mean.
    GAN artifacts (checkerboard, aliasing, upsampling ringing) show up strongly
    in high-frequency spectral components, making this complementary to spatial CLIP features.
    """
    TARGET_SIZE = (224, 224)

    def __init__(self):
        self.mean_spectrum: np.ndarray | None = None
        self.n_bins: int = 0

    def _azimuthal_power_spectrum(self, img: Image.Image) -> np.ndarray:
        gray = img.convert("L").resize(self.TARGET_SIZE, Image.BILINEAR)
        arr = np.array(gray, dtype=np.float32) / 255.0
        f = np.fft.fft2(arr)
        fshift = np.fft.fftshift(f)
        power = np.abs(fshift) ** 2
        h, w = power.shape
        cy, cx = h // 2, w // 2
        y_idx, x_idx = np.indices((h, w))
        r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)
        max_r = min(cx, cy)
        spectrum = np.zeros(max_r, dtype=np.float64)
        counts = np.zeros(max_r, dtype=np.int64)
        mask = r < max_r
        np.add.at(spectrum, r[mask], power[mask])
        np.add.at(counts, r[mask], 1)
        counts = np.maximum(counts, 1)
        spectrum = spectrum / counts
        return np.log1p(spectrum).astype(np.float32)

    def fit(self, images: List[Image.Image]) -> None:
        spectra = [self._azimuthal_power_spectrum(img) for img in images]
        arr = np.stack(spectra, axis=0)
        self.mean_spectrum = arr.mean(axis=0)
        self.n_bins = self.mean_spectrum.shape[0]

    def score(self, images: List[Image.Image]) -> np.ndarray:
        if self.mean_spectrum is None:
            raise RuntimeError("SpectralAnomalyScorer must be fit() before score().")
        scores = []
        for img in images:
            s = self._azimuthal_power_spectrum(img)
            deviation = float(np.linalg.norm(s - self.mean_spectrum))
            scores.append(deviation)
        return np.array(scores, dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "mean_spectrum": self.mean_spectrum.tolist() if self.mean_spectrum is not None else None,
            "n_bins": self.n_bins,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpectralAnomalyScorer":
        obj = cls()
        if d.get("mean_spectrum") is not None:
            obj.mean_spectrum = np.array(d["mean_spectrum"], dtype=np.float32)
            obj.n_bins = int(d.get("n_bins", obj.mean_spectrum.shape[0]))
        return obj


def multimodal_collate_fn(batch):
    images, real_prompts, fake_prompts, labels, sources, rel_paths = zip(*batch)
    return (
        list(images),
        list(real_prompts),
        list(fake_prompts),
        torch.tensor(labels, dtype=torch.long),
        list(sources),
        list(rel_paths),
    )


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


class SharedProjectionHead(nn.Module):
    def __init__(
        self,
        image_input_dim: int,
        text_input_dim: int,
        projection_dim: int,
        geometry: str,
        curvature: float,
        scale: float,
    ):
        super().__init__()
        self.geometry = geometry
        self.image_net = nn.Sequential(
            nn.Linear(image_input_dim, image_input_dim),
            nn.ReLU(),
            nn.Linear(image_input_dim, projection_dim),
        )
        self.text_net = nn.Sequential(
            nn.Linear(text_input_dim, text_input_dim),
            nn.ReLU(),
            nn.Linear(text_input_dim, projection_dim),
        )
        for module in list(self.image_net) + list(self.text_net):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        if geometry == "hyperbolic":
            self.ball = geoopt.PoincareBall(c=curvature)
            self.curvature = curvature
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.ball = None
            self.curvature = None
            self.scale = None

    def _project(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        z = net(x)
        if self.geometry == "hyperbolic":
            z = nn.functional.normalize(z, dim=-1) * self.scale.abs()
            return self.ball.expmap0(z)
        return z

    def project_image(self, x: torch.Tensor) -> torch.Tensor:
        return self._project(self.image_net, x)

    def project_text(self, x: torch.Tensor) -> torch.Tensor:
        return self._project(self.text_net, x)


def _project_image(projection_head, x: torch.Tensor) -> torch.Tensor:
    if hasattr(projection_head, "project_image"):
        return projection_head.project_image(x)
    return projection_head(x)


def _project_text(projection_head, x: torch.Tensor) -> torch.Tensor:
    if hasattr(projection_head, "project_text"):
        return projection_head.project_text(x)
    return projection_head(x)


@torch.no_grad()
def frechet_mean_iterative(points, curvature, max_iter=100, tol=1e-7):
    c = torch.tensor(float(curvature), device=points.device, dtype=points.dtype)
    ball = geoopt.PoincareBall(c=c)
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


def hyperbolic_distance(x, center, ball):
    if center.ndim == 1:
        center_exp = center.unsqueeze(0)
        return ball.dist(x, center_exp).view(-1)
    if center.ndim == 2:
        dists = [ball.dist(x, center[i].unsqueeze(0)).view(-1) for i in range(center.shape[0])]
        return torch.stack(dists, dim=1)
    raise RuntimeError(f"Invalid center ndim={center.ndim}; expected 1 or 2")


def _kmeans_torch(points: torch.Tensor, k: int, n_iter: int = 12) -> torch.Tensor:
    n = points.shape[0]
    if k <= 1 or n <= 1:
        return torch.zeros(n, dtype=torch.long, device=points.device)
    k = min(k, n)

    # deterministic init from evenly spaced indices under fixed global seed
    init_idx = torch.linspace(0, n - 1, steps=k, device=points.device).long()
    centroids = points[init_idx].clone()

    assignments = torch.zeros(n, dtype=torch.long, device=points.device)
    for _ in range(max(1, n_iter)):
        d2 = torch.cdist(points, centroids, p=2)
        assignments = torch.argmin(d2, dim=1)
        new_centroids = []
        for j in range(k):
            mask = assignments == j
            if mask.any():
                new_centroids.append(points[mask].mean(dim=0))
            else:
                new_centroids.append(centroids[j])
        centroids = torch.stack(new_centroids, dim=0)
    return assignments


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
    ax.set_ylabel("Loss")
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


def configure_clip_trainability(clip_model, vision_mode: str, backbone_type: str):
    train_vision = vision_mode != "frozen"

    if backbone_type == "dinov2":
        for p in clip_model.parameters():
            p.requires_grad = False
        if hasattr(clip_model, "image_model"):
            for p in clip_model.image_model.parameters():
                p.requires_grad = train_vision
        return

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


def load_prompts(prompt_file_path: Path) -> Tuple[List[str], List[str]]:
    with prompt_file_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "real_prompts" not in payload or "fake_prompts" not in payload:
        raise RuntimeError("Prompt file must include 'real_prompts' and 'fake_prompts' keys.")
    real_prompts = payload.get("real_prompts")
    fake_prompts = payload.get("fake_prompts")
    if not isinstance(real_prompts, list) or not isinstance(fake_prompts, list):
        raise RuntimeError("Prompt file 'real_prompts' and 'fake_prompts' must be lists.")
    if len(real_prompts) != len(fake_prompts):
        raise RuntimeError("Prompt file lists must be the same length.")
    return list(real_prompts), list(fake_prompts)


GAN_STYLE_FAKE_PROMPTS = [
    "A synthetic brain MRI image with adversarially generated texture artifacts and unrealistic high-frequency patterns.",
    "An AI-generated cranial MRI with subtle checkerboard upsampling artifacts and non-physiological edges.",
    "A forged MRI scan showing over-smoothed tissue boundaries and inconsistent local contrast statistics.",
    "A manipulated neuroimaging slice containing GAN-like texture repetition and spectral aliasing artifacts.",
    "A synthetic head MRI with implausible anatomical micro-texture and interpolation-induced ringing.",
    "An artificial MRI image with generation artifacts near cortical boundaries and abnormal fine-grain noise.",
]


def encode_text_features(clip_model, processor, texts: List[str], device: torch.device, backbone_type: str) -> torch.Tensor:
    with torch.no_grad():
        if backbone_type == "dinov2":
            text_processor = processor["text"] if isinstance(processor, dict) else processor
            inputs = text_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = clip_model.get_text_features(**inputs)
            return _to_tensor(feats)
        if backbone_type == "open_clip":
            tokens = open_clip.tokenize(texts).to(device)
            feats = clip_model.encode_text(tokens)
            return feats

        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = clip_model.get_text_features(**inputs)
        return _to_tensor(feats)


def compute_alignment_loss_only(z_img: torch.Tensor, z_text_real: torch.Tensor, geometry: str, ball_or_none):
    if geometry == "hyperbolic":
        return ball_or_none.dist(z_img, z_text_real).mean()
    return torch.norm(z_img - z_text_real, dim=-1).mean()


def multimodal_loss(
    z_img: torch.Tensor,
    z_text_real: torch.Tensor,
    p_real: torch.Tensor,
    p_fake: torch.Tensor,
    ball_or_none,
    geometry: str,
    lambda_align: float,
    lambda_reg: float,
    margin: float,
    hard_negative_weight: float,
) -> torch.Tensor:
    if geometry == "hyperbolic":
        d_real = ball_or_none.dist(z_img, p_real.unsqueeze(0)).view(-1)
        if p_fake.ndim == 1:
            d_fake = ball_or_none.dist(z_img, p_fake.unsqueeze(0)).view(-1)
        else:
            d_fake_all = hyperbolic_distance(z_img, p_fake, ball_or_none)
            d_fake = d_fake_all.min(dim=1).values
        l_align = ball_or_none.dist(z_img, z_text_real).mean()
        l_reg = (torch.norm(z_img, dim=-1) ** 2).mean() + (torch.norm(z_text_real, dim=-1) ** 2).mean()
    else:
        d_real = torch.norm(z_img - p_real.unsqueeze(0), dim=-1)
        if p_fake.ndim == 1:
            d_fake = torch.norm(z_img - p_fake.unsqueeze(0), dim=-1)
        else:
            d_fake = torch.norm(z_img.unsqueeze(1) - p_fake.unsqueeze(0), dim=-1).min(dim=1).values
        l_align = torch.norm(z_img - z_text_real, dim=-1).mean()
        l_reg = torch.tensor(0.0, device=z_img.device)

    per_sample = d_real + F.relu(margin - d_fake)
    hardness = torch.clamp(margin - (d_fake - d_real), min=0.0)
    weights = 1.0 + float(max(0.0, hard_negative_weight)) * (hardness / (margin + 1e-8))
    l_proto = (weights * per_sample).mean()
    return l_proto + lambda_align * l_align + lambda_reg * l_reg


@torch.no_grad()
def compute_prototype_real(
    clip_model,
    processor,
    projection_head,
    dataset: MultimodalPromptDataset,
    batch_size,
    device,
    geometry: str,
    backbone_type: str,
) -> torch.Tensor:
    clip_model.eval()
    projection_head.eval()
    image_embs = []
    text_embs = []
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        imgs = [dataset[i][0] for i in range(start, end)]
        real_prompts = [dataset[i][1] for i in range(start, end)]
        feats_img = _encode_image_features(clip_model, processor, imgs, device, backbone_type)
        feats_txt = encode_text_features(clip_model, processor, real_prompts, device, backbone_type)
        image_embs.append(_project_image(projection_head, feats_img).detach())
        text_embs.append(_project_text(projection_head, feats_txt).detach())

    all_embs = torch.cat(image_embs + text_embs, dim=0)
    if geometry == "hyperbolic":
        return frechet_mean_iterative(all_embs, projection_head.curvature)
    return all_embs.mean(dim=0)


@torch.no_grad()
def compute_prototype_fake(
    clip_model,
    processor,
    projection_head,
    dataset: MultimodalPromptDataset,
    batch_size,
    device,
    geometry: str,
    backbone_type: str,
) -> torch.Tensor:
    clip_model.eval()
    projection_head.eval()
    text_embs = []
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        fake_prompts = [dataset[i][2] for i in range(start, end)]
        feats_txt = encode_text_features(clip_model, processor, fake_prompts, device, backbone_type)
        text_embs.append(_project_text(projection_head, feats_txt).detach())

    all_embs = torch.cat(text_embs, dim=0)
    if geometry == "hyperbolic":
        return frechet_mean_iterative(all_embs, projection_head.curvature)
    return all_embs.mean(dim=0)


@torch.no_grad()
def compute_prototype_fake_multi(
    clip_model,
    processor,
    projection_head,
    dataset: MultimodalPromptDataset,
    batch_size,
    device,
    geometry: str,
    backbone_type: str,
    num_prototypes: int,
) -> torch.Tensor:
    base = compute_prototype_fake(
        clip_model,
        processor,
        projection_head,
        dataset,
        batch_size,
        device,
        geometry,
        backbone_type,
    )
    if int(num_prototypes) <= 1:
        return base

    clip_model.eval()
    projection_head.eval()
    text_embs = []
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        fake_prompts = [dataset[i][2] for i in range(start, end)]
        feats_txt = encode_text_features(clip_model, processor, fake_prompts, device, backbone_type)
        text_embs.append(_project_text(projection_head, feats_txt).detach())

    all_embs = torch.cat(text_embs, dim=0)
    k = int(max(1, min(num_prototypes, all_embs.shape[0])))
    if k == 1:
        return base

    if geometry == "hyperbolic":
        tangent = projection_head.ball.logmap0(all_embs)
        assignments = _kmeans_torch(tangent, k)
    else:
        assignments = _kmeans_torch(all_embs, k)

    clusters = []
    for j in range(k):
        pts = all_embs[assignments == j]
        if pts.numel() == 0:
            clusters.append(base)
            continue
        if geometry == "hyperbolic":
            clusters.append(frechet_mean_iterative(pts, projection_head.curvature))
        else:
            clusters.append(pts.mean(dim=0))

    return torch.stack(clusters, dim=0)


@torch.no_grad()
def compute_prototype_fake_gan_image(
    clip_model,
    processor,
    projection_head,
    train_dataset: MultimodalPromptDataset,
    batch_size: int,
    device: torch.device,
    geometry: str,
    backbone_type: str,
    n_aug_samples: int = 200,
    seed: int = 42,
) -> torch.Tensor:
    """
    Builds a fake prototype from IMAGE features of GAN-augmented real training images.
    Unlike the text-only fake prototypes, this places a reference point in hyperbolic
    space based on what actual GAN-style augmented images look like as image embeddings.
    This gives the model an image-space signal for GAN-like artifacts.
    """
    clip_model.eval()
    projection_head.eval()
    rng = random.Random(seed)
    augmentor = GANLikeAugmentor(prob=1.0, min_ops=2, max_ops=4)

    all_indices = list(range(len(train_dataset)))
    chosen = rng.sample(all_indices, k=min(n_aug_samples, len(all_indices)))

    image_embs = []
    for start in range(0, len(chosen), batch_size):
        batch_indices = chosen[start:start + batch_size]
        imgs = []
        for i in batch_indices:
            raw_img = Image.open(train_dataset.dataset_root / train_dataset.rel_paths[i]).convert("RGB")
            imgs.append(augmentor(raw_img))
        feats = _encode_image_features(clip_model, processor, imgs, device, backbone_type)
        image_embs.append(_project_image(projection_head, feats).detach())

    all_embs = torch.cat(image_embs, dim=0)
    if geometry == "hyperbolic":
        return frechet_mean_iterative(all_embs, projection_head.curvature)
    return all_embs.mean(dim=0)


@torch.no_grad()
def compute_anomaly_scores_multimodal(
    clip_model,
    processor,
    projection_head,
    p_real: torch.Tensor,
    p_fake: torch.Tensor,
    dataset,
    batch_size,
    device,
    geometry: str,
    backbone_type: str,
    spectral_scorer: "SpectralAnomalyScorer | None" = None,
    mu_spatial: float = 0.0,
    sigma_spatial: float = 1.0,
    mu_spectral: float = 0.0,
    sigma_spectral: float = 1.0,
    lambda_spectral: float = 0.0,
):
    """
    Computes per-image anomaly scores.
    If spectral_scorer is provided and lambda_spectral > 0, fuses:
      z_spatial = (spatial_score - mu_spatial) / sigma_spatial
      z_spectral = (spectral_score - mu_spectral) / sigma_spectral
      fused = (1 - lambda_spectral) * z_spatial + lambda_spectral * z_spectral
    Otherwise returns the raw spatial score (backward compatible).
    """
    clip_model.eval()
    projection_head.eval()
    p_real = p_real.to(device)
    p_fake = p_fake.to(device)
    spatial_scores, spectral_scores_list = [], []
    labels, sources, ids = [], [], []
    all_images_for_spectral = []

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
        proj = _project_image(projection_head, feats)

        if geometry == "hyperbolic":
            dist_real = hyperbolic_distance(proj, p_real, projection_head.ball)
            dist_fake_all = hyperbolic_distance(proj, p_fake, projection_head.ball)
            dist_fake = dist_fake_all.min(dim=1).values if dist_fake_all.ndim == 2 else dist_fake_all
        else:
            dist_real = torch.norm(proj - p_real.unsqueeze(0), dim=-1)
            if p_fake.ndim == 1:
                dist_fake = torch.norm(proj - p_fake.unsqueeze(0), dim=-1)
            else:
                dist_fake = torch.norm(proj.unsqueeze(1) - p_fake.unsqueeze(0), dim=-1).min(dim=1).values

        score = dist_real - dist_fake
        spatial_scores.extend(score.detach().cpu().numpy().tolist())
        labels.extend(lbs)
        sources.extend(srcs)
        ids.extend(rels)
        if spectral_scorer is not None and lambda_spectral > 0.0:
            all_images_for_spectral.extend(imgs)

    spatial_arr = np.array(spatial_scores, dtype=np.float32)

    if spectral_scorer is not None and lambda_spectral > 0.0 and len(all_images_for_spectral) > 0:
        spectral_arr = spectral_scorer.score(all_images_for_spectral)
        z_spatial = (spatial_arr - mu_spatial) / (sigma_spatial + 1e-8)
        z_spectral = (spectral_arr - mu_spectral) / (sigma_spectral + 1e-8)
        final_scores = (1.0 - lambda_spectral) * z_spatial + lambda_spectral * z_spectral
    else:
        final_scores = spatial_arr

    return np.array(labels), final_scores, sources, ids


def build_optimizer(clip_model, projection_head, cfg):
    lr_image = float(cfg.get("lr_image", 1e-5))
    lr_proj = float(cfg.get("lr_projection", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-4))

    trainable_clip = [p for p in clip_model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": trainable_clip, "lr": lr_image},
            {"params": projection_head.parameters(), "lr": lr_proj},
        ],
        weight_decay=wd,
    )


def train_one_epoch_multimodal(
    clip_model,
    processor,
    projection_head,
    p_real: torch.Tensor,
    p_fake: torch.Tensor,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    geometry: str,
    backbone_type: str,
    lambda_align: float,
    lambda_reg: float,
    margin: float,
    hard_negative_weight: float,
    warmup_mode: bool,
):
    clip_model.train()
    projection_head.train()
    running = 0.0
    n_batches = 0

    p_real = p_real.detach().to(device)
    p_fake = p_fake.detach().to(device)
    ball_or_none = projection_head.ball if geometry == "hyperbolic" else None

    for images, real_prompts, _, _, _, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            img_feats = _encode_image_features(clip_model, processor, images, device, backbone_type)
            z_img = _project_image(projection_head, img_feats)
            text_real_feats = encode_text_features(clip_model, processor, real_prompts, device, backbone_type)
            z_text_real = _project_text(projection_head, text_real_feats)

            if warmup_mode:
                loss = compute_alignment_loss_only(z_img, z_text_real, geometry, ball_or_none)
            else:
                loss = multimodal_loss(
                    z_img,
                    z_text_real,
                    p_real,
                    p_fake,
                    ball_or_none,
                    geometry,
                    lambda_align,
                    lambda_reg,
                    margin,
                    hard_negative_weight,
                )

        scaler.scale(loss).backward()
        if geometry == "hyperbolic":
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(projection_head.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        running += float(loss.item())
        n_batches += 1

    return running / max(n_batches, 1)


def evaluate_test_set(
    clip_model,
    processor,
    projection_head,
    p_real,
    p_fake,
    dataset,
    batch_size,
    device,
    geometry: str,
    thresholds: Dict[str, float],
    out_dir: Path,
    title_prefix: str,
    backbone_type: str,
    per_image_scores: Optional[Dict[str, Dict[str, List]]] = None,
    test_key: str = "",
    spectral_scorer: "SpectralAnomalyScorer | None" = None,
    mu_spatial: float = 0.0,
    sigma_spatial: float = 1.0,
    mu_spectral: float = 0.0,
    sigma_spectral: float = 1.0,
    lambda_spectral: float = 0.0,
):
    labels, scores, sources, ids = compute_anomaly_scores_multimodal(
        clip_model, processor, projection_head, p_real, p_fake, dataset, batch_size, device, geometry, backbone_type,
        spectral_scorer=spectral_scorer,
        mu_spatial=mu_spatial, sigma_spatial=sigma_spatial,
        mu_spectral=mu_spectral, sigma_spectral=sigma_spectral,
        lambda_spectral=lambda_spectral,
    )

    if per_image_scores is not None and test_key:
        per_image_scores[test_key] = {
            "scores": scores.tolist(),
            "labels": labels.tolist(),
            "image_paths": list(ids),
        }

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
    backbone_type = cfg.get("backbone", {}).get("type", "clip")
    backbone_model_name = cfg.get("backbone", {}).get("model_name", "openai/clip-vit-base-patch16")
    clip_model_name = cfg.get("clip_model_name", backbone_model_name)
    text_model_name = cfg.get("clip_text_model_name", clip_model_name)
    if backbone_type == "dinov2":
        clip_model_name = backbone_model_name
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

    prompt_file = cfg.get("prompt_file")
    if not prompt_file:
        raise RuntimeError("Missing required config key: prompt_file")
    warmup_epochs = int(cfg.get("warmup_epochs", 5))
    lambda_align = float(cfg.get("lambda_align", 0.5))
    lambda_reg = float(cfg.get("lambda_reg", 0.1))
    margin = float(cfg.get("margin", 0.5))
    hard_negative_weight = float(cfg.get("hard_negative_weight", 1.0))
    fake_num_prototypes = int(cfg.get("fake_num_prototypes", 2))
    enable_gan_prompt_enrichment = bool(cfg.get("enable_gan_prompt_enrichment", True))
    gan_like_augment_prob = float(cfg.get("gan_like_augment_prob", 0.35))
    fixed_fpr_targets = cfg.get("fixed_fpr_targets", [0.01, 0.05, 0.10])
    fixed_fpr_targets = [float(v) for v in fixed_fpr_targets if 0.0 < float(v) < 1.0]
    lambda_spectral = float(cfg.get("lambda_spectral", 0.3))
    enable_gan_image_prototype = bool(cfg.get("enable_gan_image_prototype", True))
    gan_image_prototype_samples = int(cfg.get("gan_image_prototype_samples", 200))

    set_global_determinism(seed)

    clip_model, processor = _load_image_backbone(
        clip_model_name,
        backbone_type,
        open_clip_pretrained,
        device,
        num_vit_layers=num_vit_layers,
        text_model_name=text_model_name,
    )
    configure_clip_trainability(clip_model, vision_mode, backbone_type)
    feature_dim = _get_clip_image_feature_dim(clip_model, backbone_type)
    text_dim = _get_clip_text_feature_dim(clip_model, backbone_type)

    if backbone_type == "dinov2":
        projection_head = SharedProjectionHead(
            image_input_dim=feature_dim,
            text_input_dim=text_dim,
            projection_dim=projection_dim,
            geometry=geometry,
            curvature=curvature,
            scale=scale,
        ).to(device)
    elif geometry == "euclidean":
        projection_head = EuclideanProjectionHead(input_dim=feature_dim, projection_dim=projection_dim).to(device)
    else:
        projection_head = HyperbolicProjectionHead(
            input_dim=feature_dim,
            projection_dim=projection_dim,
            curvature=curvature,
            scale=scale,
        ).to(device)

    prompt_path = Path(prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = PROJECT_ROOT / prompt_path
    real_prompts_pool, fake_prompts_pool = load_prompts(prompt_path)
    if enable_gan_prompt_enrichment:
        fake_prompts_pool = list(fake_prompts_pool) + GAN_STYLE_FAKE_PROMPTS
    n_train = len(fold["train_ids"])
    real_prompts_fold = list(itertools.islice(itertools.cycle(real_prompts_pool), n_train))
    fake_prompts_fold = list(itertools.islice(itertools.cycle(fake_prompts_pool), n_train))

    train_augmentor = GANLikeAugmentor(prob=gan_like_augment_prob)
    train_ds = MultimodalPromptDataset(
        dataset_root,
        fold["train_ids"],
        real_prompts_fold,
        fake_prompts_fold,
        image_augmentor=train_augmentor,
    )
    val_real_ds = ImagePathDataset(dataset_root, fold["val_ids"])
    val_eval_ds = ImagePathDataset(dataset_root, fold["val_eval_ids"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=False,
        worker_init_fn=worker_init_fn(seed),
        generator=build_loader_generator(seed),
    )

    optimizer = build_optimizer(clip_model, projection_head, cfg)
    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps_per_epoch * epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Fit spectral scorer on real training images (one-time, before training loop)
    print(f"  [{geometry}] Fitting spectral anomaly scorer on {len(train_ds)} training images...", flush=True)
    spectral_scorer = SpectralAnomalyScorer()
    train_images_for_spectral = [
        Image.open(dataset_root / train_ds.rel_paths[i]).convert("RGB")
        for i in range(len(train_ds))
    ]
    spectral_scorer.fit(train_images_for_spectral)
    del train_images_for_spectral

    best_auroc = -1.0
    best_payload = None
    log_rows = []
    warmup_rows = []
    prototype_distance_epochs = []
    prototype_distance_values = []
    best_val_labels = None
    best_val_scores = None
    best_val_thresholds = None

    for epoch in range(1, epochs + 1):
        p_real = compute_prototype_real(
            clip_model,
            processor,
            projection_head,
            train_ds,
            batch_size,
            device,
            geometry,
            backbone_type,
        )
        p_fake = compute_prototype_fake_multi(
            clip_model,
            processor,
            projection_head,
            train_ds,
            batch_size,
            device,
            geometry,
            backbone_type,
            fake_num_prototypes,
        )

        warmup_mode = epoch <= warmup_epochs
        train_loss = train_one_epoch_multimodal(
            clip_model,
            processor,
            projection_head,
            p_real,
            p_fake,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            geometry,
            backbone_type,
            lambda_align,
            lambda_reg,
            margin,
            hard_negative_weight,
            warmup_mode,
        )

        p_real = compute_prototype_real(
            clip_model,
            processor,
            projection_head,
            train_ds,
            batch_size,
            device,
            geometry,
            backbone_type,
        )
        p_fake = compute_prototype_fake_multi(
            clip_model,
            processor,
            projection_head,
            train_ds,
            batch_size,
            device,
            geometry,
            backbone_type,
            fake_num_prototypes,
        )

        if geometry == "hyperbolic":
            # ensure prototypes are on the same device as the projection head/ball
            p_real_dev = p_real.to(device) if p_real.device != device else p_real
            p_fake_dev = p_fake.to(device) if p_fake.device != device else p_fake
            dist_all = hyperbolic_distance(p_real_dev.unsqueeze(0), p_fake_dev, projection_head.ball)
            sep = float(dist_all.min().item() if dist_all.ndim > 1 else dist_all[0].item())
        else:
            p_real_dev = p_real.to(device)
            p_fake_dev = p_fake.to(device)
            if p_fake_dev.ndim == 1:
                sep = float(torch.norm(p_real_dev - p_fake_dev).item())
            else:
                sep = float(torch.norm(p_real_dev.unsqueeze(0) - p_fake_dev, dim=-1).min().item())
        prototype_distance_epochs.append(epoch)
        prototype_distance_values.append(sep)

        # GAN image-based fake prototype (computed fresh each epoch after model updates)
        if enable_gan_image_prototype:
            p_fake_gan_img = compute_prototype_fake_gan_image(
                clip_model, processor, projection_head, train_ds, batch_size,
                device, geometry, backbone_type,
                n_aug_samples=gan_image_prototype_samples, seed=seed + epoch,
            )
            # Stack with existing text-based prototypes
            if p_fake.ndim == 1:
                p_fake_combined = torch.stack([p_fake, p_fake_gan_img], dim=0)
            else:
                p_fake_combined = torch.cat([p_fake, p_fake_gan_img.unsqueeze(0)], dim=0)
        else:
            p_fake_combined = p_fake

        # Recompute val_in spatial scores with combined prototypes
        val_in_labels, val_in_scores, _, _ = compute_anomaly_scores_multimodal(
            clip_model, processor, projection_head, p_real, p_fake_combined,
            val_real_ds, batch_size, device, geometry, backbone_type,
        )

        # Compute spectral scores on val_in (real-only) for normalization stats
        val_in_images = [
            Image.open(dataset_root / val_real_ds.rel_paths[i]).convert("RGB")
            for i in range(len(val_real_ds))
        ]
        val_in_spectral = spectral_scorer.score(val_in_images)
        mu_spatial = float(np.mean(val_in_scores))
        sigma_spatial = float(max(np.std(val_in_scores), 0.05))
        mu_spectral = float(np.mean(val_in_spectral))
        sigma_spectral = float(max(np.std(val_in_spectral), 0.05))

        if epoch == 1 or epoch == warmup_epochs + 1:
            print(
                f"  [{geometry}] fold={fold['fold_index']} epoch={epoch} "
                f"norm_stats: mu_sp={mu_spatial:.4f} sig_sp={sigma_spatial:.4f} "
                f"mu_spec={mu_spectral:.4f} sig_spec={sigma_spectral:.4f}",
                flush=True,
            )
        del val_in_images

        # Recompute val_eval scores with combined prototypes and fusion
        val_eval_labels, val_eval_scores, val_eval_sources, _ = compute_anomaly_scores_multimodal(
            clip_model, processor, projection_head, p_real, p_fake_combined,
            val_eval_ds, batch_size, device, geometry, backbone_type,
            spectral_scorer=spectral_scorer,
            mu_spatial=mu_spatial, sigma_spatial=sigma_spatial,
            mu_spectral=mu_spectral, sigma_spectral=sigma_spectral,
            lambda_spectral=lambda_spectral,
        )

        default_threshold = max(float(np.percentile(val_in_scores, threshold_percentile)), 0.0)
        best_f1, best_j = calibrate_threshold(val_eval_labels, val_eval_scores)
        threshold_map = {
            "default": float(default_threshold),
            "f1": float(best_f1["threshold"]),
            "youden_j": float(best_j["threshold"]),
        }
        for fpr in fixed_fpr_targets:
            key = f"fpr_{int(round(fpr * 100.0))}"
            threshold_map[key] = float(np.quantile(val_in_scores, 1.0 - fpr))

        val_metrics = {name: compute_metrics(val_eval_labels, val_eval_scores, th) for name, th in threshold_map.items()}

        auroc = val_metrics["f1"]["auroc"]
        improved = auroc > best_auroc
        if improved:
            best_auroc = auroc
            best_payload = {
                "clip_model": clip_model.state_dict(),
                "projection_head": projection_head.state_dict(),
                "p_real": p_real.detach().cpu(),
                "p_fake": p_fake_combined.detach().cpu(),
                "val_real_scores": val_in_scores.tolist(),
                "spectral_scorer": spectral_scorer.to_dict(),
                "mu_spatial": mu_spatial,
                "sigma_spatial": sigma_spatial,
                "mu_spectral": mu_spectral,
                "sigma_spectral": sigma_spectral,
                "lambda_spectral": lambda_spectral,
                "default_threshold": default_threshold,
                "calibrated_threshold_f1": best_f1["threshold"],
                "calibrated_threshold_youden_j": best_j["threshold"],
                "thresholds": threshold_map,
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
            for k, v in threshold_map.items():
                if k not in best_val_thresholds:
                    best_val_thresholds[k] = float(v)
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
                "warmup_mode": bool(warmup_mode),
            }
        )

        if warmup_mode:
            warmup_rows.append({"epoch": epoch, "train_loss": round(train_loss, 6)})

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

    if warmup_rows:
        save_loss_curve(warmup_rows, fold_dir / "warmup_loss_curve.png")

    with (fold_dir / "prototype_distances.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "epochs": prototype_distance_epochs,
                "prototype_separation": prototype_distance_values,
            },
            f,
            indent=2,
        )

    # Save fold-level validation interpretation artifacts for all thresholds.
    val_threshold_rows = []
    for threshold_name in best_val_thresholds.keys():
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
                "thresholds": best_val_thresholds,
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
    backbone_type = cfg.get("backbone", {}).get("type", "clip")
    backbone_model_name = cfg.get("backbone", {}).get("model_name", "openai/clip-vit-base-patch16")
    clip_model_name = cfg.get("clip_model_name", backbone_model_name)
    text_model_name = cfg.get("clip_text_model_name", clip_model_name)
    if backbone_type == "dinov2":
        clip_model_name = backbone_model_name
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
        text_model_name=text_model_name,
    )
    configure_clip_trainability(clip_model, vision_mode, backbone_type)
    feature_dim = _get_clip_image_feature_dim(clip_model, backbone_type)
    text_dim = _get_clip_text_feature_dim(clip_model, backbone_type)

    if backbone_type == "dinov2":
        projection_head = SharedProjectionHead(
            image_input_dim=feature_dim,
            text_input_dim=text_dim,
            projection_dim=projection_dim,
            geometry=geometry,
            curvature=curvature,
            scale=scale,
        ).to(device)
    elif geometry == "euclidean":
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
    p_real = ckpt["p_real"].to(device)
    p_fake = ckpt["p_fake"].to(device)
    if "thresholds" in ckpt and isinstance(ckpt["thresholds"], dict):
        thresholds = {str(k): float(v) for k, v in ckpt["thresholds"].items()}
    else:
        thresholds = {
            "f1": float(ckpt["calibrated_threshold_f1"]),
            "youden_j": float(ckpt["calibrated_threshold_youden_j"]),
            "default": float(ckpt["default_threshold"]),
        }
    val_real_scores = ckpt.get("val_real_scores", [])
    spectral_scorer = SpectralAnomalyScorer.from_dict(ckpt.get("spectral_scorer", {}))
    mu_spatial = float(ckpt.get("mu_spatial", 0.0))
    sigma_spatial = float(ckpt.get("sigma_spatial", 1.0))
    mu_spectral = float(ckpt.get("mu_spectral", 0.0))
    sigma_spectral = float(ckpt.get("sigma_spectral", 1.0))
    lambda_spectral = float(ckpt.get("lambda_spectral", 0.0))
    return (clip_model, processor, projection_head, p_real, p_fake, thresholds, val_real_scores,
            spectral_scorer, mu_spatial, sigma_spatial, mu_spectral, sigma_spectral, lambda_spectral)


def run_geometry(cfg: dict, manifest: dict, dataset_root: Path, geometry: str, geometry_dir: Path) -> Dict:
    def _threshold_order_key(name: str):
        if name == "default":
            return (0, 0.0)
        if name == "f1":
            return (1, 0.0)
        if name == "youden_j":
            return (2, 0.0)
        if name.startswith("fpr_"):
            try:
                return (3, float(name.split("_", 1)[1]))
            except Exception:
                return (4, name)
        return (5, name)

    fold_summaries = []
    for fold in manifest["cv_folds"]:
        fold_dir = geometry_dir / f"fold_{fold['fold_index']}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_summaries.append(run_fold(cfg, dataset_root, geometry, fold, fold_dir))

    best_fold = max(fold_summaries, key=lambda x: (x["best_val_auroc"], x["best_val_auprc"]))
    with (geometry_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"folds": fold_summaries, "best_fold": best_fold}, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (clip_model, processor, projection_head, p_real, p_fake, thresholds, val_real_scores,
     spectral_scorer, mu_spatial, sigma_spatial, mu_spectral, sigma_spectral, lambda_spectral) = load_best_model(
        cfg,
        geometry,
        Path(best_fold["checkpoint_path"]),
        device,
    )
    backbone_type = cfg.get("backbone", {}).get("type", "clip")
    batch_size = int(cfg.get("batch_size", 32))

    test_results = {}
    summary_rows = []
    per_image_scores_payload = {
        "val_real_scores": val_real_scores,
        "test_gan": {"scores": [], "labels": [], "image_paths": []},
        "test_ldm": {"scores": [], "labels": [], "image_paths": []},
        "test_mls": {"scores": [], "labels": [], "image_paths": []},
        "test_allfake": {"scores": [], "labels": [], "image_paths": []},
    }

    for test_name, test_spec in manifest["test_sets"].items():
        test_dir = geometry_dir / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        rel_paths = sorted(test_spec["real_ids"] + test_spec["fake_ids"])
        dataset = ImagePathDataset(dataset_root, rel_paths)

        result = evaluate_test_set(
            clip_model,
            processor,
            projection_head,
            p_real,
            p_fake,
            dataset,
            batch_size,
            device,
            geometry,
            thresholds,
            test_dir,
            f"{geometry} {test_name}",
            backbone_type,
            per_image_scores=per_image_scores_payload,
            test_key=test_name,
            spectral_scorer=spectral_scorer,
            mu_spatial=mu_spatial, sigma_spatial=sigma_spatial,
            mu_spectral=mu_spectral, sigma_spectral=sigma_spectral,
            lambda_spectral=lambda_spectral,
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
        for threshold_name in sorted(thresholds.keys(), key=_threshold_order_key):
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

    with (geometry_dir / "per_image_scores.json").open("w", encoding="utf-8") as f:
        json.dump(per_image_scores_payload, f, indent=2)

        test_results[test_name] = payload
        print(
            f"[TEST] geometry={geometry} set={test_name} auroc={result['auroc']:.4f} auprc={result['auprc']:.4f} "
            f"f1@f1={result['threshold_results']['f1']['f1']:.4f} f1@j={result['threshold_results']['youden_j']['f1']:.4f}",
            flush=True,
        )

    write_summary_csv(summary_rows, geometry_dir / "summary_8run_slice.csv")
    with (geometry_dir / "test_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"best_fold": best_fold, "tests": test_results}, f, indent=2)

    all_fold_results = {}
    fold_metric_names = [
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

    for fold_summary in fold_summaries:
        fold_index = fold_summary["fold_index"]
        (fold_model, fold_processor, fold_projection_head, fold_p_real, fold_p_fake, fold_thresholds, fold_val_real_scores,
         fold_spectral_scorer, fold_mu_spatial, fold_sigma_spatial, fold_mu_spectral, fold_sigma_spectral, fold_lambda_spectral) = load_best_model(
            cfg,
            geometry,
            Path(fold_summary["checkpoint_path"]),
            device,
        )
        all_fold_results[fold_index] = {}

        for test_name, test_spec in manifest["test_sets"].items():
            test_dir = geometry_dir / test_name
            rel_paths = sorted(test_spec["real_ids"] + test_spec["fake_ids"])
            dataset = ImagePathDataset(dataset_root, rel_paths)

            result = evaluate_test_set(
                fold_model,
                fold_processor,
                fold_projection_head,
                fold_p_real,
                fold_p_fake,
                dataset,
                batch_size,
                device,
                geometry,
                fold_thresholds,
                test_dir,
                f"{geometry} {test_name} fold-{fold_index}",
                backbone_type,
                spectral_scorer=fold_spectral_scorer,
                mu_spatial=fold_mu_spatial, sigma_spatial=fold_sigma_spatial,
                mu_spectral=fold_mu_spectral, sigma_spectral=fold_sigma_spectral,
                lambda_spectral=fold_lambda_spectral,
            )

            fold_metrics = {
                "auroc": result["auroc"],
                "auprc": result["auprc"],
                "accuracy_default": result["threshold_results"]["default"]["accuracy"],
                "f1_default": result["threshold_results"]["default"]["f1"],
                "sensitivity_default": result["threshold_results"]["default"]["sensitivity"],
                "specificity_default": result["threshold_results"]["default"]["specificity"],
                "accuracy_f1": result["threshold_results"]["f1"]["accuracy"],
                "f1_f1": result["threshold_results"]["f1"]["f1"],
                "sensitivity_f1": result["threshold_results"]["f1"]["sensitivity"],
                "specificity_f1": result["threshold_results"]["f1"]["specificity"],
                "accuracy_youden_j": result["threshold_results"]["youden_j"]["accuracy"],
                "f1_youden_j": result["threshold_results"]["youden_j"]["f1"],
                "sensitivity_youden_j": result["threshold_results"]["youden_j"]["sensitivity"],
                "specificity_youden_j": result["threshold_results"]["youden_j"]["specificity"],
            }
            all_fold_results[fold_index][test_name] = fold_metrics

    mean_std_rows = []
    mean_std_payload = {
        "geometry": geometry,
        "n_folds": len(fold_summaries),
        "results": {},
    }

    for test_name in manifest["test_sets"].keys():
        mean_std_payload["results"][test_name] = {}
        for metric in fold_metric_names:
            values = [all_fold_results[fold_summary["fold_index"]][test_name][metric] for fold_summary in fold_summaries]
            metric_mean = float(np.mean(values))
            metric_std = float(np.std(values))
            mean_std_payload["results"][test_name][metric] = {
                "mean": round(metric_mean, 6),
                "std": round(metric_std, 6),
            }
            mean_std_rows.append(
                {
                    "test_set": test_name,
                    "metric": metric,
                    "mean": round(metric_mean, 6),
                    "std": round(metric_std, 6),
                }
            )

    with (geometry_dir / "mean_std_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_set", "metric", "mean", "std"])
        writer.writeheader()
        writer.writerows(mean_std_rows)

    with (geometry_dir / "mean_std_summary.json").open("w", encoding="utf-8") as f:
        json.dump(mean_std_payload, f, indent=2)

    print(f"[MEAN±STD] {geometry}", flush=True)
    print(f"{'test_set':<15} {'auroc':<18} {'auprc':<18} {'f1_youden_j':<18}", flush=True)
    for test_name in manifest["test_sets"].keys():
        auroc_stats = mean_std_payload["results"][test_name]["auroc"]
        auprc_stats = mean_std_payload["results"][test_name]["auprc"]
        f1_j_stats = mean_std_payload["results"][test_name]["f1_youden_j"]
        print(
            f"{test_name:<15} "
            f"{auroc_stats['mean']:.4f}±{auroc_stats['std']:.4f}   "
            f"{auprc_stats['mean']:.4f}±{auprc_stats['std']:.4f}   "
            f"{f1_j_stats['mean']:.4f}±{f1_j_stats['std']:.4f}",
            flush=True,
        )

    return {"geometry": geometry, "best_fold": best_fold, "summary_rows": summary_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Multimodal Hyperbolic Prototype CLIP orchestrator")
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

    run_name = cfg.get("run_name", f"multimodal_hyperbolic_prototype_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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

    geometries = cfg.get("geometries", [cfg.get("geometry", "hyperbolic")])
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