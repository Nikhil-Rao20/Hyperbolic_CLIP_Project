from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_SIMPLENET = PROJECT_ROOT / "third_party" / "SimpleNet"
DEFAULT_TEST_SET_ORDER = ["test_allfake", "test_gan", "test_ldm", "test_mls"]
SUMMARY_METRIC_KEYS = [
    "auroc",
    "auprc",
    "accuracy_default",
    "precision_default",
    "recall_default",
    "f1_default",
    "sensitivity_default",
    "specificity_default",
    "PPV_default",
    "NPV_default",
    "accuracy_f1",
    "precision_f1",
    "recall_f1",
    "f1_f1",
    "sensitivity_f1",
    "specificity_f1",
    "PPV_f1",
    "NPV_f1",
    "accuracy_youden_j",
    "precision_youden_j",
    "recall_youden_j",
    "f1_youden_j",
    "sensitivity_youden_j",
    "specificity_youden_j",
    "PPV_youden_j",
    "NPV_youden_j",
]


def _load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _validate_device_mode(device: str) -> str:
    mode = str(device).lower()
    if mode not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device mode: {device}. Use one of: auto, cpu, cuda")
    if mode == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this system")
    return mode


def _resolve_torch_device(device_mode: str) -> torch.device:
    if device_mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_mode == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def _set_global_determinism(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _clear_shadowed_modules(prefixes: Sequence[str]) -> None:
    for name in list(sys.modules.keys()):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(name, None)


def _load_simplenet_modules():
    if not THIRD_PARTY_SIMPLENET.exists():
        raise FileNotFoundError(
            f"SimpleNet official source not found at {THIRD_PARTY_SIMPLENET}. "
            "Run: python scripts/setup_official_baselines.py"
        )

    module_root = THIRD_PARTY_SIMPLENET.as_posix()
    if module_root not in sys.path:
        sys.path.insert(0, module_root)

    _clear_shadowed_modules(["backbones", "common", "metrics", "simplenet", "utils", "resnet"])

    backbones = importlib.import_module("backbones")
    simplenet_module = importlib.import_module("simplenet")
    utils_module = importlib.import_module("utils")
    return backbones, simplenet_module, utils_module


def _install_simplenet_nan_guards(simplenet_module) -> None:
    if getattr(simplenet_module.SimpleNet, "_hyperclip_nan_guarded", False):
        return

    metrics_module = simplenet_module.metrics

    def _safe_evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt):
        _ = test_data
        _ = features

        score_vals = np.asarray(scores, dtype=np.float64).reshape(-1)
        label_vals = np.asarray(labels_gt, dtype=np.int64).reshape(-1)
        if score_vals.shape[0] != label_vals.shape[0]:
            n = min(score_vals.shape[0], label_vals.shape[0])
            score_vals = score_vals[:n]
            label_vals = label_vals[:n]
        if score_vals.size == 0:
            return 0.5, -1.0, -1.0

        finite_mask = np.isfinite(score_vals)
        if not np.all(finite_mask):
            fill = float(np.median(score_vals[finite_mask])) if np.any(finite_mask) else 0.0
            score_vals = np.nan_to_num(score_vals, nan=fill, posinf=fill, neginf=fill)

        s_min = float(np.min(score_vals))
        s_max = float(np.max(score_vals))
        s_den = max(s_max - s_min, 1e-12)
        score_vals = (score_vals - s_min) / s_den

        try:
            auroc = float(metrics_module.compute_imagewise_retrieval_metrics(score_vals, label_vals)["auroc"])
        except Exception:
            auroc = 0.5

        if len(masks_gt) > 0:
            seg = np.asarray(segmentations, dtype=np.float64)
            if seg.size == 0:
                return auroc, -1.0, -1.0

            flat = seg.reshape(len(seg), -1)
            seg_min = flat.min(axis=1).reshape(-1, 1, 1, 1)
            seg_max = flat.max(axis=1).reshape(-1, 1, 1, 1)
            seg_den = np.maximum(seg_max - seg_min, 1e-12)
            norm_segmentations = (seg - seg_min) / seg_den
            norm_segmentations = np.nan_to_num(norm_segmentations, nan=0.0, posinf=1.0, neginf=0.0)

            try:
                pixel_scores = metrics_module.compute_pixelwise_retrieval_metrics(norm_segmentations, masks_gt)
                full_pixel_auroc = float(pixel_scores.get("auroc", -1.0))
            except Exception:
                full_pixel_auroc = -1.0

            try:
                pro = float(metrics_module.compute_pro(np.squeeze(np.array(masks_gt)), norm_segmentations))
            except Exception:
                pro = -1.0
        else:
            full_pixel_auroc = -1.0
            pro = -1.0

        return auroc, full_pixel_auroc, pro

    simplenet_module.SimpleNet._evaluate = _safe_evaluate
    simplenet_module.SimpleNet._hyperclip_nan_guarded = True


def _label_from_rel_path(rel_path: str) -> int:
    normalized = rel_path.replace("\\", "/")
    parts_lower = [part.lower() for part in Path(normalized).parts]
    is_real = any("real" in part for part in parts_lower) and not any("fake" in part for part in parts_lower)
    return 0 if is_real else 1


class ManifestSimpleNetDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        rel_paths: Sequence[str],
        resize: int,
        imagesize: int,
        augment: bool = False,
        rotate_degrees: int = 0,
        translate: float = 0.0,
        scale: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        gray: float = 0.0,
        hflip: float = 0.0,
        vflip: float = 0.0,
    ) -> None:
        self.dataset_root = dataset_root
        self.rel_paths = list(rel_paths)
        self.imagesize = (3, int(imagesize), int(imagesize))
        self.transform_std = [0.229, 0.224, 0.225]
        self.transform_mean = [0.485, 0.456, 0.406]

        img_transforms = [transforms.Resize(int(resize))]
        if augment:
            img_transforms.extend(
                [
                    transforms.ColorJitter(brightness, contrast, saturation),
                    transforms.RandomHorizontalFlip(hflip),
                    transforms.RandomVerticalFlip(vflip),
                    transforms.RandomGrayscale(gray),
                    transforms.RandomAffine(
                        rotate_degrees,
                        translate=(translate, translate),
                        scale=(1.0 - scale, 1.0 + scale),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                ]
            )
        img_transforms.extend(
            [
                transforms.CenterCrop(int(imagesize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.transform_mean, std=self.transform_std),
            ]
        )

        self.transform_img = transforms.Compose(img_transforms)

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, idx: int) -> Dict:
        rel_path = self.rel_paths[idx]
        abs_path = self.dataset_root / rel_path
        if not abs_path.exists():
            raise FileNotFoundError(f"Missing source image: {abs_path}")

        image = Image.open(abs_path).convert("RGB")
        image = self.transform_img(image)
        label = _label_from_rel_path(rel_path)
        anomaly_name = "good" if label == 0 else "defect"

        return {
            "image": image,
            "classname": "protocol",
            "anomaly": anomaly_name,
            "is_anomaly": int(label),
            "image_name": Path(rel_path).name,
            "image_path": abs_path.as_posix(),
        }


def _build_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, device: torch.device) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": (device.type == "cuda"),
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _to_scalar_scores(scores: Sequence) -> np.ndarray:
    vals = []
    for score in scores:
        arr = np.asarray(score)
        if arr.size == 0:
            vals.append(0.0)
        else:
            vals.append(float(arr.reshape(-1)[0]))
    return np.array(vals, dtype=np.float64)


def _predict_image_scores(model, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    scores, _, _, labels_gt, _ = model.predict(loader)
    labels = np.array([int(v) for v in labels_gt], dtype=np.int64)
    score_vals = _to_scalar_scores(scores)

    if labels.shape[0] != score_vals.shape[0]:
        raise RuntimeError(
            f"SimpleNet returned mismatched labels/scores lengths: {labels.shape[0]} vs {score_vals.shape[0]}"
        )
    if labels.shape[0] == 0:
        raise RuntimeError("SimpleNet returned empty labels/scores")
    return labels, score_vals


def _load_best_checkpoint_into_model(model, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SimpleNet best checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "discriminator" in state_dict:
        model.discriminator.load_state_dict(state_dict["discriminator"])
        if "pre_projection" in state_dict and getattr(model, "pre_proj", 0) > 0:
            model.pre_projection.load_state_dict(state_dict["pre_projection"])
    else:
        model.load_state_dict(state_dict, strict=False)


def _predict_from_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores > threshold).astype(np.int64)


def _compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict:
    preds = _predict_from_threshold(scores, threshold)
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }

    try:
        out["auroc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        out["auroc"] = 0.0
    try:
        out["auprc"] = float(average_precision_score(labels, scores))
    except ValueError:
        out["auprc"] = 0.0

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    out["confusion_matrix"] = cm.tolist()
    out["specificity"] = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
    out["sensitivity"] = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    out["PPV"] = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    out["NPV"] = round(tn / (tn + fn), 4) if (tn + fn) > 0 else 0.0
    out["classification_report"] = classification_report(
        labels,
        preds,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        output_dict=True,
        zero_division=0,
    )
    return out


def _calibrate_thresholds(labels: np.ndarray, scores: np.ndarray, default_threshold: float) -> Dict[str, float]:
    uniq = np.unique(scores)
    candidates = uniq if len(uniq) < 200 else np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 200)))
    if len(candidates) == 0:
        raise RuntimeError("No threshold candidates available for SimpleNet metrics calibration")

    best_f1 = {"threshold": float(candidates[0]), "f1": -1.0}
    best_j = {"threshold": float(candidates[0]), "youden_j": -2.0}

    for th in candidates:
        preds = _predict_from_threshold(scores, float(th))
        f1 = float(f1_score(labels, preds, zero_division=0))
        rec = float(recall_score(labels, preds, zero_division=0))
        spe = float(recall_score(labels, preds, pos_label=0, zero_division=0))
        j = rec + spe - 1.0
        if f1 > best_f1["f1"]:
            best_f1 = {"threshold": float(th), "f1": f1}
        if j > best_j["youden_j"]:
            best_j = {"threshold": float(th), "youden_j": j}

    return {
        "default": float(default_threshold),
        "f1": float(best_f1["threshold"]),
        "youden_j": float(best_j["threshold"]),
    }


def _build_summary_row(fold_index: int, test_name: str, test_spec: dict, threshold_results: Dict[str, Dict]) -> Dict:
    default_metrics = threshold_results["default"]
    f1_metrics = threshold_results["f1"]
    j_metrics = threshold_results["youden_j"]
    return {
        "geometry": "simplenet_official",
        "fold_index": int(fold_index),
        "test_set": test_name,
        "n_real": int(test_spec["n_real"]),
        "n_fake": int(test_spec["n_fake"]),
        "auroc": round(default_metrics["auroc"], 6),
        "auprc": round(default_metrics["auprc"], 6),
        "accuracy_default": round(default_metrics["accuracy"], 6),
        "precision_default": round(default_metrics["precision"], 6),
        "recall_default": round(default_metrics["recall"], 6),
        "f1_default": round(default_metrics["f1"], 6),
        "sensitivity_default": round(default_metrics["sensitivity"], 6),
        "specificity_default": round(default_metrics["specificity"], 6),
        "PPV_default": round(default_metrics["PPV"], 6),
        "NPV_default": round(default_metrics["NPV"], 6),
        "accuracy_f1": round(f1_metrics["accuracy"], 6),
        "precision_f1": round(f1_metrics["precision"], 6),
        "recall_f1": round(f1_metrics["recall"], 6),
        "f1_f1": round(f1_metrics["f1"], 6),
        "sensitivity_f1": round(f1_metrics["sensitivity"], 6),
        "specificity_f1": round(f1_metrics["specificity"], 6),
        "PPV_f1": round(f1_metrics["PPV"], 6),
        "NPV_f1": round(f1_metrics["NPV"], 6),
        "accuracy_youden_j": round(j_metrics["accuracy"], 6),
        "precision_youden_j": round(j_metrics["precision"], 6),
        "recall_youden_j": round(j_metrics["recall"], 6),
        "f1_youden_j": round(j_metrics["f1"], 6),
        "sensitivity_youden_j": round(j_metrics["sensitivity"], 6),
        "specificity_youden_j": round(j_metrics["specificity"], 6),
        "PPV_youden_j": round(j_metrics["PPV"], 6),
        "NPV_youden_j": round(j_metrics["NPV"], 6),
    }


def _write_threshold_metrics_csv(test_dir: Path, threshold_rows: List[Dict]) -> None:
    fieldnames = [
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
    ]
    with (test_dir / "threshold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(threshold_rows)


def _write_val_threshold_metrics_csv(fold_dir: Path, threshold_rows: List[Dict]) -> None:
    fieldnames = [
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
    ]
    with (fold_dir / "val_threshold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(threshold_rows)


def _write_per_fold_summary_csv(rows: List[Dict], out_csv: Path) -> None:
    fieldnames = ["geometry", "fold_index", "test_set", "n_real", "n_fake", *SUMMARY_METRIC_KEYS]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_fold_rows(fold_rows: List[Dict], test_set_order: Sequence[str]) -> tuple[List[Dict], List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in fold_rows:
        grouped[row["test_set"]].append(row)

    mean_std_rows: List[Dict] = []
    stats_rows: List[Dict] = []
    for test_name in test_set_order:
        rows = grouped.get(test_name, [])
        if not rows:
            continue

        mean_std_row = {
            "geometry": "simplenet_official",
            "test_set": test_name,
            "n_folds": len(rows),
            "n_real": int(rows[0]["n_real"]),
            "n_fake": int(rows[0]["n_fake"]),
        }
        stats_row = {
            "geometry": "simplenet_official",
            "test_set": test_name,
            "n_folds": len(rows),
            "n_real": int(rows[0]["n_real"]),
            "n_fake": int(rows[0]["n_fake"]),
        }

        for key in SUMMARY_METRIC_KEYS:
            vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            mean_std_row[key] = f"{mean_val:.6f} ± {std_val:.6f}"
            stats_row[f"{key}_mean"] = round(mean_val, 6)
            stats_row[f"{key}_std"] = round(std_val, 6)

        mean_std_rows.append(mean_std_row)
        stats_rows.append(stats_row)

    return mean_std_rows, stats_rows


def _write_mean_std_summary_csv(rows: List[Dict], out_csv: Path) -> None:
    fieldnames = ["geometry", "test_set", "n_folds", "n_real", "n_fake", *SUMMARY_METRIC_KEYS]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_stats_summary_csv(rows: List[Dict], out_csv: Path) -> None:
    metric_fields: List[str] = []
    for key in SUMMARY_METRIC_KEYS:
        metric_fields.extend([f"{key}_mean", f"{key}_std"])
    fieldnames = ["geometry", "test_set", "n_folds", "n_real", "n_fake", *metric_fields]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_test_payload(test_dir: Path, payload: Dict) -> None:
    with (test_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_official_simplenet(
    manifest_path: Path,
    dataset_root: Path,
    output_root: Path,
    device: str = "auto",
    batch_size: int = 8,
    num_workers: int = 0,
    resize: int = 329,
    imagesize: int = 288,
    threshold_percentile: float = 95.0,
    test_set_order: Sequence[str] = DEFAULT_TEST_SET_ORDER,
    seed: int | None = None,
    backbone_name: str = "wideresnet50",
    layers_to_extract_from: Sequence[str] = ("layer2", "layer3"),
    pretrain_embed_dimension: int = 1536,
    target_embed_dimension: int = 1536,
    patchsize: int = 3,
    embedding_size: int = 256,
    meta_epochs: int = 40,
    aed_meta_epochs: int = 1,
    gan_epochs: int = 4,
    noise_std: float = 0.015,
    dsc_layers: int = 2,
    dsc_hidden: int | None = 1024,
    dsc_margin: float = 0.5,
    dsc_lr: float = 0.0002,
    auto_noise: float = 0.0,
    train_backbone: bool = False,
    cos_lr: bool = False,
    pre_proj: int = 1,
    proj_layer_type: int = 0,
    mix_noise: int = 1,
    augment_train: bool = False,
    rotate_degrees: int = 0,
    translate: float = 0.0,
    scale: float = 0.0,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    gray: float = 0.0,
    hflip: float = 0.0,
    vflip: float = 0.0,
    reuse_checkpoint_if_exists: bool = False,
) -> int:
    device_mode = _validate_device_mode(device)
    torch_device = _resolve_torch_device(device_mode)
    manifest = _load_manifest(manifest_path)
    base_seed = int(manifest.get("seed", 42) if seed is None else seed)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    backbones, simplenet_module, simplenet_utils = _load_simplenet_modules()
    _install_simplenet_nan_guards(simplenet_module)

    run_dir = output_root / "simplenet_official"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_fold_summary_rows: List[Dict] = []
    fold_run_summaries: List[Dict] = []

    for fold in manifest["cv_folds"]:
        fold_index = int(fold["fold_index"])
        fold_seed = int(base_seed + (fold_index * 100))
        _set_global_determinism(fold_seed, torch_device)
        simplenet_utils.fix_seeds(
            fold_seed,
            with_torch=True,
            with_cuda=(torch_device.type == "cuda"),
        )

        fold_dir = run_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_ids = list(fold["train_ids"])
        val_real_ids = list(fold["val_ids"])
        val_eval_ids = list(fold["val_eval_ids"])

        if len(train_ids) == 0:
            raise RuntimeError(f"Fold {fold_index} has empty train_ids")
        if any(_label_from_rel_path(rel) != 0 for rel in train_ids):
            raise RuntimeError(f"Fold {fold_index} train_ids contains fake samples; expected real-only training")

        train_ds = ManifestSimpleNetDataset(
            dataset_root=dataset_root,
            rel_paths=train_ids,
            resize=resize,
            imagesize=imagesize,
            augment=augment_train,
            rotate_degrees=rotate_degrees,
            translate=translate,
            scale=scale,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            gray=gray,
            hflip=hflip,
            vflip=vflip,
        )
        val_real_ds = ManifestSimpleNetDataset(
            dataset_root=dataset_root,
            rel_paths=val_real_ids,
            resize=resize,
            imagesize=imagesize,
            augment=False,
        )
        val_eval_ds = ManifestSimpleNetDataset(
            dataset_root=dataset_root,
            rel_paths=val_eval_ids,
            resize=resize,
            imagesize=imagesize,
            augment=False,
        )

        train_loader = _build_loader(train_ds, batch_size, num_workers, shuffle=True, device=torch_device)
        val_real_loader = _build_loader(val_real_ds, batch_size, num_workers, shuffle=False, device=torch_device)
        val_eval_loader = _build_loader(val_eval_ds, batch_size, num_workers, shuffle=False, device=torch_device)

        backbone = backbones.load(backbone_name)
        backbone.name = backbone_name
        backbone.seed = None

        simplenet_inst = simplenet_module.SimpleNet(torch_device)
        simplenet_inst.load(
            backbone=backbone,
            layers_to_extract_from=list(layers_to_extract_from),
            device=torch_device,
            input_shape=(3, int(imagesize), int(imagesize)),
            pretrain_embed_dimension=int(pretrain_embed_dimension),
            target_embed_dimension=int(target_embed_dimension),
            patchsize=int(patchsize),
            embedding_size=int(embedding_size),
            meta_epochs=int(meta_epochs),
            aed_meta_epochs=int(aed_meta_epochs),
            gan_epochs=int(gan_epochs),
            noise_std=float(noise_std),
            dsc_layers=int(dsc_layers),
            dsc_hidden=None if dsc_hidden is None else int(dsc_hidden),
            dsc_margin=float(dsc_margin),
            dsc_lr=float(dsc_lr),
            auto_noise=float(auto_noise),
            train_backbone=bool(train_backbone),
            cos_lr=bool(cos_lr),
            pre_proj=int(pre_proj),
            proj_layer_type=int(proj_layer_type),
            mix_noise=int(mix_noise),
        )

        fold_model_root = fold_dir / "models"
        if fold_model_root.exists() and not reuse_checkpoint_if_exists:
            shutil.rmtree(fold_model_root)

        simplenet_inst.set_model_dir((fold_model_root / "0").as_posix(), f"protocol_fold_{fold_index}")

        print(
            f"[SimpleNet] fold={fold_index} training start | train={len(train_ds)} val_eval={len(val_eval_ds)} "
            f"device={torch_device} backbone={backbone_name}",
            flush=True,
        )
        best_record = simplenet_inst.train(train_loader, val_eval_loader)

        ckpt_path = Path(simplenet_inst.ckpt_dir) / "ckpt.pth"
        _load_best_checkpoint_into_model(simplenet_inst, ckpt_path, torch_device)

        val_eval_labels, val_eval_scores = _predict_image_scores(simplenet_inst, val_eval_loader)

        if len(val_real_ds) > 0:
            _, val_real_scores = _predict_image_scores(simplenet_inst, val_real_loader)
            default_threshold = float(np.percentile(val_real_scores, float(threshold_percentile)))
        else:
            default_threshold = 0.5

        thresholds = _calibrate_thresholds(val_eval_labels, val_eval_scores, default_threshold=default_threshold)

        val_threshold_results = {
            name: _compute_metrics(val_eval_labels, val_eval_scores, float(thresholds[name]))
            for name in ["default", "f1", "youden_j"]
        }

        val_threshold_rows: List[Dict] = []
        for threshold_name in ["default", "f1", "youden_j"]:
            metrics = val_threshold_results[threshold_name]
            val_threshold_rows.append(
                {
                    "threshold_name": threshold_name,
                    "threshold_value": float(thresholds[threshold_name]),
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
            with (fold_dir / f"val_classification_report_{threshold_name}.json").open("w", encoding="utf-8") as f:
                json.dump(metrics["classification_report"], f, indent=2)
        _write_val_threshold_metrics_csv(fold_dir, val_threshold_rows)

        fold_run_summaries.append(
            {
                "fold_index": fold_index,
                "fold_seed": fold_seed,
                "checkpoint_path": ckpt_path.as_posix(),
                "best_record": [float(x) for x in best_record],
                "thresholds": thresholds,
                "val_metrics": val_threshold_results,
            }
        )

        for test_name in test_set_order:
            test_spec = manifest["test_sets"][test_name]
            rel_paths = sorted(test_spec["real_ids"] + test_spec["fake_ids"])
            test_ds = ManifestSimpleNetDataset(
                dataset_root=dataset_root,
                rel_paths=rel_paths,
                resize=resize,
                imagesize=imagesize,
                augment=False,
            )
            test_loader = _build_loader(test_ds, batch_size, num_workers, shuffle=False, device=torch_device)

            labels, scores = _predict_image_scores(simplenet_inst, test_loader)

            threshold_results: Dict[str, Dict] = {}
            threshold_metrics_rows: List[Dict] = []
            for threshold_name in ["default", "f1", "youden_j"]:
                threshold = float(thresholds[threshold_name])
                metrics = _compute_metrics(labels, scores, threshold)
                threshold_results[threshold_name] = metrics
                threshold_metrics_rows.append(
                    {
                        "threshold_name": threshold_name,
                        "threshold_value": threshold,
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

            test_dir = fold_dir / test_name
            test_dir.mkdir(parents=True, exist_ok=True)
            _write_threshold_metrics_csv(test_dir, threshold_metrics_rows)
            for threshold_name in ["default", "f1", "youden_j"]:
                with (test_dir / f"classification_report_{threshold_name}.json").open("w", encoding="utf-8") as f:
                    json.dump(threshold_results[threshold_name]["classification_report"], f, indent=2)

            default_metrics = threshold_results["default"]
            _save_test_payload(
                test_dir,
                {
                    "fold_index": fold_index,
                    "test_set": test_name,
                    "n_real": int(test_spec["n_real"]),
                    "n_fake": int(test_spec["n_fake"]),
                    "auroc": float(default_metrics["auroc"]),
                    "auprc": float(default_metrics["auprc"]),
                    "thresholds": thresholds,
                    "threshold_results": threshold_results,
                    "official_repo": "DonaldRR/SimpleNet",
                    "device": device_mode,
                    "backbone_name": backbone_name,
                    "layers_to_extract_from": list(layers_to_extract_from),
                    "fold_seed": fold_seed,
                    "checkpoint_path": ckpt_path.as_posix(),
                    "best_record": [float(x) for x in best_record],
                },
            )

            all_fold_summary_rows.append(_build_summary_row(fold_index, test_name, test_spec, threshold_results))
            print(
                f"[SimpleNet] fold={fold_index} set={test_name} "
                f"auroc={default_metrics['auroc']:.4f} auprc={default_metrics['auprc']:.4f} "
                f"f1@default={default_metrics['f1']:.4f}",
                flush=True,
            )

    _write_per_fold_summary_csv(all_fold_summary_rows, run_dir / "final_8run_summary_per_fold.csv")
    mean_std_rows, stats_rows = _aggregate_fold_rows(all_fold_summary_rows, test_set_order)
    _write_mean_std_summary_csv(mean_std_rows, run_dir / "final_8run_summary.csv")
    _write_stats_summary_csv(stats_rows, run_dir / "final_8run_summary_stats.csv")

    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "official_repo": "DonaldRR/SimpleNet",
                "device": device_mode,
                "seed": base_seed,
                "n_folds": len(manifest.get("cv_folds", [])),
                "test_set_order": list(test_set_order),
                "backbone_name": backbone_name,
                "layers_to_extract_from": list(layers_to_extract_from),
                "threshold_percentile": float(threshold_percentile),
                "summary_rows_per_fold": all_fold_summary_rows,
                "summary_rows_mean_std": mean_std_rows,
                "summary_rows_stats": stats_rows,
                "fold_runs": fold_run_summaries,
            },
            f,
            indent=2,
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the official SimpleNet implementation on the local protocol")
    parser.add_argument("--manifest", type=str, default="configs/manifests/cleaned_protocol_manifest.json")
    parser.add_argument("--dataset-root", type=str, default="RGIIIT_clean")
    parser.add_argument("--output-root", type=str, default="experiments/SimpleNet_Official")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resize", type=int, default=329)
    parser.add_argument("--imagesize", type=int, default=288)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backbone-name", type=str, default="wideresnet50")
    parser.add_argument("--layers-to-extract-from", nargs="*", default=["layer2", "layer3"])
    parser.add_argument("--pretrain-embed-dimension", type=int, default=1536)
    parser.add_argument("--target-embed-dimension", type=int, default=1536)
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--meta-epochs", type=int, default=40)
    parser.add_argument("--aed-meta-epochs", type=int, default=1)
    parser.add_argument("--gan-epochs", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--dsc-layers", type=int, default=2)
    parser.add_argument("--dsc-hidden", type=int, default=1024)
    parser.add_argument("--dsc-margin", type=float, default=0.5)
    parser.add_argument("--dsc-lr", type=float, default=0.0002)
    parser.add_argument("--auto-noise", type=float, default=0.0)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--pre-proj", type=int, default=1)
    parser.add_argument("--proj-layer-type", type=int, default=0)
    parser.add_argument("--mix-noise", type=int, default=1)
    parser.add_argument("--augment-train", action="store_true")
    parser.add_argument("--rotate-degrees", type=int, default=0)
    parser.add_argument("--translate", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--contrast", type=float, default=0.0)
    parser.add_argument("--saturation", type=float, default=0.0)
    parser.add_argument("--gray", type=float, default=0.0)
    parser.add_argument("--hflip", type=float, default=0.0)
    parser.add_argument("--vflip", type=float, default=0.0)
    parser.add_argument("--reuse-checkpoint-if-exists", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    return run_official_simplenet(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        output_root=output_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resize=args.resize,
        imagesize=args.imagesize,
        threshold_percentile=args.threshold_percentile,
        seed=args.seed,
        backbone_name=args.backbone_name,
        layers_to_extract_from=args.layers_to_extract_from,
        pretrain_embed_dimension=args.pretrain_embed_dimension,
        target_embed_dimension=args.target_embed_dimension,
        patchsize=args.patchsize,
        embedding_size=args.embedding_size,
        meta_epochs=args.meta_epochs,
        aed_meta_epochs=args.aed_meta_epochs,
        gan_epochs=args.gan_epochs,
        noise_std=args.noise_std,
        dsc_layers=args.dsc_layers,
        dsc_hidden=args.dsc_hidden,
        dsc_margin=args.dsc_margin,
        dsc_lr=args.dsc_lr,
        auto_noise=args.auto_noise,
        train_backbone=args.train_backbone,
        cos_lr=args.cos_lr,
        pre_proj=args.pre_proj,
        proj_layer_type=args.proj_layer_type,
        mix_noise=args.mix_noise,
        augment_train=args.augment_train,
        rotate_degrees=args.rotate_degrees,
        translate=args.translate,
        scale=args.scale,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        gray=args.gray,
        hflip=args.hflip,
        vflip=args.vflip,
        reuse_checkpoint_if_exists=args.reuse_checkpoint_if_exists,
    )


if __name__ == "__main__":
    raise SystemExit(main())
