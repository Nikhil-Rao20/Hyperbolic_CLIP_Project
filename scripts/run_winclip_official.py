from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_WINCLIP = PROJECT_ROOT / "third_party" / "WinCLIP"
WINCLIP_CHECKPOINT_URL = "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt"
WINCLIP_CHECKPOINT_NAME = "vit_b_16_plus_240-laion400m_e31-8fb26589.pt"
DEFAULT_CHECKPOINT = THIRD_PARTY_WINCLIP / WINCLIP_CHECKPOINT_NAME
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_THRESHOLD_PERCENTILE = 95.0
TEST_SET_ORDER = ["test_allfake", "test_gan", "test_ldm", "test_mls"]
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


def _to_numpy_labels_scores(gt_list: Sequence, score_list: Sequence) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([int(np.asarray(v).item()) for v in gt_list], dtype=np.int64)
    scores = np.array([float(np.asarray(v).item()) for v in score_list], dtype=np.float64)
    if labels.shape[0] != scores.shape[0]:
        raise RuntimeError("WinCLIP returned mismatched labels and scores lengths")
    if labels.shape[0] == 0:
        raise RuntimeError("WinCLIP returned empty labels/scores")
    return labels, scores


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
        raise RuntimeError("No threshold candidates available for WinCLIP metrics calibration")

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


def _build_fold_summary_row(fold_index: int, test_name: str, test_spec: dict, threshold_results: Dict[str, Dict]) -> Dict:
    default_metrics = threshold_results["default"]
    f1_metrics = threshold_results["f1"]
    j_metrics = threshold_results["youden_j"]
    return {
        "geometry": "winclip_official",
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


def _write_fold_summary_csv(rows: List[Dict], out_csv: Path) -> None:
    fieldnames = ["geometry", "fold_index", "test_set", "n_real", "n_fake", *SUMMARY_METRIC_KEYS]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_fold_rows(fold_rows: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in fold_rows:
        grouped[row["test_set"]].append(row)

    mean_std_rows: List[Dict] = []
    stats_rows: List[Dict] = []
    for test_name in TEST_SET_ORDER:
        rows = grouped.get(test_name, [])
        if not rows:
            continue

        mean_std_row = {
            "geometry": "winclip_official",
            "test_set": test_name,
            "n_folds": len(rows),
            "n_real": int(rows[0]["n_real"]),
            "n_fake": int(rows[0]["n_fake"]),
        }
        stats_row = {
            "geometry": "winclip_official",
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


def _resolve_source_image(dataset_root: Path, rel_path: str) -> Path:
    path = dataset_root / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing source image: {path}")
    return path


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_blank_mask_like(src_image: Path, dst_mask: Path) -> None:
    from PIL import Image

    dst_mask.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_image) as img:
        mask = Image.new("L", img.size, color=0)
        mask.save(dst_mask)


def _ensure_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        return checkpoint_path
    urllib.request.urlretrieve(WINCLIP_CHECKPOINT_URL, checkpoint_path.as_posix())
    return checkpoint_path


def _ensure_winclip_package_markers() -> None:
    datasets_init = THIRD_PARTY_WINCLIP / "datasets" / "__init__.py"
    if datasets_init.exists():
        return
    datasets_init.parent.mkdir(parents=True, exist_ok=True)
    datasets_init.write_text(
        "# Auto-created by run_winclip_official.py for deterministic local imports.\n",
        encoding="utf-8",
    )


def _clear_shadowed_modules(prefixes: Sequence[str]) -> None:
    for name in list(sys.modules.keys()):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(name, None)


def _validate_device_mode(device: str) -> str:
    mode = device.lower()
    if mode not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device mode: {device}. Use one of: auto, cpu, cuda")
    if mode == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this system")
    return mode


def _load_winclip_module():
    module_path = THIRD_PARTY_WINCLIP / "main.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"WinCLIP official source not found at {module_path}. "
            "Run: python scripts/setup_official_baselines.py"
        )

    _ensure_winclip_package_markers()

    winclip_path = THIRD_PARTY_WINCLIP.as_posix()
    if winclip_path not in sys.path:
        sys.path.insert(0, winclip_path)

    # Avoid collisions with globally installed packages (e.g. huggingface datasets).
    _clear_shadowed_modules(["datasets", "open_clip", "binary_focal_loss"])

    spec = importlib.util.spec_from_file_location("third_party_winclip_main", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import WinCLIP entry point from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_mvtec_like_dataset(
    dataset_root: Path,
    rel_paths: Sequence[str],
    object_name: str,
    temp_root: Path,
    create_train: bool = False,
    train_rel_paths: Sequence[str] | None = None,
) -> Path:
    class_root = temp_root / object_name
    train_good = class_root / "train" / "good"
    test_good = class_root / "test" / "good"
    test_defect = class_root / "test" / "defect"
    gt_defect = class_root / "ground_truth" / "defect"

    for path in [train_good, test_good, test_defect, gt_defect]:
        path.mkdir(parents=True, exist_ok=True)

    for rel_path in rel_paths:
        src = _resolve_source_image(dataset_root, rel_path)
        label = 0 if "/Real/" in rel_path.replace("\\", "/") else 1
        if label == 0:
            dst = test_good / Path(rel_path).name
            _copy_image(src, dst)
        else:
            dst = test_defect / Path(rel_path).name
            _copy_image(src, dst)
            _write_blank_mask_like(src, gt_defect / f"{Path(rel_path).stem}_mask.png")

    if create_train:
        # For few-shot WinCLIP, support images should come from protocol real-train pool.
        support_paths = list(train_rel_paths) if train_rel_paths is not None else list(rel_paths)
        for rel_path in support_paths:
            if "/Real/" not in rel_path.replace("\\", "/"):
                continue
            src = _resolve_source_image(dataset_root, rel_path)
            _copy_image(src, train_good / Path(rel_path).name)

    return temp_root


def run_official_winclip(
    manifest_path: Path,
    dataset_root: Path,
    output_root: Path,
    object_name: str = "candle",
    shot: int = 0,
    threshold_percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
    num_workers: int = 0,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    device: str = "auto",
) -> int:
    if shot < 0:
        raise ValueError("shot must be >= 0")
    if not (0.0 < float(threshold_percentile) < 100.0):
        raise ValueError("threshold_percentile must be between 0 and 100")
    device = _validate_device_mode(device)
    manifest = _load_manifest(manifest_path)

    module = _load_winclip_module()
    checkpoint = _ensure_checkpoint(checkpoint_path)

    run_dir = output_root / "winclip_official"
    run_dir.mkdir(parents=True, exist_ok=True)
    fold_summary_rows: List[Dict] = []

    for fold in manifest["cv_folds"]:
        fold_index = int(fold["fold_index"])
        fold_dir = run_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        support_real_ids = list(fold.get("train_ids", []))
        if shot > 0 and len(support_real_ids) == 0:
            raise ValueError(f"Few-shot mode requires fold train_ids, but fold {fold_index} has none")

        # Calibrate fold-level thresholds once, then reuse for all test sets in the fold.
        # default -> val_ids real scores percentile; f1/youden_j -> val_eval_ids.
        val_real_rel_paths = sorted(list(fold["val_ids"]))
        val_eval_rel_paths = sorted(list(fold["val_eval_ids"]))

        with tempfile.TemporaryDirectory(prefix=f"winclip_{fold_index}_val_real_") as tmp_dir:
            temp_root = Path(tmp_dir) / "dataset"
            _build_mvtec_like_dataset(
                dataset_root,
                val_real_rel_paths,
                object_name,
                temp_root,
                create_train=(shot > 0),
                train_rel_paths=support_real_ids if shot > 0 else None,
            )

            val_real_config = {
                "datasetname": temp_root.name,
                "dataset_root_dir": temp_root.parent.as_posix(),
                "data_dir": temp_root.as_posix(),
                "obj_type_id": 0,
                "obj_type": object_name,
                "shot": int(shot),
                "num_workers": int(num_workers),
                "device": device,
            }

            cwd_before = Path.cwd()
            try:
                os.chdir(THIRD_PARTY_WINCLIP)
                val_real_gt_list, val_real_score_list, _, _, _ = module.run(val_real_config)
            finally:
                os.chdir(cwd_before)

        with tempfile.TemporaryDirectory(prefix=f"winclip_{fold_index}_val_eval_") as tmp_dir:
            temp_root = Path(tmp_dir) / "dataset"
            _build_mvtec_like_dataset(
                dataset_root,
                val_eval_rel_paths,
                object_name,
                temp_root,
                create_train=(shot > 0),
                train_rel_paths=support_real_ids if shot > 0 else None,
            )

            val_config = {
                "datasetname": temp_root.name,
                "dataset_root_dir": temp_root.parent.as_posix(),
                "data_dir": temp_root.as_posix(),
                "obj_type_id": 0,
                "obj_type": object_name,
                "shot": int(shot),
                "num_workers": int(num_workers),
                "device": device,
            }

            cwd_before = Path.cwd()
            try:
                os.chdir(THIRD_PARTY_WINCLIP)
                val_gt_list, val_score_list, _, _, _ = module.run(val_config)
            finally:
                os.chdir(cwd_before)

        val_real_labels, val_real_scores = _to_numpy_labels_scores(val_real_gt_list, val_real_score_list)
        val_labels, val_scores = _to_numpy_labels_scores(val_gt_list, val_score_list)
        val_real_scores = val_real_scores[val_real_labels == 0]
        if val_real_scores.size == 0:
            raise RuntimeError(f"Fold {fold_index} has no real samples in val_ids for default threshold calibration")

        default_threshold = float(np.percentile(val_real_scores, float(threshold_percentile)))
        thresholds = _calibrate_thresholds(val_labels, val_scores, default_threshold=default_threshold)

        val_threshold_results: Dict[str, Dict] = {}
        val_threshold_rows: List[Dict] = []
        for threshold_name in ["default", "f1", "youden_j"]:
            threshold = float(thresholds[threshold_name])
            metrics = _compute_metrics(val_labels, val_scores, threshold)
            val_threshold_results[threshold_name] = metrics
            val_threshold_rows.append(
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
        for threshold_name in ["default", "f1", "youden_j"]:
            with (fold_dir / f"val_classification_report_{threshold_name}.json").open("w", encoding="utf-8") as f:
                json.dump(val_threshold_results[threshold_name]["classification_report"], f, indent=2)

        for test_name in TEST_SET_ORDER:
            test_spec = manifest["test_sets"][test_name]
            rel_paths = sorted(test_spec["real_ids"] + test_spec["fake_ids"])

            with tempfile.TemporaryDirectory(prefix=f"winclip_{fold_index}_{test_name}_") as tmp_dir:
                temp_root = Path(tmp_dir) / "dataset"
                _build_mvtec_like_dataset(
                    dataset_root,
                    rel_paths,
                    object_name,
                    temp_root,
                    create_train=(shot > 0),
                    train_rel_paths=support_real_ids if shot > 0 else None,
                )

                config = {
                    "datasetname": temp_root.name,
                    "dataset_root_dir": temp_root.parent.as_posix(),
                    "data_dir": temp_root.as_posix(),
                    "obj_type_id": 0,
                    "obj_type": object_name,
                    "shot": int(shot),
                    "num_workers": int(num_workers),
                    "device": device,
                }

                cwd_before = Path.cwd()
                try:
                    os.chdir(THIRD_PARTY_WINCLIP)
                    gt_list, score_list, auroc, aupr, f1_max = module.run(config)
                finally:
                    os.chdir(cwd_before)

            labels, scores = _to_numpy_labels_scores(gt_list, score_list)
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
            with (test_dir / "results.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fold_index": fold_index,
                        "test_set": test_name,
                        "n_real": test_spec["n_real"],
                        "n_fake": test_spec["n_fake"],
                        "auroc": float(default_metrics["auroc"]),
                        "auprc": float(default_metrics["auprc"]),
                        "f1_max": float(default_metrics["f1"]),
                        "legacy_module_auroc": float(auroc),
                        "legacy_module_auprc": float(aupr),
                        "legacy_module_f1_max": float(f1_max),
                        "thresholds": thresholds,
                        "threshold_results": threshold_results,
                        "official_repo": "mala-lab/WinCLIP",
                        "checkpoint": checkpoint.as_posix(),
                        "object_name": object_name,
                        "shot": int(shot),
                        "threshold_percentile": float(threshold_percentile),
                        "calibration_split": {
                            "default": "fold.val_ids",
                            "f1": "fold.val_eval_ids",
                            "youden_j": "fold.val_eval_ids",
                        },
                        "device": device,
                    },
                    f,
                    indent=2,
                )

            fold_summary_rows.append(_build_fold_summary_row(fold_index, test_name, test_spec, threshold_results))

            print(
                f"[WinCLIP] fold={fold_index} set={test_name} "
                f"auroc={default_metrics['auroc']:.4f} auprc={default_metrics['auprc']:.4f} "
                f"f1@default={default_metrics['f1']:.4f}",
                flush=True,
            )

    _write_fold_summary_csv(fold_summary_rows, run_dir / "final_8run_summary_per_fold.csv")
    mean_std_rows, stats_rows = _aggregate_fold_rows(fold_summary_rows)
    _write_mean_std_summary_csv(mean_std_rows, run_dir / "final_8run_summary.csv")
    _write_stats_summary_csv(stats_rows, run_dir / "final_8run_summary_stats.csv")

    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "official_repo": "mala-lab/WinCLIP",
                "object_name": object_name,
                "shot": int(shot),
                "threshold_percentile": float(threshold_percentile),
                "calibration_split": {
                    "default": "fold.val_ids",
                    "f1": "fold.val_eval_ids",
                    "youden_j": "fold.val_eval_ids",
                },
                "n_folds": len(manifest.get("cv_folds", [])),
                "summary_rows_per_fold": fold_summary_rows,
                "summary_rows_mean_std": mean_std_rows,
                "summary_rows_stats": stats_rows,
            },
            f,
            indent=2,
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the official WinCLIP implementation on the local manifest")
    parser.add_argument("--manifest", type=str, default="configs/manifests/cleaned_protocol_manifest.json")
    parser.add_argument("--dataset-root", type=str, default="RGIIIT_clean")
    parser.add_argument("--output-root", type=str, default="experiments/WinCLIP_Official")
    parser.add_argument("--object-name", type=str, default="candle")
    parser.add_argument("--shot", type=int, default=0)
    parser.add_argument("--threshold-percentile", type=float, default=DEFAULT_THRESHOLD_PERCENTILE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
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
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    output_root.mkdir(parents=True, exist_ok=True)

    return run_official_winclip(
        manifest_path,
        dataset_root,
        output_root,
        object_name=args.object_name,
        shot=args.shot,
        threshold_percentile=args.threshold_percentile,
        num_workers=args.num_workers,
        checkpoint_path=checkpoint_path,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())