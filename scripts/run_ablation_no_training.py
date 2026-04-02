from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


try:
    from run_one_class_svdd_clip_v2 import (
        HyperbolicProjectionHead,
        compute_center_hyperbolic,
        compute_anomaly_scores,
        _encode_image_features,
        _load_image_backbone,
        ImagePathDataset,
        collate_fn,
        compute_metrics,
        calibrate_threshold,
    )
except Exception:
    # Fallback import path if sibling import fails in some environments.
    import importlib.util

    module_path = PROJECT_ROOT / "scripts" / "run_one_class_svdd_clip_v2.py"
    spec = importlib.util.spec_from_file_location("run_one_class_svdd_clip_v2", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    HyperbolicProjectionHead = module.HyperbolicProjectionHead
    compute_center_hyperbolic = module.compute_center_hyperbolic
    compute_anomaly_scores = module.compute_anomaly_scores
    _encode_image_features = module._encode_image_features
    _load_image_backbone = module._load_image_backbone
    ImagePathDataset = module.ImagePathDataset
    collate_fn = module.collate_fn
    compute_metrics = module.compute_metrics
    calibrate_threshold = module.calibrate_threshold


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _aggregate_fold_results(all_fold_results: List[Dict]) -> Dict[str, Dict[str, float]]:
    test_set_names = ["test_allfake", "test_gan", "test_ldm", "test_mls"]
    metric_keys = ["auroc", "auprc", "accuracy", "precision", "recall", "f1", "specificity", "sensitivity"]

    aggregated = {}
    for test_name in test_set_names:
        aggregated[test_name] = {}
        for metric in metric_keys:
            values = [fold_result["test_results"][test_name][metric] for fold_result in all_fold_results]
            aggregated[test_name][f"{metric}_mean"] = float(np.mean(values))
            aggregated[test_name][f"{metric}_std"] = float(np.std(values))
    return aggregated


def _write_svdd_style_summary_csv(rows: List[Dict], out_csv: Path) -> None:
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
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_fold(cfg: dict, manifest: dict, fold: dict, dataset_root: Path, device: torch.device, fold_index: int):
    batch_size = int(cfg.get("batch_size", 32))
    curvature = float(cfg.get("curvature", 1.0))
    scale = float(cfg.get("scale", 0.1))
    projection_dim = int(cfg.get("projection_dim", 256))
    threshold_percentile = float(cfg.get("threshold_percentile", 95))

    # STEP A — Load model with NO gradients
    clip_model, processor = _load_image_backbone(
        cfg["clip_model_name"],
        "clip",
        "openai",
        device,
        num_vit_layers=12,
    )
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model.eval()

    projection_head = HyperbolicProjectionHead(
        input_dim=512,
        projection_dim=projection_dim,
        curvature=curvature,
        scale=scale,
    ).to(device)
    for p in projection_head.parameters():
        p.requires_grad = False
    projection_head.eval()

    # STEP B — Build datasets
    train_ds = ImagePathDataset(dataset_root, fold["train_ids"])
    val_real_ds = ImagePathDataset(dataset_root, fold["val_ids"])
    val_eval_ds = ImagePathDataset(dataset_root, fold["val_eval_ids"])

    print(f"[INFO] Fold {fold_index} | Computing hyperbolic center from random projections...", flush=True)

    # STEP C — Compute center from random projections
    with torch.no_grad():
        center = compute_center_hyperbolic(
            clip_model,
            processor,
            projection_head,
            train_ds,
            batch_size,
            device,
            backbone_type="clip",
        )

    # Threshold from val real scores (95th percentile), with optional calibration fallback
    val_real_labels, val_real_scores, _, _ = compute_anomaly_scores(
        clip_model,
        processor,
        projection_head,
        center,
        val_real_ds,
        batch_size,
        device,
        geometry="hyperbolic",
        backbone_type="clip",
    )
    _ = val_real_labels
    default_threshold = float(np.percentile(val_real_scores, threshold_percentile))

    val_eval_labels, val_eval_scores, _, _ = compute_anomaly_scores(
        clip_model,
        processor,
        projection_head,
        center,
        val_eval_ds,
        batch_size,
        device,
        geometry="hyperbolic",
        backbone_type="clip",
    )

    # Keep threshold strategy aligned with SVDD utility (calibrated available, default simpler baseline).
    best_f1, best_j = calibrate_threshold(val_eval_labels, val_eval_scores)
    threshold = default_threshold

    val_metrics = compute_metrics(val_eval_labels, val_eval_scores, threshold)
    val_auroc = float(val_metrics["auroc"])

    print(f"[INFO] Fold {fold_index} | Center computed. Evaluating on test sets...", flush=True)

    # STEP D — Evaluate on all 4 test sets
    test_set_names = ["test_allfake", "test_gan", "test_ldm", "test_mls"]
    fold_test_results = {}

    for test_name in test_set_names:
        test_spec = manifest["test_sets"][test_name]
        rel_paths = sorted(test_spec["real_ids"] + test_spec["fake_ids"])
        test_ds = ImagePathDataset(dataset_root, rel_paths)

        labels, scores, _, _ = compute_anomaly_scores(
            clip_model,
            processor,
            projection_head,
            center,
            test_ds,
            batch_size,
            device,
            geometry="hyperbolic",
            backbone_type="clip",
        )

        metrics = compute_metrics(labels, scores, threshold)
        fold_test_results[test_name] = {
            "auroc": float(metrics["auroc"]),
            "auprc": float(metrics["auprc"]),
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "specificity": float(metrics["specificity"]),
            "sensitivity": float(metrics["sensitivity"]),
        }

    print(f"[INFO] Fold {fold_index} | Test Results:", flush=True)
    for test_name, metrics in fold_test_results.items():
        print(
            f"       {test_name:<15} → AUROC: {metrics['auroc']:.4f} | "
            f"ACC: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}",
            flush=True,
        )

    return {
        "fold_index": fold_index,
        "val_auroc": val_auroc,
        "threshold_default": default_threshold,
        "threshold_calibrated_f1": float(best_f1["threshold"]),
        "threshold_calibrated_youden_j": float(best_j["threshold"]),
        "test_results": fold_test_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablation: CLIP + Projection without training")
    parser.add_argument("--config", type=str, default="configs/ablation_no_training.yaml")
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    cfg = _load_config(cfg_path)

    manifest_path = PROJECT_ROOT / cfg["protocol_manifest_path"]
    manifest = _load_manifest(manifest_path)

    set_seed(int(cfg.get("seed", 42)))

    dataset_root = PROJECT_ROOT / cfg["dataset_root"]
    output_dir = PROJECT_ROOT / cfg["output_root"]
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading manifest: {manifest_path}", flush=True)
    print("[INFO] Experiment: CLIP + Random Projection (No Training)", flush=True)
    print("[INFO] Frozen CLIP ViT B16 + randomly initialized HyperbolicProjectionHead", flush=True)
    print("[INFO] No training — computing center directly from random projections", flush=True)
    print(f"[INFO] Device: {device}", flush=True)

    all_fold_results = []

    for fold in manifest["cv_folds"]:
        fold_index = int(fold["fold_index"])
        print(f"\n[INFO] ── Fold {fold_index}/5 " + "─" * 42, flush=True)
        fold_result = run_fold(cfg, manifest, fold, dataset_root, device, fold_index)
        all_fold_results.append(fold_result)

    best_fold = max(all_fold_results, key=lambda item: item["val_auroc"])
    best_fold_results = best_fold["test_results"]

    print("\n[INFO] ── All Folds Complete " + "─" * 33, flush=True)
    allfake = best_fold_results["test_allfake"]
    print(
        "[INFO] Best-Fold Results (test_allfake):\n"
        f"       AUROC: {allfake['auroc']:.4f} | "
        f"ACC: {allfake['accuracy']:.4f} | "
        f"F1: {allfake['f1']:.4f}",
        flush=True,
    )

    summary_rows = []
    for test_name in ["test_gan", "test_ldm", "test_mls", "test_allfake"]:
        metrics = best_fold_results[test_name]
        test_spec = manifest["test_sets"][test_name]
        summary_rows.append(
            {
                "geometry": "hyperbolic",
                "test_set": test_name,
                "n_real": test_spec["n_real"],
                "n_fake": test_spec["n_fake"],
                "auroc": round(float(metrics["auroc"]), 6),
                "auprc": round(float(metrics["auprc"]), 6),
                "accuracy_default": round(float(metrics["accuracy"]), 6),
                "f1_default": round(float(metrics["f1"]), 6),
                "sensitivity_default": round(float(metrics["sensitivity"]), 6),
                "specificity_default": round(float(metrics["specificity"]), 6),
            }
        )

    _write_svdd_style_summary_csv(summary_rows, output_dir / "final_8run_summary.csv")

    run_summary = {
        "geometries": [
            {
                "geometry": "hyperbolic",
                "best_fold": {
                    "fold_index": int(best_fold["fold_index"]),
                    "best_val_auroc": float(best_fold["val_auroc"]),
                },
                "summary_rows": summary_rows,
                "elapsed_sec": None,
            }
        ]
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    results_summary = {
        "experiment": "ablation_clip_random_projection",
        "description": "Frozen CLIP + randomly initialized untrained HyperbolicProjectionHead. No gradient updates. Tests whether training is essential.",
        "reporting_style": "svdd_like_best_fold",
        "config": {
            "clip_model_name": cfg["clip_model_name"],
            "batch_size": int(cfg.get("batch_size", 32)),
            "projection_dim": int(cfg.get("projection_dim", 256)),
            "curvature": float(cfg.get("curvature", 1.0)),
            "scale": float(cfg.get("scale", 0.1)),
            "seed": int(cfg.get("seed", 42)),
            "n_folds": int(cfg.get("n_folds", 5)),
            "protocol_manifest_path": cfg["protocol_manifest_path"],
        },
        "best_fold": {
            "fold_index": int(best_fold["fold_index"]),
            "val_auroc": float(best_fold["val_auroc"]),
        },
        "test_results": best_fold_results,
    }

    with (output_dir / "results_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    with (output_dir / "fold_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_fold_results, f, indent=2)

    # training_log.csv is intentionally a note-only artifact for this no-training ablation.
    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["note"])
        writer.writeheader()
        writer.writerow({"note": "No training loop executed in this ablation."})

    print(f"[INFO] Results saved to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
