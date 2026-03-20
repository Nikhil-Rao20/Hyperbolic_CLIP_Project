from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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
)
from torch.utils.data import Dataset

import geoopt
from transformers import (
    CLIPModel,
    CLIPProcessor,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


class FrozenBackboneExtractor:
    def __init__(self, model_id: str, model_type: str, device: torch.device, instruct_prompt: str):
        self.model_id = model_id
        self.model_type = model_type
        self.device = device
        self.instruct_prompt = instruct_prompt

        if model_type == "clip":
            self.model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_id)
        elif model_type == "instructblip":
            self.model = InstructBlipForConditionalGeneration.from_pretrained(model_id).to(device)
            self.processor = InstructBlipProcessor.from_pretrained(model_id)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        if self.model_type == "clip":
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self.model.get_image_features(**inputs)
            feats = F.normalize(feats.float(), dim=-1)
            return feats

        # InstructBLIP: use vision encoder pooled output as frozen image embedding.
        texts = [self.instruct_prompt] * len(images)
        inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(self.device)
        vision_out = self.model.vision_model(pixel_values=pixel_values)

        if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
            feats = vision_out.pooler_output
        else:
            feats = vision_out.last_hidden_state.mean(dim=1)

        feats = F.normalize(feats.float(), dim=-1)
        return feats


@torch.no_grad()
def compute_embeddings(extractor: FrozenBackboneExtractor, dataset: ImagePathDataset, batch_size: int) -> Tuple[torch.Tensor, np.ndarray, List[str], List[str]]:
    all_embs = []
    labels = []
    sources = []
    rels = []

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = [dataset[i] for i in range(start, end)]
        images = [x[0] for x in batch]
        lbs = [x[1] for x in batch]
        srcs = [x[2] for x in batch]
        ids = [x[3] for x in batch]

        embs = extractor.encode_images(images)
        all_embs.append(embs.cpu())
        labels.extend(lbs)
        sources.extend(srcs)
        rels.extend(ids)

    return torch.cat(all_embs, dim=0), np.array(labels), sources, rels


def predict_from_threshold(scores: np.ndarray, threshold: float):
    return (scores > threshold).astype(int)


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict:
    preds = predict_from_threshold(scores, threshold)
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

    cm = confusion_matrix(labels, preds)
    out["confusion_matrix"] = cm.tolist()
    tn, fp = cm[0]
    fn, tp = cm[1]
    out["specificity"] = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
    out["sensitivity"] = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    out["balanced_mean_accuracy"] = round((out["specificity"] + out["sensitivity"]) / 2.0, 4)
    return out


def calibrate_threshold(labels: np.ndarray, scores: np.ndarray):
    uniq = np.unique(scores)
    candidates = uniq if len(uniq) < 10 else np.quantile(scores, np.linspace(0.01, 0.99, 200))

    best_f1 = {"threshold": float(candidates[0]), "f1": -1.0}
    best_j = {"threshold": float(candidates[0]), "youden_j": -2.0}

    for th in candidates:
        preds = predict_from_threshold(scores, float(th))
        f1 = float(f1_score(labels, preds, zero_division=0))
        rec = float(recall_score(labels, preds, zero_division=0))
        spe = float(recall_score(labels, preds, pos_label=0, zero_division=0))
        j = rec + spe - 1.0
        if f1 > best_f1["f1"]:
            best_f1 = {"threshold": float(th), "f1": f1}
        if j > best_j["youden_j"]:
            best_j = {"threshold": float(th), "youden_j": j}

    return best_f1, best_j


@torch.no_grad()
def frechet_mean_iterative(points: torch.Tensor, curvature: float, max_iter: int = 100, tol: float = 1e-7):
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


def evaluate_fold_geometry(
    fold: Dict,
    extractor: FrozenBackboneExtractor,
    dataset_root: Path,
    batch_size: int,
    geometry: str,
    threshold_percentile: float,
    curvature: float,
    scale: float,
):
    train_ds = ImagePathDataset(dataset_root, fold["train_ids"])
    val_real_ds = ImagePathDataset(dataset_root, fold["val_ids"])
    val_eval_ds = ImagePathDataset(dataset_root, fold["val_eval_ids"])

    train_emb, _, _, _ = compute_embeddings(extractor, train_ds, batch_size)
    val_real_emb, val_real_labels, _, _ = compute_embeddings(extractor, val_real_ds, batch_size)
    val_eval_emb, val_eval_labels, _, _ = compute_embeddings(extractor, val_eval_ds, batch_size)

    if geometry == "euclidean":
        center = train_emb.mean(dim=0)
        val_in_scores = torch.norm(val_real_emb - center.unsqueeze(0), dim=-1).numpy()
        val_eval_scores = torch.norm(val_eval_emb - center.unsqueeze(0), dim=-1).numpy()
    else:
        ball = geoopt.PoincareBall(c=curvature)
        train_h = ball.expmap0(F.normalize(train_emb, dim=-1) * scale)
        val_real_h = ball.expmap0(F.normalize(val_real_emb, dim=-1) * scale)
        val_eval_h = ball.expmap0(F.normalize(val_eval_emb, dim=-1) * scale)
        center = frechet_mean_iterative(train_h, curvature=curvature)
        val_in_scores = ball.dist(val_real_h, center.unsqueeze(0)).cpu().numpy()
        val_eval_scores = ball.dist(val_eval_h, center.unsqueeze(0)).cpu().numpy()

    default_threshold = float(np.percentile(val_in_scores, threshold_percentile))
    best_f1, best_j = calibrate_threshold(val_eval_labels, val_eval_scores)

    val_metrics = {
        "default": compute_metrics(val_eval_labels, val_eval_scores, default_threshold),
        "f1": compute_metrics(val_eval_labels, val_eval_scores, best_f1["threshold"]),
        "youden_j": compute_metrics(val_eval_labels, val_eval_scores, best_j["threshold"]),
    }

    return {
        "fold_index": fold["fold_index"],
        "center": center,
        "val_metrics": val_metrics,
        "thresholds": {
            "default": default_threshold,
            "f1": float(best_f1["threshold"]),
            "youden_j": float(best_j["threshold"]),
        },
    }


def _save_confusion_matrix(labels: np.ndarray, preds: np.ndarray, out_path: Path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_test_set(
    extractor: FrozenBackboneExtractor,
    dataset_root: Path,
    test_spec: Dict,
    batch_size: int,
    geometry: str,
    center: torch.Tensor,
    thresholds: Dict[str, float],
    curvature: float,
    scale: float,
    out_dir: Path,
):
    rel_paths = sorted(test_spec["real_ids"] + test_spec["fake_ids"])
    ds = ImagePathDataset(dataset_root, rel_paths)
    embs, labels, _, _ = compute_embeddings(extractor, ds, batch_size)

    if geometry == "euclidean":
        scores = torch.norm(embs - center.unsqueeze(0), dim=-1).numpy()
    else:
        ball = geoopt.PoincareBall(c=curvature)
        h = ball.expmap0(F.normalize(embs, dim=-1) * scale)
        scores = ball.dist(h, center.unsqueeze(0)).cpu().numpy()

    threshold_results = {}
    for name, th in thresholds.items():
        m = compute_metrics(labels, scores, th)
        preds = predict_from_threshold(scores, th)
        _save_confusion_matrix(labels, preds, out_dir / f"confusion_matrix_{name}.png")
        threshold_results[name] = m

    return {
        "n_real": test_spec["n_real"],
        "n_fake": test_spec["n_fake"],
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "threshold_results": threshold_results,
        "thresholds": thresholds,
    }


def write_summary_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "geometry",
        "test_set",
        "n_real",
        "n_fake",
        "auroc",
        "auprc",
        "accuracy_default",
        "f1_default",
        "precision_default",
        "sensitivity_default",
        "specificity_default",
        "balanced_mean_accuracy_default",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen backbone protocol evaluation (no fine-tuning)")
    parser.add_argument("--config", type=str, default="configs/frozen_backbone_protocol.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = _load_config(cfg_path)

    dataset_root = Path(cfg["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root

    manifest_path = Path(cfg["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    output_root = Path(cfg.get("output_root", "experiments_frozen_protocol"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    run_name = cfg.get("run_name", f"frozen_protocol_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg.get("batch_size", 16))
    geometries = cfg.get("geometries", ["euclidean", "hyperbolic"])
    threshold_percentile = float(cfg.get("threshold_percentile", 95))
    curvature = float(cfg.get("curvature", 1.0))
    scale = float(cfg.get("scale", 0.1))
    instruct_prompt = cfg.get("instruct_prompt", "Describe whether this MRI slice is real or synthetic.")

    all_rows: List[Dict] = []

    for model_cfg in cfg.get("models", []):
        model_name = model_cfg["name"]
        model_id = model_cfg["model_id"]
        model_type = model_cfg.get("type", "clip")
        print(f"\n[MODEL] {model_name} ({model_id})", flush=True)

        model_dir = run_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        extractor = FrozenBackboneExtractor(
            model_id=model_id,
            model_type=model_type,
            device=device,
            instruct_prompt=instruct_prompt,
        )

        model_summary = {}
        for geometry in geometries:
            print(f"  [GEOMETRY] {geometry}", flush=True)
            geometry_dir = model_dir / geometry
            geometry_dir.mkdir(parents=True, exist_ok=True)

            fold_runs = []
            for fold in manifest["cv_folds"]:
                fold_res = evaluate_fold_geometry(
                    fold=fold,
                    extractor=extractor,
                    dataset_root=dataset_root,
                    batch_size=batch_size,
                    geometry=geometry,
                    threshold_percentile=threshold_percentile,
                    curvature=curvature,
                    scale=scale,
                )
                fold_runs.append(fold_res)

            best_fold = max(fold_runs, key=lambda x: x["val_metrics"]["f1"]["auroc"])
            best_fold_index = best_fold["fold_index"]

            with (geometry_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_fold_index": best_fold_index,
                        "folds": [
                            {
                                "fold_index": fr["fold_index"],
                                "val_auroc_f1": fr["val_metrics"]["f1"]["auroc"],
                                "thresholds": fr["thresholds"],
                            }
                            for fr in fold_runs
                        ],
                    },
                    f,
                    indent=2,
                )

            test_payload = {}
            for test_name, test_spec in manifest["test_sets"].items():
                test_dir = geometry_dir / test_name
                test_dir.mkdir(parents=True, exist_ok=True)
                result = evaluate_test_set(
                    extractor=extractor,
                    dataset_root=dataset_root,
                    test_spec=test_spec,
                    batch_size=batch_size,
                    geometry=geometry,
                    center=best_fold["center"],
                    thresholds=best_fold["thresholds"],
                    curvature=curvature,
                    scale=scale,
                    out_dir=test_dir,
                )
                test_payload[test_name] = result

                d = result["threshold_results"]["default"]
                all_rows.append(
                    {
                        "model_name": model_name,
                        "geometry": geometry,
                        "test_set": test_name,
                        "n_real": result["n_real"],
                        "n_fake": result["n_fake"],
                        "auroc": round(result["auroc"], 6),
                        "auprc": round(result["auprc"], 6),
                        "accuracy_default": round(d["accuracy"], 6),
                        "f1_default": round(d["f1"], 6),
                        "precision_default": round(d["precision"], 6),
                        "sensitivity_default": round(d["sensitivity"], 6),
                        "specificity_default": round(d["specificity"], 6),
                        "balanced_mean_accuracy_default": round(d["balanced_mean_accuracy"], 6),
                    }
                )

            with (geometry_dir / "test_summary.json").open("w", encoding="utf-8") as f:
                json.dump({"best_fold": best_fold_index, "tests": test_payload}, f, indent=2)

            model_summary[geometry] = {"best_fold_index": best_fold_index}

        with (model_dir / "model_summary.json").open("w", encoding="utf-8") as f:
            json.dump(model_summary, f, indent=2)

    write_summary_csv(all_rows, run_dir / "final_frozen_protocol_summary.csv")
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"run_dir": run_dir.as_posix(), "n_rows": len(all_rows)}, f, indent=2)

    print("\n[INFO] Completed frozen protocol evaluation.")
    print("[INFO] Summary:", run_dir / "final_frozen_protocol_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
