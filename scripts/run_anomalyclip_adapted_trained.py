from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from .run_anomalyclip_official import (
        _build_official_eval_dataset,
        _extract_image_metrics_from_log,
        _run_subprocess,
        _subprocess_env_for_device,
        _validate_device_mode,
    )
except ImportError:
    from run_anomalyclip_official import (
        _build_official_eval_dataset,
        _extract_image_metrics_from_log,
        _run_subprocess,
        _subprocess_env_for_device,
        _validate_device_mode,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_ANOMALYCLIP = PROJECT_ROOT / "third_party" / "AnomalyCLIP"


class ImageLabelDataset(Dataset):
    def __init__(self, dataset_root: Path, rel_paths: Sequence[str], labels: Sequence[int], transform):
        self.dataset_root = dataset_root
        self.rel_paths = list(rel_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, idx: int):
        rel_path = self.rel_paths[idx]
        image_path = self.dataset_root / rel_path
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for training/eval: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)
        label = int(self.labels[idx])
        return image_tensor, label, rel_path


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _is_real_path(rel_path: str) -> bool:
    return "/real/" in rel_path.replace("\\", "/").lower()


def _labels_from_rel_paths(rel_paths: Sequence[str]) -> np.ndarray:
    # Real=0, Fake=1
    return np.array([0 if _is_real_path(p) else 1 for p in rel_paths], dtype=np.int64)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_torch_device(mode: str) -> torch.device:
    mode = _validate_device_mode(mode)
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(mode)


def _import_anomalyclip_modules():
    if not THIRD_PARTY_ANOMALYCLIP.exists():
        raise FileNotFoundError(
            f"AnomalyCLIP official source not found at {THIRD_PARTY_ANOMALYCLIP}. "
            "Run: python scripts/setup_official_baselines.py"
        )

    added = False
    if THIRD_PARTY_ANOMALYCLIP.as_posix() not in sys.path:
        sys.path.insert(0, THIRD_PARTY_ANOMALYCLIP.as_posix())
        added = True

    try:
        import AnomalyCLIP_lib  # type: ignore
        from prompt_ensemble import AnomalyCLIP_PromptLearner  # type: ignore
        from utils import get_transform  # type: ignore
    finally:
        if added:
            try:
                sys.path.remove(THIRD_PARTY_ANOMALYCLIP.as_posix())
            except ValueError:
                pass

    return AnomalyCLIP_lib, AnomalyCLIP_PromptLearner, get_transform


def _compute_text_features(model, prompt_learner):
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def _compute_logits(model, prompt_learner, images: torch.Tensor, features_list: Sequence[int]):
    with torch.no_grad():
        image_features, _ = model.encode_image(images, features_list, DPAM_layer=20)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    text_features = _compute_text_features(model, prompt_learner)
    logits = (image_features.unsqueeze(1) @ text_features.permute(0, 2, 1))[:, 0, :] / 0.07
    return logits


def _evaluate_image_level(
    model,
    prompt_learner,
    data_loader: DataLoader,
    device: torch.device,
    features_list: Sequence[int],
) -> Tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    prompt_learner.eval()
    all_labels: List[int] = []
    all_scores: List[float] = []

    with torch.no_grad():
        text_features = _compute_text_features(model, prompt_learner)
        for images, labels, _ in data_loader:
            images = images.to(device)
            with torch.no_grad():
                image_features, _ = model.encode_image(images, features_list, DPAM_layer=20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = (image_features.unsqueeze(1) @ text_features.permute(0, 2, 1))[:, 0, :] / 0.07
            probs_fake = torch.softmax(logits, dim=-1)[:, 1]

            all_labels.extend(labels.numpy().astype(int).tolist())
            all_scores.extend(probs_fake.detach().cpu().numpy().astype(float).tolist())

    labels_np = np.array(all_labels, dtype=np.int64)
    scores_np = np.array(all_scores, dtype=np.float32)

    if len(np.unique(labels_np)) < 2:
        auroc = 0.0
        auprc = 0.0
    else:
        auroc = float(roc_auc_score(labels_np, scores_np))
        auprc = float(average_precision_score(labels_np, scores_np))

    return auroc, auprc


def _train_fold(
    fold: dict,
    manifest: dict,
    dataset_root: Path,
    run_dir: Path,
    cfg: dict,
    device: torch.device,
) -> dict:
    fold_index = int(fold["fold_index"])
    fold_dir = run_dir / f"fold_{fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_ids = sorted(fold["train_ids"] + fold.get("calibration_fake_ids", []))
    val_eval_ids = sorted(fold["val_eval_ids"])

    train_labels = _labels_from_rel_paths(train_ids)
    val_labels = _labels_from_rel_paths(val_eval_ids)

    AnomalyCLIP_lib, AnomalyCLIP_PromptLearner, get_transform = _import_anomalyclip_modules()

    args_ns = SimpleNamespace(image_size=int(cfg.get("image_size", 518)))
    preprocess, _ = get_transform(args_ns)

    train_ds = ImageLabelDataset(dataset_root, train_ids, train_labels, preprocess)
    val_ds = ImageLabelDataset(dataset_root, val_eval_ids, val_labels, preprocess)

    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    params = {
        "Prompt_length": int(cfg.get("n_ctx", 12)),
        "learnabel_text_embedding_depth": int(cfg.get("depth", 9)),
        "learnabel_text_embedding_length": int(cfg.get("t_n_ctx", 4)),
    }
    features_list = list(cfg.get("features_list", [6, 12, 18, 24]))

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device.type, design_details=params)
    model.eval()

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    optimizer = torch.optim.Adam(
        list(prompt_learner.parameters()),
        lr=float(cfg.get("learning_rate", 1.0e-3)),
        betas=(0.5, 0.999),
    )

    epochs = int(cfg.get("epochs", 15))
    best = {
        "epoch": 0,
        "val_auroc": -1.0,
        "val_auprc": -1.0,
        "checkpoint": fold_dir / "best_model.pth",
    }
    log_rows: List[Dict] = []

    for epoch in range(1, epochs + 1):
        prompt_learner.train()
        train_losses: List[float] = []

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = _compute_logits(model, prompt_learner, images, features_list)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        val_auroc, val_auprc = _evaluate_image_level(model, prompt_learner, val_loader, device, features_list)

        mean_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(mean_train_loss, 6),
                "val_auroc": round(val_auroc, 6),
                "val_auprc": round(val_auprc, 6),
            }
        )

        improved = (val_auroc > best["val_auroc"]) or (
            val_auroc == best["val_auroc"] and val_auprc > best["val_auprc"]
        )
        if improved:
            best["epoch"] = epoch
            best["val_auroc"] = float(val_auroc)
            best["val_auprc"] = float(val_auprc)
            torch.save(
                {
                    "prompt_learner": prompt_learner.state_dict(),
                    "fold_index": fold_index,
                    "best_epoch": epoch,
                    "val_auroc": float(val_auroc),
                    "val_auprc": float(val_auprc),
                },
                best["checkpoint"],
            )

        print(
            f"[TRAIN] fold={fold_index} epoch={epoch}/{epochs} loss={mean_train_loss:.4f} "
            f"val_auroc={val_auroc:.4f}{' *' if improved else ''}",
            flush=True,
        )

    with (fold_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    fold_result = {
        "fold_index": fold_index,
        "train_n_images": len(train_ids),
        "val_eval_n_images": len(val_eval_ids),
        "best_epoch": best["epoch"],
        "best_val_auroc": best["val_auroc"],
        "best_val_auprc": best["val_auprc"],
        "checkpoint_path": best["checkpoint"].as_posix(),
    }
    with (fold_dir / "fold_results.json").open("w", encoding="utf-8") as f:
        json.dump(fold_result, f, indent=2)

    return fold_result


def _write_summary_csv(rows: List[Dict], out_csv: Path) -> None:
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
        writer.writerows(rows)


def _evaluate_best_fold(
    manifest: dict,
    dataset_root: Path,
    run_dir: Path,
    checkpoint_path: Path,
    cfg: dict,
) -> List[Dict]:
    dataset_name = str(cfg.get("dataset_name", "ISBI"))
    cls_name = str(cfg.get("class_name", "skin"))
    metrics = str(cfg.get("metrics", "image-level"))
    device_mode = str(cfg.get("device", "auto")).lower()
    _validate_device_mode(device_mode)
    subprocess_env = _subprocess_env_for_device(device_mode)

    eval_dir = run_dir / "best_fold_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict] = []

    for test_name in ["test_allfake", "test_gan", "test_ldm", "test_mls"]:
        test_spec = manifest["test_sets"][test_name]
        test_ids = sorted(test_spec["real_ids"] + test_spec["fake_ids"])

        test_dir = eval_dir / test_name
        result_dir = test_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix=f"anomalyclip_adapted_eval_{test_name}_") as tmp_dir:
            temp_root = Path(tmp_dir) / dataset_name
            _build_official_eval_dataset(dataset_root, test_ids, temp_root, cls_name=cls_name)

            test_cmd = [
                sys.executable,
                "test.py",
                "--data_path",
                temp_root.as_posix(),
                "--save_path",
                result_dir.as_posix(),
                "--checkpoint_path",
                checkpoint_path.as_posix(),
                "--dataset",
                dataset_name,
                "--metrics",
                metrics,
            ]
            _run_subprocess(test_cmd, THIRD_PARTY_ANOMALYCLIP, env=subprocess_env)

        parsed_metrics = _extract_image_metrics_from_log(result_dir / "log.txt")
        auroc = round(parsed_metrics["image_auroc"], 6) if parsed_metrics is not None else ""
        auprc = round(parsed_metrics["image_ap"], 6) if parsed_metrics is not None else ""

        with (test_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "test_set": test_name,
                    "n_real": test_spec["n_real"],
                    "n_fake": test_spec["n_fake"],
                    "auroc": auroc,
                    "auprc": auprc,
                    "checkpoint": checkpoint_path.as_posix(),
                    "dataset_name": dataset_name,
                    "metrics": metrics,
                    "device": device_mode,
                    "variant": "anomalyclip_adapted_trained",
                },
                f,
                indent=2,
            )

        summary_rows.append(
            {
                "geometry": "anomalyclip_adapted_trained",
                "test_set": test_name,
                "n_real": test_spec["n_real"],
                "n_fake": test_spec["n_fake"],
                "auroc": auroc,
                "auprc": auprc,
                "accuracy_default": "",
                "f1_default": "",
                "sensitivity_default": "",
                "specificity_default": "",
                "accuracy_f1": "",
                "f1_f1": "",
                "sensitivity_f1": "",
                "specificity_f1": "",
                "accuracy_youden_j": "",
                "f1_youden_j": "",
                "sensitivity_youden_j": "",
                "specificity_youden_j": "",
            }
        )

        print(
            f"[EVAL] test_set={test_name} auroc={auroc if auroc != '' else 'NA'} "
            f"auprc={auprc if auprc != '' else 'NA'}",
            flush=True,
        )

    _write_summary_csv(summary_rows, run_dir / "final_8run_summary.csv")
    return summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adapted AnomalyCLIP training (image-level prompt tuning) with protocol-aligned evaluation"
    )
    parser.add_argument("--config", type=str, default="configs/anomalyclip_adapted_trained.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    cfg = _load_yaml(cfg_path)

    manifest_path = Path(cfg.get("protocol_manifest_path", "configs/manifests/cleaned_protocol_manifest.json"))
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    dataset_root = Path(cfg.get("dataset_root", "RGIIIT_clean"))
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root

    output_root = Path(cfg.get("output_root", "experiments/AnomalyCLIP_Adapted_Trained"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    run_name = cfg.get("run_name", f"anomalyclip_adapted_trained_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "used_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    seed = int(cfg.get("seed", 111))
    _set_seed(seed)

    manifest = _load_manifest(manifest_path)
    with (run_dir / "protocol_manifest_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    device = _resolve_torch_device(str(cfg.get("device", "auto")))
    print(f"[INFO] Using device: {device}", flush=True)

    fold_summaries: List[Dict] = []
    for fold in manifest["cv_folds"]:
        fold_summaries.append(_train_fold(fold, manifest, dataset_root, run_dir, cfg, device))

    best_fold = max(fold_summaries, key=lambda x: (x["best_val_auroc"], x["best_val_auprc"]))
    best_checkpoint = Path(best_fold["checkpoint_path"])

    with (run_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"folds": fold_summaries, "best_fold": best_fold}, f, indent=2)

    summary_rows = _evaluate_best_fold(manifest, dataset_root, run_dir, best_checkpoint, cfg)

    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "variant": "anomalyclip_adapted_trained",
                "best_fold": best_fold,
                "summary_rows": summary_rows,
            },
            f,
            indent=2,
        )

    print(f"[INFO] Completed adapted AnomalyCLIP run at {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
