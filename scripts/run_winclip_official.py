from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_WINCLIP = PROJECT_ROOT / "third_party" / "WinCLIP"
WINCLIP_CHECKPOINT_URL = "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt"
WINCLIP_CHECKPOINT_NAME = "vit_b_16_plus_240-laion400m_e31-8fb26589.pt"
DEFAULT_CHECKPOINT = THIRD_PARTY_WINCLIP / WINCLIP_CHECKPOINT_NAME


def _load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


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
    spec = importlib.util.spec_from_file_location("third_party_winclip_main", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import WinCLIP entry point from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys_path_added = False
    if THIRD_PARTY_WINCLIP.as_posix() not in sys.path:
        sys.path.insert(0, THIRD_PARTY_WINCLIP.as_posix())
        sys_path_added = True
    try:
        spec.loader.exec_module(module)
    finally:
        if sys_path_added:
            try:
                sys.path.remove(THIRD_PARTY_WINCLIP.as_posix())
            except ValueError:
                pass
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
    num_workers: int = 0,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    device: str = "auto",
) -> int:
    if shot < 0:
        raise ValueError("shot must be >= 0")
    device = _validate_device_mode(device)
    manifest = _load_manifest(manifest_path)
    support_real_ids = list(manifest.get("real_train", {}).get("image_ids", []))
    if shot > 0 and len(support_real_ids) == 0:
        raise ValueError("Few-shot mode requires manifest real_train.image_ids, but none were found")

    module = _load_winclip_module()
    checkpoint = _ensure_checkpoint(checkpoint_path)

    run_dir = output_root / "winclip_official"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict] = []

    for fold in manifest["cv_folds"]:
        fold_index = int(fold["fold_index"])
        fold_dir = run_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        for test_name in ["test_allfake", "test_gan", "test_ldm", "test_mls"]:
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

            test_dir = fold_dir / test_name
            test_dir.mkdir(parents=True, exist_ok=True)
            with (test_dir / "results.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fold_index": fold_index,
                        "test_set": test_name,
                        "n_real": test_spec["n_real"],
                        "n_fake": test_spec["n_fake"],
                        "auroc": float(auroc),
                        "auprc": float(aupr),
                        "f1_max": float(f1_max),
                        "official_repo": "mala-lab/WinCLIP",
                        "checkpoint": checkpoint.as_posix(),
                        "shot": int(shot),
                        "device": device,
                    },
                    f,
                    indent=2,
                )

            summary_rows.append(
                {
                    "geometry": "winclip_official",
                    "test_set": test_name,
                    "n_real": test_spec["n_real"],
                    "n_fake": test_spec["n_fake"],
                    "auroc": round(float(auroc), 6),
                    "auprc": round(float(aupr), 6),
                    "accuracy_default": "",
                    "f1_default": round(float(f1_max), 6),
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

    with (run_dir / "final_8run_summary.csv").open("w", newline="", encoding="utf-8") as f:
        import csv

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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "official_repo": "mala-lab/WinCLIP",
                "shot": int(shot),
                "summary_rows": summary_rows,
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
        num_workers=args.num_workers,
        checkpoint_path=checkpoint_path,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())