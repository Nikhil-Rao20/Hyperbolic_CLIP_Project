from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_ANOMALYCLIP = PROJECT_ROOT / "third_party" / "AnomalyCLIP"
DEFAULT_CHECKPOINT = THIRD_PARTY_ANOMALYCLIP / "checkpoints" / "9_12_4_multiscale_visa" / "epoch_15.pth"


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
    dst_mask.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_image) as img:
        mask = Image.new("L", img.size, color=0)
        mask.save(dst_mask)


def _build_official_eval_dataset(
    dataset_root: Path,
    test_ids: Sequence[str],
    temp_root: Path,
    cls_name: str = "skin",
) -> Path:
    temp_root.mkdir(parents=True, exist_ok=True)
    meta = {"train": {cls_name: []}, "test": {cls_name: []}}
    mask_root = temp_root / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)

    for rel_path in test_ids:
        src = _resolve_source_image(dataset_root, rel_path)
        img_dst = temp_root / rel_path
        _copy_image(src, img_dst)
        is_real = "/Real/" in rel_path.replace("\\", "/")
        item = {
            "img_path": rel_path.replace("\\", "/"),
            "mask_path": str((mask_root / f"{Path(rel_path).stem}_mask.png").relative_to(temp_root)).replace("\\", "/"),
            "cls_name": cls_name,
            "specie_name": "real" if is_real else "fake",
            "anomaly": 0 if is_real else 1,
        }
        _write_blank_mask_like(src, temp_root / item["mask_path"])
        meta["test"][cls_name].append(item)

    with (temp_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return temp_root


def _run_subprocess(args: List[str], cwd: Path, env: Optional[Dict[str, str]] = None) -> None:
    subprocess.run(args, cwd=cwd, check=True, env=env)


def _validate_device_mode(device: str) -> str:
    mode = device.lower()
    if mode not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device mode: {device}. Use one of: auto, cpu, cuda")
    if mode == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this system")
    return mode


def _subprocess_env_for_device(device: str) -> Dict[str, str]:
    env = os.environ.copy()
    if device == "cpu":
        # Hide GPUs from child process so official code runs CPU path deterministically.
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _extract_image_metrics_from_log(log_path: Path) -> Optional[Dict[str, float]]:
    if not log_path.exists():
        return None
    content = log_path.read_text(encoding="utf-8", errors="ignore")
    # AnomalyCLIP logs a markdown table; extract the mean row for image-level metrics.
    match = re.search(r"\|\s*mean\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|", content)
    if not match:
        return None
    image_auroc_pct = float(match.group(1))
    image_ap_pct = float(match.group(2))
    return {
        "image_auroc": image_auroc_pct / 100.0,
        "image_ap": image_ap_pct / 100.0,
    }


def run_official_anomalyclip(
    manifest_path: Path,
    dataset_root: Path,
    output_root: Path,
    checkpoint_path: Path,
    dataset_name: str = "ISBI",
    metrics: str = "image-level",
    cls_name: str = "skin",
    device: str = "auto",
) -> int:
    device = _validate_device_mode(device)
    subprocess_env = _subprocess_env_for_device(device)

    test_entry = THIRD_PARTY_ANOMALYCLIP / "test.py"
    if not test_entry.exists():
        raise FileNotFoundError(
            f"AnomalyCLIP official source not found at {test_entry}. "
            "Run: python scripts/setup_official_baselines.py"
        )

    manifest = _load_manifest(manifest_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Official checkpoint not found at {checkpoint_path}. "
            "Clone AnomalyCLIP with checkpoints or provide --checkpoint-path."
        )

    run_dir = output_root / "anomalyclip_official"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict] = []

    for fold in manifest["cv_folds"]:
        fold_index = int(fold["fold_index"])
        fold_dir = run_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        for test_name in ["test_allfake", "test_gan", "test_ldm", "test_mls"]:
            test_spec = manifest["test_sets"][test_name]
            test_ids = sorted(test_spec["real_ids"] + test_spec["fake_ids"])

            test_dir = fold_dir / test_name
            result_dir = test_dir / "results"
            result_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory(prefix=f"anomalyclip_eval_{fold_index}_{test_name}_") as tmp_dir:
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
                        "fold_index": fold_index,
                        "test_set": test_name,
                        "n_real": test_spec["n_real"],
                        "n_fake": test_spec["n_fake"],
                        "auroc": auroc,
                        "auprc": auprc,
                        "official_repo": "zqhang/AnomalyCLIP",
                        "checkpoint": checkpoint_path.as_posix(),
                        "dataset_name": dataset_name,
                        "metrics": metrics,
                        "device": device,
                    },
                    f,
                    indent=2,
                )

            summary_rows.append(
                {
                    "geometry": "anomalyclip_official",
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

    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "official_repo": "zqhang/AnomalyCLIP",
                "checkpoint_path": checkpoint_path.as_posix(),
                "dataset_name": dataset_name,
                "metrics": metrics,
                "device": device,
                "summary_rows": summary_rows,
            },
            f,
            indent=2,
        )

    with (run_dir / "final_8run_summary.csv").open("w", newline="", encoding="utf-8") as f:
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

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the official AnomalyCLIP test implementation on the local manifest")
    parser.add_argument("--manifest", type=str, default="configs/manifests/cleaned_protocol_manifest.json")
    parser.add_argument("--dataset-root", type=str, default="RGIIIT_clean")
    parser.add_argument("--output-root", type=str, default="experiments/AnomalyCLIP_Official")
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--dataset-name", type=str, default="ISBI")
    parser.add_argument("--metrics", type=str, default="image-level", choices=["image-level", "pixel-level", "image-pixel-level"])
    parser.add_argument("--class-name", type=str, default="skin")
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

    return run_official_anomalyclip(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        output_root=output_root,
        checkpoint_path=checkpoint_path,
        dataset_name=args.dataset_name,
        metrics=args.metrics,
        cls_name=args.class_name,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
