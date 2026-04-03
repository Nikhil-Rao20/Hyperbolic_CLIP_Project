from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import yaml

try:
    from .run_anomalyclip_official import run_official_anomalyclip
except ImportError:
    from run_anomalyclip_official import run_official_anomalyclip

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper. Runs official AnomalyCLIP implementation from third_party/AnomalyCLIP "
            "using the local protocol manifest and dataset."
        )
    )
    parser.add_argument("--config", type=str, default="configs/anomalyclip_baseline.yaml")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Override checkpoint path for official AnomalyCLIP test.py",
    )
    parser.add_argument("--dataset-name", type=str, default=None, help="Override dataset name (default from config)")
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        choices=["image-level", "pixel-level", "image-pixel-level"],
        help="Override evaluation metrics mode",
    )
    parser.add_argument("--class-name", type=str, default=None, help="Override class name (default from config)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Deprecated and ignored in official test-only wrapper",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Execution device mode for AnomalyCLIP (default from config)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    cfg = _load_config(config_path)

    manifest_path = Path(cfg.get("protocol_manifest_path", "configs/manifests/cleaned_protocol_manifest.json"))
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    dataset_root = Path(cfg.get("dataset_root", "RGIIIT_clean"))
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root

    output_root = Path(cfg.get("output_root", "experiments/AnomalyCLIP_Official"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    checkpoint_path = Path(
        args.checkpoint_path
        if args.checkpoint_path is not None
        else cfg.get("checkpoint_path", "third_party/AnomalyCLIP/checkpoints/9_12_4_multiscale_visa/epoch_15.pth")
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    dataset_name = args.dataset_name or cfg.get("dataset_name", "ISBI")
    metrics = args.metrics or cfg.get("metrics", "image-level")
    class_name = args.class_name or cfg.get("class_name", "skin")
    device = (args.device or cfg.get("device", "auto")).lower()

    if args.epochs is not None:
        warnings.warn("--epochs is deprecated and ignored by the official test-only wrapper", stacklevel=2)

    return run_official_anomalyclip(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        output_root=output_root,
        checkpoint_path=checkpoint_path,
        dataset_name=dataset_name,
        metrics=metrics,
        cls_name=class_name,
        device=device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
