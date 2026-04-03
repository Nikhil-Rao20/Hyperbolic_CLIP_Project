from __future__ import annotations

import argparse
from pathlib import Path

import yaml

try:
    from .run_winclip_official import run_official_winclip
except ImportError:
    from run_winclip_official import run_official_winclip

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper. Runs official WinCLIP implementation from third_party/WinCLIP "
            "using the local protocol manifest and dataset."
        )
    )
    parser.add_argument("--config", type=str, default="configs/winclip_baseline.yaml")
    parser.add_argument(
        "--object-name",
        type=str,
        default=None,
        help="Override object name used by official WinCLIP MVTec-style loader (default from config or candle)",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=None,
        help="WinCLIP support-shot count (0 for zero-shot, >0 for few-shot)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader workers for WinCLIP evaluation (default from config, recommended 0 on slow CPU)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Override checkpoint path for official WinCLIP weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Execution device mode for WinCLIP (default from config)",
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

    output_root = Path(cfg.get("output_root", "experiments/WinCLIP_Official"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    object_name = args.object_name or cfg.get("object_name", "candle")
    shot = int(args.shot if args.shot is not None else cfg.get("shot", 0))
    num_workers = int(args.num_workers if args.num_workers is not None else cfg.get("num_workers", 0))
    device = (args.device or cfg.get("device", "auto")).lower()
    checkpoint_path = Path(
        args.checkpoint_path
        if args.checkpoint_path is not None
        else cfg.get("checkpoint_path", "third_party/WinCLIP/vit_b_16_plus_240-laion400m_e31-8fb26589.pt")
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    return run_official_winclip(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        output_root=output_root,
        object_name=object_name,
        shot=shot,
        num_workers=num_workers,
        checkpoint_path=checkpoint_path,
        device=device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
