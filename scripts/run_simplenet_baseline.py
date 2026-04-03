from __future__ import annotations

import argparse
from pathlib import Path

import yaml

try:
    from .run_simplenet_official import run_official_simplenet
except ImportError:
    from run_simplenet_official import run_official_simplenet

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper. Runs official SimpleNet implementation from third_party/SimpleNet "
            "using the local protocol manifest and dataset."
        )
    )
    parser.add_argument("--config", type=str, default="configs/simplenet_baseline.yaml")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--meta-epochs", type=int, default=None)
    parser.add_argument("--gan-epochs", type=int, default=None)
    parser.add_argument("--backbone-name", type=str, default=None)
    parser.add_argument("--reuse-checkpoint-if-exists", action="store_true")
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

    output_root = Path(cfg.get("output_root", "experiments/SimpleNet_Official"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    layers_to_extract_from = cfg.get("layers_to_extract_from", ["layer2", "layer3"])
    if not isinstance(layers_to_extract_from, list) or len(layers_to_extract_from) == 0:
        raise ValueError("layers_to_extract_from must be a non-empty list")

    device = (args.device or cfg.get("device", "auto")).lower()

    return run_official_simplenet(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        output_root=output_root,
        device=device,
        batch_size=int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 8)),
        num_workers=int(args.num_workers if args.num_workers is not None else cfg.get("num_workers", 0)),
        resize=int(cfg.get("resize", 329)),
        imagesize=int(cfg.get("imagesize", 288)),
        threshold_percentile=float(cfg.get("threshold_percentile", 95.0)),
        seed=(args.seed if args.seed is not None else cfg.get("seed")),
        backbone_name=str(args.backbone_name if args.backbone_name is not None else cfg.get("backbone_name", "wideresnet50")),
        layers_to_extract_from=layers_to_extract_from,
        pretrain_embed_dimension=int(cfg.get("pretrain_embed_dimension", 1536)),
        target_embed_dimension=int(cfg.get("target_embed_dimension", 1536)),
        patchsize=int(cfg.get("patchsize", 3)),
        embedding_size=int(cfg.get("embedding_size", 256)),
        meta_epochs=int(args.meta_epochs if args.meta_epochs is not None else cfg.get("meta_epochs", 40)),
        aed_meta_epochs=int(cfg.get("aed_meta_epochs", 1)),
        gan_epochs=int(args.gan_epochs if args.gan_epochs is not None else cfg.get("gan_epochs", 4)),
        noise_std=float(cfg.get("noise_std", 0.015)),
        dsc_layers=int(cfg.get("dsc_layers", 2)),
        dsc_hidden=None if cfg.get("dsc_hidden", 1024) is None else int(cfg.get("dsc_hidden", 1024)),
        dsc_margin=float(cfg.get("dsc_margin", 0.5)),
        dsc_lr=float(cfg.get("dsc_lr", 2e-4)),
        auto_noise=float(cfg.get("auto_noise", 0.0)),
        train_backbone=bool(cfg.get("train_backbone", False)),
        cos_lr=bool(cfg.get("cos_lr", False)),
        pre_proj=int(cfg.get("pre_proj", 1)),
        proj_layer_type=int(cfg.get("proj_layer_type", 0)),
        mix_noise=int(cfg.get("mix_noise", 1)),
        augment_train=bool(cfg.get("augment_train", False)),
        rotate_degrees=int(cfg.get("rotate_degrees", 0)),
        translate=float(cfg.get("translate", 0.0)),
        scale=float(cfg.get("scale", 0.0)),
        brightness=float(cfg.get("brightness", 0.0)),
        contrast=float(cfg.get("contrast", 0.0)),
        saturation=float(cfg.get("saturation", 0.0)),
        gray=float(cfg.get("gray", 0.0)),
        hflip=float(cfg.get("hflip", 0.0)),
        vflip=float(cfg.get("vflip", 0.0)),
        reuse_checkpoint_if_exists=bool(
            args.reuse_checkpoint_if_exists or cfg.get("reuse_checkpoint_if_exists", False)
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
