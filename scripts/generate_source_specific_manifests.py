from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.source_specific_ood import ALL_DOMAINS, load_or_create_manifest


def load_cfg(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_for_config(config_path: Path, force: bool = False):
    cfg = load_cfg(config_path)

    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    experiment_dir = PROJECT_ROOT / cfg.get("experiment_dir")
    seed = int(cfg.get("seed", 42))
    fake_sampling_policy = cfg.get("eval_fake_sampling_policy", "generator_balanced_strict")
    domains = cfg.get("domains", ALL_DOMAINS)

    print(f"\nConfig: {config_path.relative_to(PROJECT_ROOT)}")
    print(f"Dataset: {dataset_path}")
    print(f"Experiment dir: {experiment_dir}")
    print(f"Seed: {seed}")
    print(f"Fake eval policy: {fake_sampling_policy}")

    for domain in domains:
        run_dir = experiment_dir / f"domain_{domain}"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "split_manifest.json"

        if force and manifest_path.exists():
            manifest_path.unlink()

        manifest = load_or_create_manifest(
            manifest_path=manifest_path,
            dataset_root=dataset_path,
            domain=domain,
            seed=seed,
            fake_sampling_policy=fake_sampling_policy,
        )

        vmeta = manifest["val_eval_meta"]
        tmeta = manifest["test_eval_meta"]
        print(
            f"  {domain:>4s} | val(fake_alloc)={vmeta['fake_alloc']} "
            f"test(fake_alloc)={tmeta['fake_alloc']} "
            f"split={manifest['hashes']['global_split_hash'][:10]}"
        )


def main():
    parser = argparse.ArgumentParser(description="Generate source-specific rerun split manifests without training")
    parser.add_argument(
        "--geometry",
        type=str,
        default="both",
        choices=["euclidean", "hyperbolic", "both"],
        help="Which geometry config(s) to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate existing split manifests.",
    )
    args = parser.parse_args()

    eu_cfg = PROJECT_ROOT / "configs/source_specific_ood_euclidean.yaml"
    hyp_cfg = PROJECT_ROOT / "configs/source_specific_ood_hyperbolic.yaml"

    if args.geometry in ["euclidean", "both"]:
        generate_for_config(eu_cfg, force=args.force)
    if args.geometry in ["hyperbolic", "both"]:
        generate_for_config(hyp_cfg, force=args.force)


if __name__ == "__main__":
    main()
