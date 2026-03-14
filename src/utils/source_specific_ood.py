"""Utilities for source-specific one-class OOD experiments.

This module centralizes deterministic split generation and manifest persistence so
experiments are reproducible across machines (local, Kaggle, different GPUs).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

LABEL_MAP = {"Real": 0, "Fake": 1}
REAL_SOURCES = ["cermep", "tcga", "upenn"]
GENERATOR_SOURCES = {
    "GAN": ["GAN"],
    "LDM": ["LDM"],
    "MLS": ["MLS_CERMEP", "MLS_TCGA", "MLS_UPenn"],
}
ALL_DOMAINS = ["Real", "GAN", "LDM", "MLS"]


def source_from_path(path: Path) -> str:
    stem = path.stem
    return stem.split("__")[0] if "__" in stem else "unknown"


def set_global_determinism(seed: int):
    """Set deterministic controls for Python/NumPy/PyTorch."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(base_seed: int):
    def _init(worker_id: int):
        s = base_seed + worker_id
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    return _init


def build_loader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def collect_split_samples(dataset_root: Path, split: str) -> List[Dict]:
    """Collect sorted sample metadata from RGIIIT_clean split directory."""
    out: List[Dict] = []
    split_root = dataset_root / split

    for class_name, label in LABEL_MAP.items():
        class_dir = split_root / class_name
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.glob("*.png")):
            rel_path = img_path.relative_to(dataset_root).as_posix()
            out.append(
                {
                    "rel_path": rel_path,
                    "label": label,
                    "source": source_from_path(img_path),
                    "split": split,
                }
            )

    out.sort(key=lambda s: s["rel_path"])
    return out


def is_in_domain(sample: Dict, domain: str) -> bool:
    if domain == "Real":
        return sample["label"] == 0 and sample["source"] in REAL_SOURCES
    if domain in GENERATOR_SOURCES:
        return sample["label"] == 1 and sample["source"] in GENERATOR_SOURCES[domain]
    raise ValueError(f"Unknown domain: {domain}")


def _allocate_proportional(total_needed: int, capacities: Dict[str, int]) -> Dict[str, int]:
    """Allocate counts proportionally with deterministic rounding and redistribution."""
    total_capacity = sum(capacities.values())
    if total_capacity <= 0:
        return {k: 0 for k in capacities}

    target = min(total_needed, total_capacity)
    raw = {k: target * (v / total_capacity) for k, v in capacities.items()}
    alloc = {k: int(np.floor(raw[k])) for k in capacities}

    # Largest-fraction remainder fill.
    rem = target - sum(alloc.values())
    fractions = sorted(((raw[k] - alloc[k], k) for k in capacities), reverse=True)
    for _, k in fractions:
        if rem <= 0:
            break
        if alloc[k] < capacities[k]:
            alloc[k] += 1
            rem -= 1

    # Capacity-constrained redistribution if any bucket still exceeds capacity.
    overflow = True
    while overflow:
        overflow = False
        extra = 0
        for k in alloc:
            if alloc[k] > capacities[k]:
                extra += alloc[k] - capacities[k]
                alloc[k] = capacities[k]
                overflow = True
        if extra > 0:
            for k in sorted(capacities.keys()):
                if extra <= 0:
                    break
                room = capacities[k] - alloc[k]
                if room <= 0:
                    continue
                take = min(room, extra)
                alloc[k] += take
                extra -= take

    return alloc


def _sample_without_replacement(pool: Sequence[Dict], n: int, rng: np.random.Generator) -> List[Dict]:
    if n <= 0:
        return []
    if n >= len(pool):
        return list(pool)
    idxs = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(idxs.tolist())]


def build_balanced_eval_subset(
    split_samples: Sequence[Dict],
    seed: int,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Build class-balanced subset with proportional fake-generator composition."""
    rng = np.random.default_rng(seed)

    real_pool = [s for s in split_samples if s["label"] == 0 and s["source"] in REAL_SOURCES]
    fake_pools = {
        "GAN": [s for s in split_samples if s["label"] == 1 and s["source"] in GENERATOR_SOURCES["GAN"]],
        "LDM": [s for s in split_samples if s["label"] == 1 and s["source"] in GENERATOR_SOURCES["LDM"]],
        "MLS": [s for s in split_samples if s["label"] == 1 and s["source"] in GENERATOR_SOURCES["MLS"]],
    }

    n_real = len(real_pool)
    n_fake_total = sum(len(v) for v in fake_pools.values())
    target = min(n_real, n_fake_total)

    fake_caps = {k: len(v) for k, v in fake_pools.items()}
    fake_alloc = _allocate_proportional(target, fake_caps)

    real_selected = _sample_without_replacement(real_pool, target, rng)
    fake_selected = []
    for k in ["GAN", "LDM", "MLS"]:
        fake_selected.extend(_sample_without_replacement(fake_pools[k], fake_alloc[k], rng))

    selected = sorted(real_selected + fake_selected, key=lambda s: s["rel_path"])

    meta = {
        "n_real_selected": len(real_selected),
        "n_fake_selected": len(fake_selected),
        "n_total_selected": len(selected),
        "fake_alloc": fake_alloc,
        "target_per_class": target,
        "n_real_pool": n_real,
        "n_fake_pool": n_fake_total,
    }
    return selected, meta


def _hash_ids(ids: Sequence[str]) -> str:
    joined = "\n".join(ids)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def split_hash(train_ids: Sequence[str], val_eval_ids: Sequence[str], test_eval_ids: Sequence[str]) -> str:
    payload = (
        "train_ids\n" + "\n".join(train_ids) + "\n"
        "val_eval_ids\n" + "\n".join(val_eval_ids) + "\n"
        "test_eval_ids\n" + "\n".join(test_eval_ids)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_payload(
    domain: str,
    seed: int,
    train_ids: Sequence[str],
    val_in_domain_ids: Sequence[str],
    val_eval_ids: Sequence[str],
    test_eval_ids: Sequence[str],
    val_meta: Dict,
    test_meta: Dict,
):
    return {
        "domain": domain,
        "seed": seed,
        "train_ids": list(train_ids),
        "val_in_domain_ids": list(val_in_domain_ids),
        "val_eval_ids": list(val_eval_ids),
        "test_eval_ids": list(test_eval_ids),
        "hashes": {
            "train_hash": _hash_ids(train_ids),
            "val_in_domain_hash": _hash_ids(val_in_domain_ids),
            "val_eval_hash": _hash_ids(val_eval_ids),
            "test_eval_hash": _hash_ids(test_eval_ids),
            "global_split_hash": split_hash(train_ids, val_eval_ids, test_eval_ids),
        },
        "val_eval_meta": val_meta,
        "test_eval_meta": test_meta,
        "determinism": {
            "pythonhashseed": str(seed),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "torch_use_deterministic_algorithms": True,
        },
    }


def load_or_create_manifest(manifest_path: Path, dataset_root: Path, domain: str, seed: int) -> Dict:
    """Load fixed split manifest if present; else create deterministically and save."""
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    train_samples = collect_split_samples(dataset_root, "train")
    val_samples = collect_split_samples(dataset_root, "val")
    test_samples = collect_split_samples(dataset_root, "test")

    train_in_domain = sorted(
        [s for s in train_samples if is_in_domain(s, domain)], key=lambda s: s["rel_path"]
    )
    val_in_domain = sorted(
        [s for s in val_samples if is_in_domain(s, domain)], key=lambda s: s["rel_path"]
    )

    val_eval, val_meta = build_balanced_eval_subset(val_samples, seed=seed + 101)
    test_eval, test_meta = build_balanced_eval_subset(test_samples, seed=seed + 202)

    payload = _manifest_payload(
        domain=domain,
        seed=seed,
        train_ids=[s["rel_path"] for s in train_in_domain],
        val_in_domain_ids=[s["rel_path"] for s in val_in_domain],
        val_eval_ids=[s["rel_path"] for s in val_eval],
        test_eval_ids=[s["rel_path"] for s in test_eval],
        val_meta=val_meta,
        test_meta=test_meta,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def path_to_label_source(rel_path: str) -> Tuple[int, str]:
    path = Path(rel_path)
    label = 0 if path.parts[1] == "Real" else 1
    source = source_from_path(path)
    return label, source
