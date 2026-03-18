from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


TARGET_REAL_TRAIN_IMAGES = 500
TARGET_PER_GENERATOR = 104
N_FOLDS = 5
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    rel_path: str
    class_name: str
    generator: str
    source: str
    subject_id: str


_SLICE_SUFFIX_RE = re.compile(r"_slice\d+$", re.IGNORECASE)


def _strip_slice_suffix(stem: str) -> str:
    return _SLICE_SUFFIX_RE.sub("", stem)


def _parse_from_name(stem: str) -> Tuple[str, str]:
    if "__" in stem:
        source, rest = stem.split("__", 1)
        return source, _strip_slice_suffix(rest)
    return "unknown", _strip_slice_suffix(stem)


def _path_has(parts_lower: Sequence[str], token: str) -> bool:
    return any(token in part for part in parts_lower)


def _infer_source_from_parts(parts_lower: Sequence[str], generator: str) -> str:
    if _path_has(parts_lower, "cermep"):
        return "cermep"
    if _path_has(parts_lower, "tcga"):
        return "tcga"
    if _path_has(parts_lower, "upenn"):
        return "upenn"

    if generator == "GAN":
        return "GAN"
    if generator == "LDM":
        return "LDM"
    if generator == "MLS":
        if _path_has(parts_lower, "mls_cermep"):
            return "MLS_CERMEP"
        if _path_has(parts_lower, "mls_tcga"):
            return "MLS_TCGA"
        if _path_has(parts_lower, "mls_upenn"):
            return "MLS_UPenn"
        return "MLS"

    return "unknown"


def _infer_class_and_generator(parts_lower: Sequence[str], stem_lower: str) -> Tuple[str, str]:
    if _path_has(parts_lower, "real") and not _path_has(parts_lower, "fake"):
        return "Real", "Real"

    if _path_has(parts_lower, "gan") or stem_lower.startswith("gan__"):
        return "Fake", "GAN"
    if _path_has(parts_lower, "ldm") or stem_lower.startswith("ldm__"):
        return "Fake", "LDM"

    if _path_has(parts_lower, "mls") or stem_lower.startswith("mls_") or stem_lower.startswith("mls__"):
        return "Fake", "MLS"

    if _path_has(parts_lower, "fake"):
        if stem_lower.startswith("gan"):
            return "Fake", "GAN"
        if stem_lower.startswith("ldm"):
            return "Fake", "LDM"
        if stem_lower.startswith("mls"):
            return "Fake", "MLS"

    raise ValueError("Could not infer class/generator")


def collect_samples(dataset_root: Path) -> List[Sample]:
    samples: List[Sample] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        rel = path.relative_to(dataset_root).as_posix()
        parts = [p.lower() for p in path.relative_to(dataset_root).parts]
        stem = path.stem
        stem_lower = stem.lower()

        try:
            class_name, generator = _infer_class_and_generator(parts, stem_lower)
        except ValueError:
            continue

        source, subject_tail = _parse_from_name(stem)
        if source == "unknown":
            source = _infer_source_from_parts(parts, generator)

        if class_name == "Real":
            subject_id = f"Real::{source}::{subject_tail}"
        else:
            subject_id = f"{generator}::{source}::{subject_tail}"

        samples.append(
            Sample(
                rel_path=rel,
                class_name=class_name,
                generator=generator,
                source=source,
                subject_id=subject_id,
            )
        )

    return samples


def _group_by_subject(samples: Sequence[Sample]) -> Dict[str, List[Sample]]:
    groups: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        groups[s.subject_id].append(s)
    return dict(groups)


def _select_subjects_exact_count(
    subject_groups: Dict[str, List[Sample]],
    target_images: int,
    seed: int,
) -> Tuple[List[str], int]:
    items = [(sid, len(group)) for sid, group in subject_groups.items()]
    rng = random.Random(seed)
    rng.shuffle(items)

    dp: Dict[int, List[str]] = {0: []}
    for sid, cnt in items:
        next_dp = dict(dp)
        for total, chosen in dp.items():
            new_total = total + cnt
            if new_total > target_images:
                continue
            if new_total not in next_dp:
                next_dp[new_total] = chosen + [sid]
        dp = next_dp

    if target_images in dp:
        return dp[target_images], target_images

    feasible = [k for k in dp.keys() if k <= target_images]
    best = max(feasible) if feasible else 0
    return dp.get(best, []), best


def _allocate_proportional_with_caps(target: int, caps: Dict[str, int]) -> Dict[str, int]:
    total_cap = sum(caps.values())
    if target <= 0 or total_cap <= 0:
        return {k: 0 for k in caps}

    effective_target = min(target, total_cap)
    raw = {k: effective_target * (v / total_cap) for k, v in caps.items()}
    alloc = {k: int(raw[k]) for k in caps}

    remainder = effective_target - sum(alloc.values())
    by_fraction = sorted(((raw[k] - alloc[k], k) for k in caps), reverse=True)
    for _, k in by_fraction:
        if remainder <= 0:
            break
        if alloc[k] < caps[k]:
            alloc[k] += 1
            remainder -= 1

    while True:
        overflow = 0
        for k in alloc:
            if alloc[k] > caps[k]:
                overflow += alloc[k] - caps[k]
                alloc[k] = caps[k]
        if overflow == 0:
            break
        for k in sorted(caps.keys()):
            if overflow <= 0:
                break
            room = caps[k] - alloc[k]
            if room <= 0:
                continue
            take = min(room, overflow)
            alloc[k] += take
            overflow -= take

    return alloc


def _sample_without_replacement(samples: Sequence[Sample], n: int, seed: int) -> List[Sample]:
    if n >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    chosen = sorted(idx[:n])
    return [samples[i] for i in chosen]


def _allocate_generator_calibration_counts(target: int, fake_samples: Sequence[Sample]) -> Dict[str, int]:
    caps = Counter(s.generator for s in fake_samples)
    ordered = {k: caps.get(k, 0) for k in ["GAN", "LDM", "MLS"] if caps.get(k, 0) > 0}
    return _allocate_proportional_with_caps(target, ordered)


def _build_subject_folds(
    subject_groups: Dict[str, List[Sample]],
    n_folds: int,
    seed: int,
) -> List[Dict[str, List[str]]]:
    items = [(sid, len(subject_groups[sid])) for sid in subject_groups]
    rng = random.Random(seed)
    rng.shuffle(items)
    items.sort(key=lambda x: x[1], reverse=True)

    fold_subjects: List[List[str]] = [[] for _ in range(n_folds)]
    fold_counts = [0 for _ in range(n_folds)]

    for sid, count in items:
        target_fold = min(range(n_folds), key=lambda i: (fold_counts[i], i))
        fold_subjects[target_fold].append(sid)
        fold_counts[target_fold] += count

    all_subjects = set(subject_groups.keys())
    folds = []
    for i in range(n_folds):
        val_subjects = set(fold_subjects[i])
        train_subjects = all_subjects - val_subjects

        train_ids: List[str] = []
        val_ids: List[str] = []

        for sid in sorted(train_subjects):
            train_ids.extend(sorted(s.rel_path for s in subject_groups[sid]))
        for sid in sorted(val_subjects):
            val_ids.extend(sorted(s.rel_path for s in subject_groups[sid]))

        folds.append(
            {
                "fold_index": i,
                "train_subject_ids": sorted(train_subjects),
                "val_subject_ids": sorted(val_subjects),
                "train_ids": train_ids,
                "val_ids": val_ids,
                "train_n_images": len(train_ids),
                "val_n_images": len(val_ids),
            }
        )

    return folds


def build_protocol_manifest(
    dataset_root: Path,
    seed: int,
    target_real_train_images: int = TARGET_REAL_TRAIN_IMAGES,
    target_per_generator: int = TARGET_PER_GENERATOR,
    n_folds: int = N_FOLDS,
) -> Dict:
    samples = collect_samples(dataset_root)
    real_samples = [s for s in samples if s.class_name == "Real"]
    gan_samples = [s for s in samples if s.generator == "GAN"]
    ldm_samples = [s for s in samples if s.generator == "LDM"]
    mls_samples = [s for s in samples if s.generator == "MLS"]

    if len(real_samples) < target_real_train_images:
        raise RuntimeError(
            f"Insufficient Real images: need {target_real_train_images}, found {len(real_samples)}"
        )

    real_by_subject = _group_by_subject(real_samples)
    train_subject_ids, selected_count = _select_subjects_exact_count(
        real_by_subject,
        target_images=target_real_train_images,
        seed=seed,
    )

    if selected_count != target_real_train_images:
        raise RuntimeError(
            f"Could not select exactly {target_real_train_images} Real images with subject integrity; "
            f"best feasible was {selected_count}."
        )

    train_subject_set = set(train_subject_ids)
    real_train_ids: List[str] = []
    real_test_pool_ids: List[str] = []
    for sid, group in real_by_subject.items():
        ids = sorted(s.rel_path for s in group)
        if sid in train_subject_set:
            real_train_ids.extend(ids)
        else:
            real_test_pool_ids.extend(ids)

    if len(real_train_ids) != target_real_train_images:
        raise RuntimeError("Real train image count mismatch after subject allocation")

    if len(gan_samples) < target_per_generator:
        raise RuntimeError(f"Insufficient GAN images: need {target_per_generator}, found {len(gan_samples)}")
    if len(ldm_samples) < target_per_generator:
        raise RuntimeError(f"Insufficient LDM images: need {target_per_generator}, found {len(ldm_samples)}")
    if len(mls_samples) < target_per_generator:
        raise RuntimeError(f"Insufficient MLS images: need {target_per_generator}, found {len(mls_samples)}")

    gan_ids = sorted(s.rel_path for s in gan_samples)[:target_per_generator]

    ldm_ids = sorted(s.rel_path for s in _sample_without_replacement(ldm_samples, target_per_generator, seed + 11))

    mls_by_source: Dict[str, List[Sample]] = defaultdict(list)
    for s in mls_samples:
        mls_by_source[s.source].append(s)

    source_caps = {k: len(v) for k, v in mls_by_source.items()}
    mls_alloc = _allocate_proportional_with_caps(target_per_generator, source_caps)

    mls_ids: List[str] = []
    mls_alloc_effective: Dict[str, int] = {}
    for source in sorted(mls_by_source.keys()):
        count = mls_alloc.get(source, 0)
        chosen = _sample_without_replacement(mls_by_source[source], count, seed + 21 + abs(hash(source)) % 997)
        mls_ids.extend(sorted(s.rel_path for s in chosen))
        mls_alloc_effective[source] = len(chosen)

    if len(mls_ids) != target_per_generator:
        raise RuntimeError(
            f"MLS allocation mismatch: expected {target_per_generator}, got {len(mls_ids)}"
        )

    fake_test_union = set(gan_ids) | set(ldm_ids) | set(mls_ids)
    calibration_fake_pool = [
        s for s in samples if s.class_name == "Fake" and s.rel_path not in fake_test_union
    ]

    real_test_pool_samples = sorted(real_test_pool_ids)
    needed_real_for_test_sets = target_per_generator * 3 + target_per_generator * 3
    if len(real_test_pool_samples) < needed_real_for_test_sets:
        raise RuntimeError(
            "Insufficient held-out Real images for non-overlapping 4 test sets: "
            f"need {needed_real_for_test_sets}, found {len(real_test_pool_samples)}"
        )

    rng = random.Random(seed + 31)
    shuffled_real_pool = list(real_test_pool_samples)
    rng.shuffle(shuffled_real_pool)

    real_gan = sorted(shuffled_real_pool[0:target_per_generator])
    real_ldm = sorted(shuffled_real_pool[target_per_generator:2 * target_per_generator])
    real_mls = sorted(shuffled_real_pool[2 * target_per_generator:3 * target_per_generator])
    real_all = sorted(shuffled_real_pool[3 * target_per_generator:6 * target_per_generator])

    test_sets = {
        "test_gan": {
            "real_ids": real_gan,
            "fake_ids": gan_ids,
            "n_real": len(real_gan),
            "n_fake": len(gan_ids),
        },
        "test_ldm": {
            "real_ids": real_ldm,
            "fake_ids": ldm_ids,
            "n_real": len(real_ldm),
            "n_fake": len(ldm_ids),
        },
        "test_mls": {
            "real_ids": real_mls,
            "fake_ids": sorted(mls_ids),
            "n_real": len(real_mls),
            "n_fake": len(mls_ids),
            "mls_subsource_counts": dict(sorted(mls_alloc_effective.items())),
        },
        "test_allfake": {
            "real_ids": real_all,
            "fake_ids": sorted(gan_ids + ldm_ids + mls_ids),
            "n_real": len(real_all),
            "n_fake": len(gan_ids) + len(ldm_ids) + len(mls_ids),
        },
    }

    real_union = set(real_gan) | set(real_ldm) | set(real_mls) | set(real_all)
    if len(real_union) != needed_real_for_test_sets:
        raise RuntimeError("Real test-set overlap detected; expected disjoint Real samples across 4 test sets")

    train_subject_groups = {sid: grp for sid, grp in real_by_subject.items() if sid in train_subject_set}
    folds = _build_subject_folds(train_subject_groups, n_folds=n_folds, seed=seed + 41)

    for fold in folds:
        val_target = fold["val_n_images"]
        alloc = _allocate_generator_calibration_counts(val_target, calibration_fake_pool)
        calibration_ids: List[str] = []
        calibration_breakdown: Dict[str, int] = {}
        for generator in ["GAN", "LDM", "MLS"]:
            if alloc.get(generator, 0) <= 0:
                continue
            generator_pool = [s for s in calibration_fake_pool if s.generator == generator]
            chosen = _sample_without_replacement(
                generator_pool,
                alloc[generator],
                seed + 1000 + fold["fold_index"] * 37 + sum(ord(ch) for ch in generator),
            )
            calibration_ids.extend(sorted(s.rel_path for s in chosen))
            calibration_breakdown[generator] = len(chosen)

        fold["calibration_fake_ids"] = sorted(calibration_ids)
        fold["val_eval_ids"] = sorted(fold["val_ids"] + calibration_ids)
        fold["val_eval_n_images"] = len(fold["val_eval_ids"])
        fold["calibration_fake_counts"] = dict(sorted(calibration_breakdown.items()))

    summary = {
        "n_total_images": len(samples),
        "n_real_total": len(real_samples),
        "n_gan_total": len(gan_samples),
        "n_ldm_total": len(ldm_samples),
        "n_mls_total": len(mls_samples),
        "n_real_subjects_total": len(real_by_subject),
        "n_real_train_images": len(real_train_ids),
        "n_real_test_pool_images": len(real_test_pool_ids),
        "n_real_train_subjects": len(train_subject_set),
        "n_real_test_pool_subjects": len(real_by_subject) - len(train_subject_set),
        "n_calibration_fake_pool_images": len(calibration_fake_pool),
        "calibration_fake_pool_by_generator": dict(sorted(Counter(s.generator for s in calibration_fake_pool).items())),
        "mls_source_pool_counts": dict(sorted(Counter(s.source for s in mls_samples).items())),
        "mls_source_sample_counts": dict(sorted(mls_alloc_effective.items())),
    }

    return {
        "dataset_root": dataset_root.as_posix(),
        "seed": seed,
        "target_real_train_images": target_real_train_images,
        "target_per_generator": target_per_generator,
        "n_folds": n_folds,
        "summary": summary,
        "real_train": {
            "subject_ids": sorted(train_subject_set),
            "image_ids": sorted(real_train_ids),
        },
        "real_test_pool": {
            "image_ids": sorted(real_test_pool_ids),
        },
        "fake_balanced_pool": {
            "gan_ids": sorted(gan_ids),
            "ldm_ids": sorted(ldm_ids),
            "mls_ids": sorted(mls_ids),
        },
        "calibration_fake_pool": {
            "image_ids": sorted(s.rel_path for s in calibration_fake_pool),
            "counts_by_generator": dict(sorted(Counter(s.generator for s in calibration_fake_pool).items())),
        },
        "test_sets": test_sets,
        "cv_folds": folds,
    }


def save_manifest(manifest: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
