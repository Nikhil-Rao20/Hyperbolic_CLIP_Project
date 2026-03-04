#!/usr/bin/env python3
"""
MRI Dataset Cleaning & Splitting Pipeline
==========================================
Implements the 11-step cleaning plan for the RGIIIT MRI dataset.

Steps:
  1. Scan all images and extract metadata
  2. Remove nearly-blank / mostly-black boundary slices
  3. MD5 deduplication (keep 1 representative per group)
  4. Quarantine cross-class pHash matches (non-blank only)
  5. Subject-level stratified 70 / 15 / 15 split
  6. Resize to 224×224, save as PNG into RGIIIT_clean/
  7. Generate metadata (split_manifest.csv, cleaning_log.csv,
     subject_split_map.json, dataset_config.json)

Usage:
    python clean_and_split_dataset.py
"""

import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────
# Project root is 2 directory levels above scripts/cleaning/
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = BASE_DIR / "RGIIIT"
OUTPUT_ROOT = BASE_DIR / "RGIIIT_clean"
LOG_DIR = BASE_DIR / "dataset_cleaning"

# Cleaning thresholds
BLANK_MEAN_THRESHOLD = 5.0        # mean intensity < 5 → blank
BLANK_BLACK_PCT_THRESHOLD = 95.0  # >95% black pixels → blank
BLANK_STD_THRESHOLD = 2.0         # std < 2 → nearly uniform / blank

# Target resolution
TARGET_SIZE = (224, 224)

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
SEED = 42

# Source → class mapping (derived from folder hierarchy)
FAKE_SOURCES = {"GAN", "LDM", "MLS_CERMEP", "MLS_TCGA", "MLS_UPenn"}
REAL_SOURCES = {"cermep", "tcga", "upenn"}

# Source folder mapping
SOURCE_PATHS = {
    "GAN":        DATASET_ROOT / "NewFake_W" / "GAN",
    "LDM":        DATASET_ROOT / "NewFake_W" / "LDM",
    "MLS_CERMEP": DATASET_ROOT / "NewFake_W" / "MLS_CERMEP",
    "MLS_TCGA":   DATASET_ROOT / "NewFake_W" / "MLS_TCGA",
    "MLS_UPenn":  DATASET_ROOT / "NewFake_W" / "MLS_UPenn",
    "cermep":     DATASET_ROOT / "NewReal" / "cermep",
    "tcga":       DATASET_ROOT / "NewReal" / "tcga",
    "upenn":      DATASET_ROOT / "NewReal" / "upenn",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

np.random.seed(SEED)


def to_relpath(p: Path) -> str:
    """Convert absolute path to workspace-relative POSIX string."""
    try:
        return p.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(p)


def extract_subject_id(filename: str) -> str:
    """Extract subject ID from a filename by removing _sliceN suffix."""
    stem = Path(filename).stem
    parts = stem.rsplit("_slice", 1)
    return parts[0] if len(parts) == 2 else stem


def get_class_label(source: str) -> str:
    if source in FAKE_SOURCES:
        return "Fake"
    elif source in REAL_SOURCES:
        return "Real"
    raise ValueError(f"Unknown source: {source}")


def md5_file(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_image_stats(filepath: Path) -> dict:
    """Compute per-image statistics needed for cleaning decisions."""
    img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    arr = img.astype(np.float64)
    mean_val = arr.mean()
    std_val = arr.std()
    total_px = h * w
    black_pct = (np.count_nonzero(img == 0) / total_px) * 100.0
    return {
        "width": w,
        "height": h,
        "mean_intensity": round(mean_val, 4),
        "std_intensity": round(std_val, 4),
        "pct_black_pixels": round(black_pct, 4),
    }


# ────────────────────────────────────────────────────────────
# Step 1: Scan & Catalogue
# ────────────────────────────────────────────────────────────
def step1_scan_dataset():
    """Scan all images; collect path, source, class, subject, stats, MD5."""
    print("\n" + "=" * 60)
    print("STEP 1 — Scanning dataset & computing image statistics + MD5")
    print("=" * 60)

    records = []
    for source, src_dir in SOURCE_PATHS.items():
        if not src_dir.exists():
            print(f"  [WARN] Source directory missing: {src_dir}")
            continue
        files = sorted([
            f for f in src_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])
        cls = get_class_label(source)
        for f in tqdm(files, desc=f"  {source}", unit="img"):
            stats = compute_image_stats(f)
            if stats is None:
                continue
            records.append({
                "filepath": to_relpath(f),
                "filename": f.name,
                "class": cls,
                "source": source,
                "subject_id": extract_subject_id(f.name),
                "md5": md5_file(f),
                **stats,
            })

    df = pd.DataFrame(records)
    print(f"\n  Total images scanned: {len(df)}")
    return df


# ────────────────────────────────────────────────────────────
# Step 2: Remove blank / nearly-blank boundary slices
# ────────────────────────────────────────────────────────────
def step2_remove_blanks(df: pd.DataFrame, log: list):
    """Flag images that are blank/nearly-blank for removal."""
    print("\n" + "=" * 60)
    print("STEP 2 — Flagging blank / nearly-blank images")
    print("=" * 60)

    blank_mask = (
        (df["mean_intensity"] < BLANK_MEAN_THRESHOLD) |
        (df["pct_black_pixels"] > BLANK_BLACK_PCT_THRESHOLD) |
        (df["std_intensity"] < BLANK_STD_THRESHOLD)
    )
    blanks = df[blank_mask]
    keep = df[~blank_mask].copy()

    # Log removals
    for _, row in blanks.iterrows():
        reasons = []
        if row["mean_intensity"] < BLANK_MEAN_THRESHOLD:
            reasons.append(f"mean={row['mean_intensity']:.2f}<{BLANK_MEAN_THRESHOLD}")
        if row["pct_black_pixels"] > BLANK_BLACK_PCT_THRESHOLD:
            reasons.append(f"black%={row['pct_black_pixels']:.1f}>{BLANK_BLACK_PCT_THRESHOLD}")
        if row["std_intensity"] < BLANK_STD_THRESHOLD:
            reasons.append(f"std={row['std_intensity']:.2f}<{BLANK_STD_THRESHOLD}")
        log.append({
            "action": "removed_blank",
            "filepath": row["filepath"],
            "source": row["source"],
            "class": row["class"],
            "reason": "; ".join(reasons),
        })

    print(f"  Removed: {len(blanks)} blank/nearly-blank images")
    print(f"  Per-source breakdown:")
    if len(blanks) > 0:
        for src, cnt in blanks.groupby("source").size().items():
            print(f"    {src}: {cnt}")
    print(f"  Remaining: {len(keep)}")
    return keep


# ────────────────────────────────────────────────────────────
# Step 3: MD5 deduplication
# ────────────────────────────────────────────────────────────
def step3_dedup_md5(df: pd.DataFrame, log: list):
    """Remove exact MD5 duplicates, keeping 1 per group."""
    print("\n" + "=" * 60)
    print("STEP 3 — MD5 deduplication")
    print("=" * 60)

    before = len(df)
    dup_groups = df.groupby("md5").filter(lambda g: len(g) > 1)

    if len(dup_groups) == 0:
        print("  No MD5 duplicates found (blanks already removed).")
        return df

    # For each MD5 group, keep the first occurrence, log the rest
    keep_idx = set()
    remove_idx = set()
    for md5_val, group in df.groupby("md5"):
        if len(group) == 1:
            keep_idx.add(group.index[0])
        else:
            sorted_group = group.sort_values("filepath")
            keep_idx.add(sorted_group.index[0])
            for idx in sorted_group.index[1:]:
                remove_idx.add(idx)
                log.append({
                    "action": "removed_md5_duplicate",
                    "filepath": df.loc[idx, "filepath"],
                    "source": df.loc[idx, "source"],
                    "class": df.loc[idx, "class"],
                    "reason": f"MD5 duplicate of {df.loc[sorted_group.index[0], 'filepath']}",
                })

    keep = df.loc[~df.index.isin(remove_idx)].copy()
    removed = before - len(keep)
    print(f"  Removed: {removed} MD5 duplicates")
    print(f"  Remaining: {len(keep)}")
    return keep


# ────────────────────────────────────────────────────────────
# Step 4: Cross-class pHash quarantine
# ────────────────────────────────────────────────────────────
def step4_quarantine_phash(df: pd.DataFrame, log: list):
    """
    Quarantine images with non-trivial pHash matches across classes.
    Uses the data_leakage_report.csv from the audit.
    Only acts on non-zero pHash matches (zero = blank, already removed).
    """
    print("\n" + "=" * 60)
    print("STEP 4 — Cross-class pHash quarantine")
    print("=" * 60)

    leakage_csv = BASE_DIR / "dataset_audit" / "reports" / "data_leakage_report.csv"
    if not leakage_csv.exists():
        print("  [WARN] data_leakage_report.csv not found — skipping pHash quarantine.")
        return df

    leak_df = pd.read_csv(leakage_csv)
    # Only care about cross_class_phash with non-zero hashes
    cross = leak_df[
        (leak_df["check"] == "cross_class_phash") &
        (~leak_df["detail"].str.contains("pHash=0000000000000000", na=False))
    ]

    if len(cross) == 0:
        print("  No non-trivial cross-class pHash matches to quarantine.")
        return df

    # Collect unique filepaths to quarantine
    quarantine_files = set()
    for _, row in cross.iterrows():
        # Quarantine the Fake file (not the Real), since we want to preserve originals
        quarantine_files.add(row["fake_file"])

    remaining_paths = set(df["filepath"])
    to_quarantine = quarantine_files & remaining_paths

    quarantine_mask = df["filepath"].isin(to_quarantine)
    quarantined = df[quarantine_mask]
    keep = df[~quarantine_mask].copy()

    for _, row in quarantined.iterrows():
        log.append({
            "action": "quarantined_phash",
            "filepath": row["filepath"],
            "source": row["source"],
            "class": row["class"],
            "reason": "Cross-class pHash match with Real image (non-blank)",
        })

    print(f"  Quarantined: {len(quarantined)} images with cross-class pHash matches")
    print(f"  Remaining: {len(keep)}")
    return keep


# ────────────────────────────────────────────────────────────
# Step 5: Subject-level stratified split
# ────────────────────────────────────────────────────────────
def step5_subject_split(df: pd.DataFrame):
    """
    Split by subject ID, stratified per (class, source) group.
    Ensures no subject has slices in multiple splits.
    Ratios: 70% train, 15% val, 15% test.
    """
    print("\n" + "=" * 60)
    print("STEP 5 — Subject-level stratified split (70/15/15)")
    print("=" * 60)

    rng = np.random.RandomState(SEED)
    split_assignments = {}  # subject_id → split

    # Group subjects by (class, source)
    subject_groups = df.groupby(["class", "source", "subject_id"]).size().reset_index(name="n_slices")

    for (cls, src), grp in subject_groups.groupby(["class", "source"]):
        subjects = grp["subject_id"].unique().tolist()
        rng.shuffle(subjects)
        n = len(subjects)
        n_train = max(1, int(round(n * TRAIN_RATIO)))
        n_val = max(1 if n > 1 else 0, int(round(n * VAL_RATIO)))
        n_test = n - n_train - n_val
        # Safety: ensure at least 1 in test if possible
        if n_test <= 0 and n > 2:
            n_test = 1
            n_train = n - n_val - n_test

        train_subj = subjects[:n_train]
        val_subj = subjects[n_train:n_train + n_val]
        test_subj = subjects[n_train + n_val:]

        for s in train_subj:
            split_assignments[f"{cls}/{src}/{s}"] = "train"
        for s in val_subj:
            split_assignments[f"{cls}/{src}/{s}"] = "val"
        for s in test_subj:
            split_assignments[f"{cls}/{src}/{s}"] = "test"

    # Assign split to each image row
    df = df.copy()
    df["split"] = df.apply(
        lambda r: split_assignments.get(f"{r['class']}/{r['source']}/{r['subject_id']}", "train"),
        axis=1,
    )

    # Print summary
    for split_name in ["train", "val", "test"]:
        subset = df[df["split"] == split_name]
        n_subj = subset.groupby(["class", "source", "subject_id"]).ngroups
        print(f"  {split_name}: {len(subset)} images, {n_subj} subjects")
        for cls in ["Fake", "Real"]:
            cls_sub = subset[subset["class"] == cls]
            print(f"    {cls}: {len(cls_sub)} images")

    # Verify no subject leaks across splits
    subj_splits = df.groupby(["class", "source", "subject_id"])["split"].nunique()
    leaked = subj_splits[subj_splits > 1]
    if len(leaked) > 0:
        print(f"  [ERROR] {len(leaked)} subjects leaked across splits!")
    else:
        print("  ✓ No subject leakage across splits.")

    return df, split_assignments


# ────────────────────────────────────────────────────────────
# Step 6: Resize, convert, and copy to RGIIIT_clean/
# ────────────────────────────────────────────────────────────
def step6_build_clean_dataset(df: pd.DataFrame, log: list):
    """
    Resize to 224×224 using LANCZOS, save as PNG into:
        RGIIIT_clean/{split}/{class}/{source}__{filename}.png

    Filenames are prefixed with source to maintain provenance.
    """
    print("\n" + "=" * 60)
    print("STEP 6 — Resize to 224×224, save as PNG → RGIIIT_clean/")
    print("=" * 60)

    if OUTPUT_ROOT.exists():
        print(f"  Removing existing {to_relpath(OUTPUT_ROOT)}/...")
        shutil.rmtree(OUTPUT_ROOT)

    saved = 0
    errors = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Copying", unit="img"):
        src_path = BASE_DIR / row["filepath"]
        split = row["split"]
        cls = row["class"]
        source = row["source"]
        fname = Path(row["filename"]).stem + ".png"
        # Prefix with source for provenance tracking
        out_name = f"{source}__{fname}"
        out_dir = OUTPUT_ROOT / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_name

        try:
            img = Image.open(src_path).convert("L")
            img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)
            img_resized.save(out_path, format="PNG")
            saved += 1
        except Exception as e:
            errors += 1
            log.append({
                "action": "copy_error",
                "filepath": row["filepath"],
                "source": source,
                "class": cls,
                "reason": str(e),
            })

    print(f"  Saved: {saved} images")
    if errors:
        print(f"  Errors: {errors}")
    return saved


# ────────────────────────────────────────────────────────────
# Step 7: Generate metadata files
# ────────────────────────────────────────────────────────────
def step7_generate_metadata(df: pd.DataFrame, split_assignments: dict, log: list, start_ts: str):
    """
    Generate:
      - split_manifest.csv   (per-image manifest)
      - cleaning_log.csv     (all cleaning actions)
      - subject_split_map.json
      - dataset_config.json
    """
    print("\n" + "=" * 60)
    print("STEP 7 — Generating metadata")
    print("=" * 60)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 7a. split_manifest.csv
    manifest = df[["filepath", "filename", "class", "source", "subject_id", "split",
                    "width", "height", "mean_intensity", "std_intensity"]].copy()
    # Add the clean filepath
    manifest["clean_filepath"] = manifest.apply(
        lambda r: f"RGIIIT_clean/{r['split']}/{r['class']}/{r['source']}__{Path(r['filename']).stem}.png",
        axis=1,
    )
    manifest_path = LOG_DIR / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"  → {to_relpath(manifest_path)}  ({len(manifest)} rows)")

    # 7b. cleaning_log.csv
    log_path = LOG_DIR / "cleaning_log.csv"
    log_df = pd.DataFrame(log)
    log_df.to_csv(log_path, index=False)
    print(f"  → {to_relpath(log_path)}  ({len(log_df)} actions)")

    # 7c. subject_split_map.json
    ssmap_path = LOG_DIR / "subject_split_map.json"
    with open(ssmap_path, "w") as f:
        json.dump(split_assignments, f, indent=2)
    print(f"  → {to_relpath(ssmap_path)}  ({len(split_assignments)} subjects)")

    # 7d. dataset_config.json
    config = {
        "created": start_ts,
        "source_dataset": "RGIIIT",
        "clean_dataset": "RGIIIT_clean",
        "target_resolution": list(TARGET_SIZE),
        "image_format": "PNG",
        "color_mode": "grayscale",
        "dtype": "uint8",
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "split_strategy": "subject-level, stratified by (class, source)",
        "seed": SEED,
        "cleaning_thresholds": {
            "blank_mean": BLANK_MEAN_THRESHOLD,
            "blank_black_pct": BLANK_BLACK_PCT_THRESHOLD,
            "blank_std": BLANK_STD_THRESHOLD,
        },
        "class_mapping": {"Fake": 0, "Real": 1},
        "sources": {
            "Fake": sorted(FAKE_SOURCES),
            "Real": sorted(REAL_SOURCES),
        },
        "total_clean_images": len(df),
        "per_split": {
            split: {
                "total": int((df["split"] == split).sum()),
                "Fake": int(((df["split"] == split) & (df["class"] == "Fake")).sum()),
                "Real": int(((df["split"] == split) & (df["class"] == "Real")).sum()),
            }
            for split in ["train", "val", "test"]
        },
    }
    config_path = LOG_DIR / "dataset_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  → {to_relpath(config_path)}")


# ────────────────────────────────────────────────────────────
# Step 8: Post-clean summary
# ────────────────────────────────────────────────────────────
def step8_summary(df_original: pd.DataFrame, df_clean: pd.DataFrame, log: list, elapsed: float):
    """Print a final summary of the cleaning."""
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE — SUMMARY")
    print("=" * 60)

    n_orig = len(df_original)
    n_clean = len(df_clean)
    n_removed = n_orig - n_clean

    print(f"\n  Original images:   {n_orig}")
    print(f"  Removed:           {n_removed}  ({n_removed / n_orig * 100:.1f}%)")
    print(f"  Clean images:      {n_clean}")
    print()

    # Breakdown of removals
    action_counts = Counter(entry["action"] for entry in log)
    for action, count in sorted(action_counts.items()):
        print(f"    {action}: {count}")

    print(f"\n  Target resolution: {TARGET_SIZE[0]}×{TARGET_SIZE[1]} PNG (grayscale)")
    print(f"\n  Split distribution:")
    for split in ["train", "val", "test"]:
        sub = df_clean[df_clean["split"] == split]
        fake = len(sub[sub["class"] == "Fake"])
        real = len(sub[sub["class"] == "Real"])
        print(f"    {split:5s}: {len(sub):5d}  (Fake={fake}, Real={real})")

    print(f"\n  Elapsed time: {elapsed:.1f} seconds")
    print(f"  Clean dataset: {to_relpath(OUTPUT_ROOT)}/")
    print(f"  Metadata:      {to_relpath(LOG_DIR)}/")


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    start = time.time()
    start_ts = datetime.now().isoformat()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   MRI Dataset Cleaning & Splitting Pipeline              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Source:  {to_relpath(DATASET_ROOT)}")
    print(f"  Output:  {to_relpath(OUTPUT_ROOT)}")
    print(f"  Target:  {TARGET_SIZE[0]}×{TARGET_SIZE[1]} PNG grayscale")
    print(f"  Splits:  {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"  Seed:    {SEED}")

    log = []  # cleaning log accumulates across all steps

    # Step 1 — Scan
    df_all = step1_scan_dataset()

    # Step 2 — Remove blanks
    df_clean = step2_remove_blanks(df_all, log)

    # Step 3 — MD5 dedup
    df_clean = step3_dedup_md5(df_clean, log)

    # Step 4 — pHash quarantine
    df_clean = step4_quarantine_phash(df_clean, log)

    # Step 5 — Subject-level split
    df_clean, split_assignments = step5_subject_split(df_clean)

    # Step 6 — Build clean dataset
    step6_build_clean_dataset(df_clean, log)

    # Step 7 — Metadata
    step7_generate_metadata(df_clean, split_assignments, log, start_ts)

    # Step 8 — Summary
    elapsed = time.time() - start
    step8_summary(df_all, df_clean, log, elapsed)

    print("\n✓ Pipeline finished successfully.\n")


if __name__ == "__main__":
    main()
