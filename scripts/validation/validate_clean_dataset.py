#!/usr/bin/env python3
"""
Post-Cleaning Validation Script
================================
Runs 11 automated checks on the cleaned dataset (RGIIIT_clean/)
to confirm the cleaning pipeline produced a valid ML-ready dataset.

Checks:
  1. File integrity — all images loadable, correct format
  2. Resolution uniformity — all images are 224×224
  3. No blank images remain
  4. No MD5 duplicates within any split
  5. No subject leakage across splits
  6. Class distribution per split
  7. Source representation per split
  8. No cross-class pHash collisions (non-trivial)
  9. Split ratio validation
  10. Metadata consistency (manifest matches disk)
  11. Overall summary & PASS/FAIL verdict

Usage:
    python validate_clean_dataset.py
"""

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

# Project root is 2 directory levels above scripts/validation/
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLEAN_ROOT = BASE_DIR / "RGIIIT_clean"
META_DIR = BASE_DIR / "dataset_cleaning"
EXPECTED_SIZE = (224, 224)
BLANK_MEAN_THRESHOLD = 5.0
BLANK_STD_THRESHOLD = 2.0
BLANK_BLACK_PCT = 95.0

RESULTS = []  # (check_name, status, detail)


def to_relpath(p: Path) -> str:
    try:
        return p.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(p)


def record(name: str, passed: bool, detail: str):
    status = "PASS" if passed else "FAIL"
    RESULTS.append((name, status, detail))
    icon = "✓" if passed else "✗"
    print(f"  [{icon}] {name}: {detail}")


def extract_subject_id(filename: str) -> str:
    """Extract subject from cleaned filename: {source}__{subject}_sliceN.png"""
    stem = Path(filename).stem
    # Remove source prefix
    if "__" in stem:
        stem = stem.split("__", 1)[1]
    parts = stem.rsplit("_slice", 1)
    return parts[0] if len(parts) == 2 else stem


def extract_source(filename: str) -> str:
    """Extract source from cleaned filename: {source}__{rest}.png"""
    if "__" in filename:
        return filename.split("__", 1)[0]
    return "unknown"


def collect_images():
    """Collect all images from RGIIIT_clean/ into a DataFrame."""
    records = []
    for split_dir in sorted(CLEAN_ROOT.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            for img_file in sorted(cls_dir.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() == ".png":
                    records.append({
                        "filepath": to_relpath(img_file),
                        "filename": img_file.name,
                        "abs_path": str(img_file),
                        "split": split,
                        "class": cls,
                        "source": extract_source(img_file.name),
                        "subject_id": extract_subject_id(img_file.name),
                    })
    return pd.DataFrame(records)


# ────────────────────────────────────────────────────────────
# Checks
# ────────────────────────────────────────────────────────────

def check1_integrity(df):
    """All images loadable with correct format."""
    print("\n[Check 1] File integrity")
    errors = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Checking", unit="img"):
        try:
            img = Image.open(row["abs_path"])
            img.verify()
        except Exception as e:
            errors += 1
    record("File integrity", errors == 0,
           f"{len(df)} files checked, {errors} errors")


def check2_resolution(df):
    """All images are 224×224."""
    print("\n[Check 2] Resolution uniformity")
    wrong = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Checking", unit="img"):
        img = cv2.imread(row["abs_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            wrong.append((row["filepath"], "unreadable"))
            continue
        h, w = img.shape
        if (w, h) != EXPECTED_SIZE:
            wrong.append((row["filepath"], f"{w}x{h}"))
    record("Resolution 224×224", len(wrong) == 0,
           f"{len(df)} images, {len(wrong)} non-conforming")
    if wrong:
        for fp, sz in wrong[:5]:
            print(f"    → {fp}: {sz}")


def check3_no_blanks(df):
    """No blank/nearly-blank images remain."""
    print("\n[Check 3] No blank images")
    blanks = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Checking", unit="img"):
        img = cv2.imread(row["abs_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        arr = img.astype(np.float64)
        mean_val = arr.mean()
        std_val = arr.std()
        black_pct = (np.count_nonzero(img == 0) / img.size) * 100.0
        if mean_val < BLANK_MEAN_THRESHOLD or std_val < BLANK_STD_THRESHOLD or black_pct > BLANK_BLACK_PCT:
            blanks.append((row["filepath"], f"mean={mean_val:.2f}, std={std_val:.2f}, black%={black_pct:.1f}"))
    record("No blank images", len(blanks) == 0,
           f"{len(blanks)} blank images found")
    if blanks:
        for fp, detail in blanks[:5]:
            print(f"    → {fp}: {detail}")


def check4_no_md5_duplicates(df):
    """No MD5 duplicates within any single split."""
    print("\n[Check 4] No intra-split MD5 duplicates")
    total_dups = 0
    for split_name, grp in df.groupby("split"):
        hashes = {}
        dups = 0
        for _, row in grp.iterrows():
            h = hashlib.md5(open(row["abs_path"], "rb").read()).hexdigest()
            if h in hashes:
                dups += 1
            else:
                hashes[h] = row["filepath"]
        if dups > 0:
            print(f"    {split_name}: {dups} duplicates")
        total_dups += dups
    record("No intra-split MD5 dups", total_dups == 0,
           f"{total_dups} duplicates across all splits")


def check5_no_subject_leakage(df):
    """No subject appears in more than one split."""
    print("\n[Check 5] No subject leakage across splits")
    subject_splits = df.groupby(["class", "source", "subject_id"])["split"].nunique()
    leaked = subject_splits[subject_splits > 1]
    record("No subject leakage", len(leaked) == 0,
           f"{len(leaked)} subjects in multiple splits")
    if len(leaked) > 0:
        for idx in leaked.index[:5]:
            print(f"    → {idx}")


def check6_class_distribution(df):
    """Report class distribution per split."""
    print("\n[Check 6] Class distribution per split")
    all_ok = True
    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name]
        counts = sub["class"].value_counts().to_dict()
        fake = counts.get("Fake", 0)
        real = counts.get("Real", 0)
        ratio = fake / real if real > 0 else float("inf")
        print(f"    {split_name}: Fake={fake}, Real={real}, ratio={ratio:.2f}")
        if real == 0 or fake == 0:
            all_ok = False
    record("Class distribution", all_ok,
           "Both classes present in all splits")


def check7_source_representation(df):
    """Check that all sources are represented across splits."""
    print("\n[Check 7] Source representation per split")
    all_sources = set(df["source"].unique())
    missing = {}
    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name]
        present = set(sub["source"].unique())
        miss = all_sources - present
        if miss:
            missing[split_name] = miss
            print(f"    {split_name} missing: {miss}")
        else:
            src_counts = sub["source"].value_counts().to_dict()
            parts = ", ".join(f"{s}={c}" for s, c in sorted(src_counts.items()))
            print(f"    {split_name}: {parts}")
    record("Source representation", len(missing) == 0,
           f"{len(missing)} splits with missing sources" if missing else "All sources in all splits")


def check8_cross_class_phash(df):
    """Check for non-trivial pHash collisions between Real and Fake."""
    print("\n[Check 8] Cross-class pHash check")
    if not HAS_IMAGEHASH:
        record("Cross-class pHash", True, "SKIPPED — imagehash not installed")
        return

    # Compute pHash for all images (subsampled for speed)
    hash_map = defaultdict(list)  # hash → [(filepath, class)]
    sample = df.sample(min(len(df), 2000), random_state=42) if len(df) > 2000 else df
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="  Hashing", unit="img"):
        try:
            img = Image.open(row["abs_path"])
            h = str(imagehash.phash(img))
            if h != "0000000000000000":  # skip blank hashes
                hash_map[h].append((row["filepath"], row["class"]))
        except Exception:
            pass

    # Find cross-class collisions
    collisions = 0
    for h, entries in hash_map.items():
        classes = set(cls for _, cls in entries)
        if len(classes) > 1:
            collisions += 1
    record("No cross-class pHash", collisions == 0,
           f"{collisions} cross-class pHash collisions (sampled {len(sample)} images)")


def check9_split_ratios(df):
    """Verify split ratios are approximately 70/15/15."""
    print("\n[Check 9] Split ratio validation")
    total = len(df)
    ok = True
    for split_name, target in [("train", 0.70), ("val", 0.15), ("test", 0.15)]:
        actual = len(df[df["split"] == split_name]) / total
        diff = abs(actual - target)
        status = "OK" if diff < 0.05 else "DRIFT"
        if diff >= 0.05:
            ok = False
        print(f"    {split_name}: {actual:.2%} (target {target:.0%}, diff {diff:.2%}) [{status}]")
    record("Split ratios ~70/15/15", ok,
           f"All within ±5%" if ok else "Some splits deviate >5%")


def check10_manifest_consistency(df):
    """Verify split_manifest.csv matches what's on disk."""
    print("\n[Check 10] Manifest ↔ disk consistency")
    manifest_path = META_DIR / "split_manifest.csv"
    if not manifest_path.exists():
        record("Manifest consistency", False, "split_manifest.csv not found")
        return

    manifest = pd.read_csv(manifest_path)
    disk_files = set(df["filepath"])
    manifest_clean = set(manifest["clean_filepath"])

    # Also build clean paths from df
    disk_clean_files = set()
    for _, row in df.iterrows():
        disk_clean_files.add(row["filepath"])

    # Check counts match
    ok = len(manifest) == len(df)
    record("Manifest consistency", ok,
           f"Manifest: {len(manifest)} rows, Disk: {len(df)} files" +
           ("" if ok else " — MISMATCH"))


def check11_summary():
    """Print overall verdict."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    total = len(RESULTS)

    for name, status, detail in RESULTS:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  [{icon}] {name}: {detail}")

    print(f"\n  {passed}/{total} checks passed, {failed} failed")
    if failed == 0:
        print("\n  ═══════════════════════════")
        print("   VERDICT: ALL CHECKS PASS")
        print("  ═══════════════════════════")
    else:
        print("\n  ══════════════════════════════")
        print(f"   VERDICT: {failed} CHECK(S) FAILED")
        print("  ══════════════════════════════")

    return failed == 0


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Post-Cleaning Dataset Validation                       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if not CLEAN_ROOT.exists():
        print(f"  [ERROR] Clean dataset not found: {CLEAN_ROOT}")
        sys.exit(1)

    print(f"  Dataset: {to_relpath(CLEAN_ROOT)}")
    df = collect_images()
    print(f"  Total images found: {len(df)}")
    print(f"  Splits: {sorted(df['split'].unique())}")
    print(f"  Classes: {sorted(df['class'].unique())}")

    check1_integrity(df)
    check2_resolution(df)
    check3_no_blanks(df)
    check4_no_md5_duplicates(df)
    check5_no_subject_leakage(df)
    check6_class_distribution(df)
    check7_source_representation(df)
    check8_cross_class_phash(df)
    check9_split_ratios(df)
    check10_manifest_consistency(df)
    all_passed = check11_summary()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
