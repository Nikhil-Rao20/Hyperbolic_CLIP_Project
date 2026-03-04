#!/usr/bin/env python3
"""
=============================================================================
COMPREHENSIVE MRI DATASET AUDIT SCRIPT
=============================================================================
10-Step rigorous and systematic audit for the RGIIIT MRI brain image dataset.

Steps:
  1. Dataset Structure Mapping
  2. File Integrity Verification
  3. Image Statistics Collection
  4. Class Balance Analysis
  5. Duplicate Detection (exact MD5 + perceptual hashing)
  6. Data Leakage Check (cross-source overlap)
  7. Visual Inspection Samples (grid montages)
  8. MRI-Specific Checks (intensity anomalies, blank/corrupt)
  9. Synthetic Artifact Analysis (FFT-based)
 10. Final Summary Report

Outputs saved to: dataset_audit/
=============================================================================
"""

import os
import sys
import json
import csv
import hashlib
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import cv2
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# Try optional imports
try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("[WARN] imagehash not installed. Perceptual hashing will be skipped.")

# ===========================================================================
# CONFIGURATION
# ===========================================================================
# Project root is 2 directory levels above scripts/audit/
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = BASE_DIR / "RGIIIT"
OUTPUT_DIR = BASE_DIR / "dataset_audit"
REPORTS_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR = OUTPUT_DIR / "plots"
SAMPLES_DIR = OUTPUT_DIR / "samples"
LOGS_DIR = OUTPUT_DIR / "logs"

# Ensure output dirs
for d in [REPORTS_DIR, PLOTS_DIR, SAMPLES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging setup
log_file = LOGS_DIR / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Class mapping
CLASS_MAP = {
    "NewFake_W": "Fake",
    "NewReal": "Real"
}

# Sub-source directories
FAKE_SOURCES = ["GAN", "LDM", "MLS_CERMEP", "MLS_TCGA", "MLS_UPenn"]
REAL_SOURCES = ["cermep", "tcga", "upenn"]

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================
def md5_hash(filepath):
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(filepath):
    """Compute perceptual hash using imagehash."""
    if not HAS_IMAGEHASH:
        return None
    try:
        img = Image.open(filepath).convert('L')
        return str(imagehash.phash(img))
    except Exception:
        return None


def load_image_safe(filepath):
    """Safely load image with OpenCV, returns (image, error_msg)."""
    try:
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, "cv2.imread returned None"
        return img, None
    except Exception as e:
        return None, str(e)


def to_relpath(p):
    """Convert an absolute path to a path relative to BASE_DIR (workspace root)."""
    try:
        return str(Path(p).relative_to(BASE_DIR)).replace('\\', '/')
    except ValueError:
        return str(p)


def get_file_info(filepath):
    """Get basic file metadata."""
    p = Path(filepath)
    return {
        'path': to_relpath(p),
        'name': p.name,
        'extension': p.suffix.lower(),
        'size_bytes': p.stat().st_size,
    }


# ===========================================================================
# STEP 1: DATASET STRUCTURE MAPPING
# ===========================================================================
def step1_dataset_structure():
    """Map and document the complete dataset structure."""
    logger.info("=" * 70)
    logger.info("STEP 1: Dataset Structure Mapping")
    logger.info("=" * 70)

    structure = {
        "dataset_root": to_relpath(DATASET_ROOT),
        "scan_time": datetime.now().isoformat(),
        "classes": {},
        "total_files": 0,
        "total_dirs": 0,
        "sources": {}
    }

    all_files = []

    for class_dir_name, class_label in CLASS_MAP.items():
        class_dir = DATASET_ROOT / class_dir_name
        if not class_dir.exists():
            logger.warning(f"  Class directory not found: {class_dir}")
            continue

        sources = FAKE_SOURCES if class_label == "Fake" else REAL_SOURCES
        class_info = {"label": class_label, "sources": {}, "total_files": 0}

        for source in sources:
            source_dir = class_dir / source
            if not source_dir.exists():
                logger.warning(f"  Source directory not found: {source_dir}")
                continue

            files = sorted([
                f for f in source_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            ])
            n_files = len(files)
            class_info["sources"][source] = {
                "path": to_relpath(source_dir),
                "num_files": n_files,
                "extensions": dict(Counter(f.suffix.lower() for f in files)),
                "sample_filenames": [f.name for f in files[:5]]
            }
            class_info["total_files"] += n_files

            # Extract subject/sample IDs
            subject_ids = set()
            for f in files:
                name = f.stem
                # Remove slice suffix to get subject ID
                parts = name.rsplit('_slice', 1)
                if len(parts) == 2:
                    subject_ids.add(parts[0])
                else:
                    subject_ids.add(name)
            class_info["sources"][source]["num_subjects"] = len(subject_ids)
            class_info["sources"][source]["subject_ids_sample"] = sorted(list(subject_ids))[:10]

            structure["sources"][f"{class_label}/{source}"] = {
                "class": class_label,
                "source": source,
                "num_files": n_files,
                "num_subjects": len(subject_ids)
            }

            for f in files:
                all_files.append({
                    'filepath': to_relpath(f),
                    'filename': f.name,
                    'class': class_label,
                    'source': source,
                    'class_dir': class_dir_name,
                    'extension': f.suffix.lower(),
                    'size_bytes': f.stat().st_size
                })

            structure["total_files"] += n_files
            structure["total_dirs"] += 1

        structure["classes"][class_label] = class_info
        logger.info(f"  {class_label}: {class_info['total_files']} files across {len(class_info['sources'])} sources")

    # Save structure report
    report_path = REPORTS_DIR / "dataset_structure_report.json"
    with open(report_path, 'w') as f:
        json.dump(structure, f, indent=2)
    logger.info(f"  Structure report saved to {report_path}")
    logger.info(f"  Total files: {structure['total_files']}, Total source dirs: {structure['total_dirs']}")

    return all_files, structure


# ===========================================================================
# STEP 2: FILE INTEGRITY VERIFICATION
# ===========================================================================
def step2_file_integrity(all_files):
    """Check every image file for corruption or loading errors."""
    logger.info("=" * 70)
    logger.info("STEP 2: File Integrity Verification")
    logger.info("=" * 70)

    corrupted = []
    zero_size = []
    load_errors = []
    valid_count = 0

    for finfo in tqdm(all_files, desc="Checking integrity", unit="file"):
        fp = finfo['filepath']

        # Zero-size check
        if finfo['size_bytes'] == 0:
            zero_size.append({**finfo, 'error': 'Zero-size file'})
            continue

        # Try loading with PIL
        try:
            with Image.open(fp) as img:
                img.verify()
        except Exception as e:
            corrupted.append({**finfo, 'error': f'PIL verify failed: {e}'})
            continue

        # Try loading with OpenCV
        img_cv, err = load_image_safe(fp)
        if img_cv is None:
            load_errors.append({**finfo, 'error': f'OpenCV load failed: {err}'})
            continue

        valid_count += 1

    all_issues = corrupted + zero_size + load_errors
    logger.info(f"  Valid: {valid_count}, Corrupted: {len(corrupted)}, Zero-size: {len(zero_size)}, Load errors: {len(load_errors)}")

    # Save report
    report_path = REPORTS_DIR / "corrupted_files_report.csv"
    if all_issues:
        df = pd.DataFrame(all_issues)
        df.to_csv(report_path, index=False)
    else:
        # Write header-only file
        pd.DataFrame(columns=['filepath', 'filename', 'class', 'source', 'error']).to_csv(report_path, index=False)
    logger.info(f"  Corrupted files report saved to {report_path}")

    return {
        'valid': valid_count,
        'corrupted': len(corrupted),
        'zero_size': len(zero_size),
        'load_errors': len(load_errors),
        'issues': all_issues
    }


# ===========================================================================
# STEP 3: IMAGE STATISTICS COLLECTION
# ===========================================================================
def step3_image_statistics(all_files):
    """Collect per-image statistics: dimensions, channels, intensity stats."""
    logger.info("=" * 70)
    logger.info("STEP 3: Image Statistics Collection")
    logger.info("=" * 70)

    stats_list = []

    for finfo in tqdm(all_files, desc="Collecting stats", unit="file"):
        fp = finfo['filepath']
        try:
            img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            h, w = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1

            # Convert to grayscale for intensity stats
            if channels > 1:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            stats_list.append({
                'filepath': fp,
                'filename': finfo['filename'],
                'class': finfo['class'],
                'source': finfo['source'],
                'width': w,
                'height': h,
                'channels': channels,
                'dtype': str(img.dtype),
                'size_bytes': finfo['size_bytes'],
                'mean_intensity': float(np.mean(gray)),
                'std_intensity': float(np.std(gray)),
                'min_intensity': float(np.min(gray)),
                'max_intensity': float(np.max(gray)),
                'median_intensity': float(np.median(gray)),
                'pct_black_pixels': float(np.mean(gray < 5) * 100),
                'pct_white_pixels': float(np.mean(gray > 250) * 100),
                'aspect_ratio': round(w / h, 4) if h > 0 else 0,
                'total_pixels': w * h
            })
        except Exception as e:
            logger.warning(f"  Error reading {fp}: {e}")

    df = pd.DataFrame(stats_list)
    report_path = REPORTS_DIR / "image_statistics.csv"
    df.to_csv(report_path, index=False)
    logger.info(f"  Image statistics saved to {report_path}")
    logger.info(f"  Processed {len(stats_list)} images successfully")

    # Log summary statistics
    if len(df) > 0:
        logger.info(f"  Resolution range: {df['width'].min()}x{df['height'].min()} to {df['width'].max()}x{df['height'].max()}")
        logger.info(f"  Mean intensity: {df['mean_intensity'].mean():.2f} +/- {df['mean_intensity'].std():.2f}")
        logger.info(f"  Unique resolutions: {df.groupby(['width','height']).ngroups}")

    # ---- Generate Plots ----
    _plot_statistics(df)

    return df


def _plot_statistics(df):
    """Generate statistical distribution plots."""
    if len(df) == 0:
        return

    # 1. Resolution histogram
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Image Statistics Overview", fontsize=16, fontweight='bold')

    # Width distribution
    for cls in df['class'].unique():
        subset = df[df['class'] == cls]
        axes[0, 0].hist(subset['width'], bins=30, alpha=0.6, label=cls)
    axes[0, 0].set_title("Width Distribution")
    axes[0, 0].set_xlabel("Width (px)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()

    # Height distribution
    for cls in df['class'].unique():
        subset = df[df['class'] == cls]
        axes[0, 1].hist(subset['height'], bins=30, alpha=0.6, label=cls)
    axes[0, 1].set_title("Height Distribution")
    axes[0, 1].set_xlabel("Height (px)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()

    # Mean intensity distribution
    for cls in df['class'].unique():
        subset = df[df['class'] == cls]
        axes[1, 0].hist(subset['mean_intensity'], bins=50, alpha=0.6, label=cls)
    axes[1, 0].set_title("Mean Intensity Distribution")
    axes[1, 0].set_xlabel("Mean Intensity")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend()

    # Aspect ratio distribution
    for cls in df['class'].unique():
        subset = df[df['class'] == cls]
        axes[1, 1].hist(subset['aspect_ratio'], bins=30, alpha=0.6, label=cls)
    axes[1, 1].set_title("Aspect Ratio Distribution")
    axes[1, 1].set_xlabel("Aspect Ratio (W/H)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "image_statistics_overview.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Per-source intensity box plot
    fig, ax = plt.subplots(figsize=(14, 6))
    sources_order = sorted(df['source'].unique())
    data_by_source = [df[df['source'] == s]['mean_intensity'].values for s in sources_order]
    bp = ax.boxplot(data_by_source, labels=sources_order, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(sources_order)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title("Mean Intensity by Source", fontsize=14, fontweight='bold')
    ax.set_xlabel("Source")
    ax.set_ylabel("Mean Pixel Intensity")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "intensity_by_source_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. File size distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    for cls in df['class'].unique():
        subset = df[df['class'] == cls]
        ax.hist(subset['size_bytes'] / 1024, bins=50, alpha=0.6, label=cls)
    ax.set_title("File Size Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("File Size (KB)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "file_size_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Resolution scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    color_map = {'Fake': 'red', 'Real': 'blue'}
    for cls in df['class'].unique():
        subset = df[df['class'] == cls]
        ax.scatter(subset['width'], subset['height'], alpha=0.3, label=cls,
                   c=color_map.get(cls, 'gray'), s=10)
    ax.set_title("Image Resolution Scatter", fontsize=14, fontweight='bold')
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "resolution_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("  Plots saved to plots/ directory")


# ===========================================================================
# STEP 4: CLASS BALANCE ANALYSIS
# ===========================================================================
def step4_class_balance(all_files, structure):
    """Analyze class distribution and imbalance."""
    logger.info("=" * 70)
    logger.info("STEP 4: Class Balance Analysis")
    logger.info("=" * 70)

    class_counts = Counter(f['class'] for f in all_files)
    source_counts = Counter(f['source'] for f in all_files)
    class_source_counts = Counter((f['class'], f['source']) for f in all_files)

    # Compute ratios
    total = sum(class_counts.values())
    class_pcts = {k: round(v / total * 100, 2) for k, v in class_counts.items()}

    # Imbalance ratio
    counts = list(class_counts.values())
    imbalance_ratio = round(max(counts) / min(counts), 2) if min(counts) > 0 else float('inf')

    report = {
        "total_images": total,
        "class_counts": dict(class_counts),
        "class_percentages": class_pcts,
        "imbalance_ratio": imbalance_ratio,
        "source_counts": dict(source_counts),
        "class_source_breakdown": {f"{k[0]}/{k[1]}": v for k, v in class_source_counts.items()},
        "subject_counts_per_source": {}
    }

    # Subject counts from structure
    if "sources" in structure:
        for key, info in structure["sources"].items():
            report["subject_counts_per_source"][key] = info.get("num_subjects", 0)

    logger.info(f"  Class counts: {dict(class_counts)}")
    logger.info(f"  Class percentages: {class_pcts}")
    logger.info(f"  Imbalance ratio: {imbalance_ratio}")
    for key, cnt in sorted(source_counts.items()):
        logger.info(f"    Source '{key}': {cnt} files")

    # Save report
    report_path = REPORTS_DIR / "class_distribution.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Class distribution report saved to {report_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Class-level pie chart
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())
    colors_pie = ['#ff6b6b', '#4ecdc4']
    explode = [0.05] * len(labels)
    axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie,
                explode=explode, startangle=90, shadow=True)
    axes[0].set_title("Class Distribution", fontsize=14, fontweight='bold')

    # Source-level bar chart
    src_names = sorted(source_counts.keys())
    src_vals = [source_counts[s] for s in src_names]
    src_colors = []
    for s in src_names:
        for finfo in all_files:
            if finfo['source'] == s:
                src_colors.append('#ff6b6b' if finfo['class'] == 'Fake' else '#4ecdc4')
                break
    bars = axes[1].bar(src_names, src_vals, color=src_colors, edgecolor='black', linewidth=0.5)
    axes[1].set_title("Files per Source", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Source")
    axes[1].set_ylabel("Number of Files")
    axes[1].tick_params(axis='x', rotation=45)
    # Add count labels on bars
    for bar, val in zip(bars, src_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                     str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_distribution_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Class distribution plot saved")

    return report


# ===========================================================================
# STEP 5: DUPLICATE DETECTION
# ===========================================================================
def step5_duplicate_detection(all_files):
    """Detect exact and near-duplicate images."""
    logger.info("=" * 70)
    logger.info("STEP 5: Duplicate Detection")
    logger.info("=" * 70)

    # Phase 1: Exact duplicates via MD5
    logger.info("  Phase 1: Computing MD5 hashes...")
    md5_map = defaultdict(list)
    phash_map = defaultdict(list)

    for finfo in tqdm(all_files, desc="Hashing files", unit="file"):
        fp = finfo['filepath']
        try:
            h = md5_hash(fp)
            md5_map[h].append(finfo)
        except Exception as e:
            logger.warning(f"  MD5 error for {fp}: {e}")

        # Perceptual hash
        if HAS_IMAGEHASH:
            try:
                ph = perceptual_hash(fp)
                if ph:
                    phash_map[ph].append(finfo)
            except Exception as e:
                pass

    # Exact duplicates
    exact_dupes = {h: files for h, files in md5_map.items() if len(files) > 1}

    # Log group sizes and generate a summary instead of all O(n^2) pairs
    exact_dupe_pairs = []
    exact_dupe_set = set()  # For fast lookup
    total_exact_pairs = 0
    MAX_PAIRS_PER_GROUP = 50  # Cap to avoid huge output

    for h, files in exact_dupes.items():
        group_size = len(files)
        n_pairs = group_size * (group_size - 1) // 2
        total_exact_pairs += n_pairs
        logger.info(f"    MD5 group {h[:12]}...: {group_size} identical files, {n_pairs} pairs")

        # Store sample pairs (capped)
        pair_count = 0
        for i in range(len(files)):
            if pair_count >= MAX_PAIRS_PER_GROUP:
                break
            for j in range(i + 1, len(files)):
                if pair_count >= MAX_PAIRS_PER_GROUP:
                    break
                pair_key = tuple(sorted([files[i]['filepath'], files[j]['filepath']]))
                exact_dupe_set.add(pair_key)
                exact_dupe_pairs.append({
                    'type': 'exact_md5',
                    'file1': files[i]['filepath'],
                    'file1_class': files[i]['class'],
                    'file1_source': files[i]['source'],
                    'file2': files[j]['filepath'],
                    'file2_class': files[j]['class'],
                    'file2_source': files[j]['source'],
                    'hash': h,
                    'group_size': group_size,
                    'total_pairs_in_group': n_pairs
                })
                pair_count += 1

    logger.info(f"  Exact duplicate groups: {len(exact_dupes)}")
    logger.info(f"  Total exact duplicate pairs: {total_exact_pairs}")
    logger.info(f"  Sample pairs saved: {len(exact_dupe_pairs)}")

    # Near-duplicates via perceptual hash (using set for fast lookup)
    near_dupe_pairs = []
    if HAS_IMAGEHASH:
        near_dupes = {h: files for h, files in phash_map.items() if len(files) > 1}
        MAX_NEAR_PAIRS = 500  # Cap near-dupe output
        for h, files in near_dupes.items():
            if len(near_dupe_pairs) >= MAX_NEAR_PAIRS:
                break
            for i in range(min(len(files), 20)):  # Cap per group too
                if len(near_dupe_pairs) >= MAX_NEAR_PAIRS:
                    break
                for j in range(i + 1, min(len(files), 20)):
                    if len(near_dupe_pairs) >= MAX_NEAR_PAIRS:
                        break
                    pair_key = tuple(sorted([files[i]['filepath'], files[j]['filepath']]))
                    if pair_key not in exact_dupe_set:
                        near_dupe_pairs.append({
                            'type': 'perceptual_phash',
                            'file1': files[i]['filepath'],
                            'file1_class': files[i]['class'],
                            'file1_source': files[i]['source'],
                            'file2': files[j]['filepath'],
                            'file2_class': files[j]['class'],
                            'file2_source': files[j]['source'],
                            'hash': h
                        })
        logger.info(f"  Near-duplicate groups (pHash): {len(near_dupes)}")
        logger.info(f"  Near-duplicate pairs (excluding exact, sampled): {len(near_dupe_pairs)}")

    all_dupe_pairs = exact_dupe_pairs + near_dupe_pairs

    # Save report
    report_path = REPORTS_DIR / "duplicates_report.csv"
    if all_dupe_pairs:
        df = pd.DataFrame(all_dupe_pairs)
        df.to_csv(report_path, index=False)
    else:
        pd.DataFrame(columns=['type', 'file1', 'file1_class', 'file1_source',
                               'file2', 'file2_class', 'file2_source', 'hash']).to_csv(report_path, index=False)
    logger.info(f"  Duplicates report saved to {report_path}")

    return {
        'exact_duplicate_groups': len(exact_dupes),
        'exact_duplicate_pairs_total': total_exact_pairs,
        'exact_duplicate_pairs_sampled': len(exact_dupe_pairs),
        'near_duplicate_pairs': len(near_dupe_pairs),
        'total_duplicate_pairs_sampled': len(all_dupe_pairs)
    }


# ===========================================================================
# STEP 6: DATA LEAKAGE CHECK
# ===========================================================================
def step6_data_leakage(all_files):
    """Check for data leakage between Real and Fake sources."""
    logger.info("=" * 70)
    logger.info("STEP 6: Data Leakage Check")
    logger.info("=" * 70)

    leakage_issues = []

    # Group files by class
    real_files = [f for f in all_files if f['class'] == 'Real']
    fake_files = [f for f in all_files if f['class'] == 'Fake']

    # Check 1: Cross-class exact MD5 matches
    logger.info("  Check 1: Cross-class MD5 overlap...")
    real_hashes = {}
    for f in tqdm(real_files, desc="Hashing real files", unit="file"):
        try:
            h = md5_hash(f['filepath'])
            real_hashes[h] = f
        except:
            pass

    cross_md5_matches = 0
    for f in tqdm(fake_files, desc="Checking fake vs real", unit="file"):
        try:
            h = md5_hash(f['filepath'])
            if h in real_hashes:
                cross_md5_matches += 1
                real_match = real_hashes[h]
                leakage_issues.append({
                    'check': 'cross_class_md5',
                    'severity': 'CRITICAL',
                    'real_file': real_match['filepath'],
                    'real_source': real_match['source'],
                    'fake_file': f['filepath'],
                    'fake_source': f['source'],
                    'detail': 'Exact MD5 match between real and fake image'
                })
        except:
            pass
    logger.info(f"    Cross-class MD5 matches: {cross_md5_matches}")

    # Check 2: Filename-based subject overlap
    logger.info("  Check 2: Subject ID overlap between matched real/fake sources...")
    source_pairs = [
        ('cermep', 'MLS_CERMEP'),
        ('tcga', 'MLS_TCGA'),
        ('upenn', 'MLS_UPenn'),
    ]

    for real_src, fake_src in source_pairs:
        real_src_files = [f for f in real_files if f['source'] == real_src]
        fake_src_files = [f for f in fake_files if f['source'] == fake_src]

        # Extract subject IDs from real files
        real_subjects = set()
        for f in real_src_files:
            parts = f['filename'].rsplit('_slice', 1)
            if len(parts) == 2:
                real_subjects.add(parts[0])

        # Note: MLS fake files use generic Training_N naming
        # so no direct subject-ID overlap. But we should check if the
        # number of fake subjects matches real subjects (potential 1:1 mapping)
        fake_subjects = set()
        for f in fake_src_files:
            parts = f['filename'].rsplit('_slice', 1)
            if len(parts) == 2:
                fake_subjects.add(parts[0])

        logger.info(f"    {real_src} vs {fake_src}: {len(real_subjects)} real subjects, {len(fake_subjects)} fake subjects")

        # If counts match exactly, this suggests a 1:1 mapping
        if len(real_subjects) == len(fake_subjects):
            leakage_issues.append({
                'check': 'subject_count_match',
                'severity': 'WARNING',
                'real_file': f'{real_src} ({len(real_subjects)} subjects)',
                'real_source': real_src,
                'fake_file': f'{fake_src} ({len(fake_subjects)} subjects)',
                'fake_source': fake_src,
                'detail': f'Exact subject count match ({len(real_subjects)}). '
                          f'Possible 1:1 real-to-fake mapping. '
                          f'Ensure no data leakage in train/test splits.'
            })

    # Check 3: Cross-class perceptual similarity (sample based)
    logger.info("  Check 3: Cross-class perceptual hash overlap (sampled)...")
    if HAS_IMAGEHASH:
        # Hash all real files
        real_phashes = {}
        for f in tqdm(real_files, desc="pHash real", unit="file"):
            ph = perceptual_hash(f['filepath'])
            if ph:
                real_phashes[ph] = f

        phash_cross_matches = 0
        for f in tqdm(fake_files, desc="pHash fake vs real", unit="file"):
            ph = perceptual_hash(f['filepath'])
            if ph and ph in real_phashes:
                phash_cross_matches += 1
                real_match = real_phashes[ph]
                leakage_issues.append({
                    'check': 'cross_class_phash',
                    'severity': 'HIGH',
                    'real_file': real_match['filepath'],
                    'real_source': real_match['source'],
                    'fake_file': f['filepath'],
                    'fake_source': f['source'],
                    'detail': f'Perceptual hash match (pHash={ph})'
                })
        logger.info(f"    Cross-class pHash matches: {phash_cross_matches}")

    # Check 4: No explicit train/test/val split
    has_split = False
    for split_name in ['train', 'test', 'val', 'validation']:
        if (DATASET_ROOT / split_name).exists():
            has_split = True
            break
    if not has_split:
        leakage_issues.append({
            'check': 'no_split_detected',
            'severity': 'INFO',
            'real_file': 'N/A',
            'real_source': 'N/A',
            'fake_file': 'N/A',
            'fake_source': 'N/A',
            'detail': 'No train/test/val directory split found. '
                      'If splitting is done at runtime, ensure subject-level splitting '
                      'to prevent data leakage (same subject slices in both train and test).'
        })

    logger.info(f"  Total leakage issues: {len(leakage_issues)}")

    # Save report
    report_path = REPORTS_DIR / "data_leakage_report.csv"
    if leakage_issues:
        df = pd.DataFrame(leakage_issues)
        df.to_csv(report_path, index=False)
    else:
        pd.DataFrame(columns=['check', 'severity', 'real_file', 'real_source',
                               'fake_file', 'fake_source', 'detail']).to_csv(report_path, index=False)
    logger.info(f"  Data leakage report saved to {report_path}")

    return leakage_issues


# ===========================================================================
# STEP 7: VISUAL INSPECTION SAMPLES
# ===========================================================================
def step7_visual_samples(all_files):
    """Generate sample grids for visual inspection."""
    logger.info("=" * 70)
    logger.info("STEP 7: Visual Inspection Sample Grids")
    logger.info("=" * 70)

    for class_label in ['Real', 'Fake']:
        class_files = [f for f in all_files if f['class'] == class_label]
        sources = sorted(set(f['source'] for f in class_files))

        n_sources = len(sources)
        n_samples_per_source = 5

        fig, axes = plt.subplots(n_sources, n_samples_per_source,
                                  figsize=(3 * n_samples_per_source, 3 * n_sources))
        if n_sources == 1:
            axes = [axes]

        fig.suptitle(f"{class_label} Images - Sample Grid", fontsize=16, fontweight='bold', y=1.02)

        for row, source in enumerate(sources):
            src_files = [f for f in class_files if f['source'] == source]
            # Sample evenly spaced
            indices = np.linspace(0, len(src_files) - 1, n_samples_per_source, dtype=int)
            for col, idx in enumerate(indices):
                fp = src_files[idx]['filepath']
                try:
                    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        ax = axes[row][col] if n_sources > 1 else axes[col]
                        ax.imshow(img, cmap='gray')
                        ax.set_title(f"{source}\n{Path(fp).name[:30]}...", fontsize=7)
                    else:
                        ax = axes[row][col] if n_sources > 1 else axes[col]
                        ax.text(0.5, 0.5, "Load Error", ha='center', va='center')
                except Exception as e:
                    ax = axes[row][col] if n_sources > 1 else axes[col]
                    ax.text(0.5, 0.5, "Error", ha='center', va='center')

                ax = axes[row][col] if n_sources > 1 else axes[col]
                ax.axis('off')

        plt.tight_layout()
        save_path = SAMPLES_DIR / f"sample_{class_label.lower()}_grid.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  {class_label} sample grid saved to {save_path}")


# ===========================================================================
# STEP 8: MRI-SPECIFIC CHECKS
# ===========================================================================
def step8_mri_checks(all_files, stats_df):
    """MRI-specific quality checks: blank images, intensity anomalies."""
    logger.info("=" * 70)
    logger.info("STEP 8: MRI-Specific Quality Checks")
    logger.info("=" * 70)

    anomalies = []

    if stats_df is None or len(stats_df) == 0:
        logger.warning("  No stats available, skipping MRI checks")
        return anomalies

    # Check 1: Nearly blank images (very low mean intensity)
    threshold_blank = 5.0
    blank_mask = stats_df['mean_intensity'] < threshold_blank
    blank_images = stats_df[blank_mask]
    for _, row in blank_images.iterrows():
        anomalies.append({
            'check': 'nearly_blank',
            'filepath': row['filepath'],
            'filename': row['filename'],
            'class': row['class'],
            'source': row['source'],
            'severity': 'HIGH',
            'detail': f"Mean intensity={row['mean_intensity']:.2f} (threshold={threshold_blank})"
        })
    logger.info(f"  Nearly blank images (mean < {threshold_blank}): {len(blank_images)}")

    # Check 2: Saturated images (very high mean intensity)
    threshold_saturated = 240.0
    sat_mask = stats_df['mean_intensity'] > threshold_saturated
    sat_images = stats_df[sat_mask]
    for _, row in sat_images.iterrows():
        anomalies.append({
            'check': 'saturated',
            'filepath': row['filepath'],
            'filename': row['filename'],
            'class': row['class'],
            'source': row['source'],
            'severity': 'HIGH',
            'detail': f"Mean intensity={row['mean_intensity']:.2f} (threshold={threshold_saturated})"
        })
    logger.info(f"  Saturated images (mean > {threshold_saturated}): {len(sat_images)}")

    # Check 3: Extremely low variance (uniform images)
    threshold_std = 2.0
    low_var_mask = stats_df['std_intensity'] < threshold_std
    low_var = stats_df[low_var_mask]
    for _, row in low_var.iterrows():
        anomalies.append({
            'check': 'low_variance',
            'filepath': row['filepath'],
            'filename': row['filename'],
            'class': row['class'],
            'source': row['source'],
            'severity': 'MEDIUM',
            'detail': f"Std intensity={row['std_intensity']:.2f} (threshold={threshold_std})"
        })
    logger.info(f"  Low variance images (std < {threshold_std}): {len(low_var)}")

    # Check 4: High percentage of black pixels (>90%)
    high_black = stats_df[stats_df['pct_black_pixels'] > 90]
    for _, row in high_black.iterrows():
        anomalies.append({
            'check': 'mostly_black',
            'filepath': row['filepath'],
            'filename': row['filename'],
            'class': row['class'],
            'source': row['source'],
            'severity': 'MEDIUM',
            'detail': f"Black pixels: {row['pct_black_pixels']:.1f}%"
        })
    logger.info(f"  Mostly black images (>90% black): {len(high_black)}")

    # Check 5: Intensity outliers per source (Z-score > 3)
    for source in stats_df['source'].unique():
        src_df = stats_df[stats_df['source'] == source]
        if len(src_df) < 10:
            continue
        mean_val = src_df['mean_intensity'].mean()
        std_val = src_df['mean_intensity'].std()
        if std_val > 0:
            z_scores = (src_df['mean_intensity'] - mean_val) / std_val
            outlier_mask = z_scores.abs() > 3
            outliers = src_df[outlier_mask]
            for _, row in outliers.iterrows():
                z = (row['mean_intensity'] - mean_val) / std_val
                anomalies.append({
                    'check': 'intensity_outlier',
                    'filepath': row['filepath'],
                    'filename': row['filename'],
                    'class': row['class'],
                    'source': row['source'],
                    'severity': 'LOW',
                    'detail': f"Z-score={z:.2f}, mean_intensity={row['mean_intensity']:.2f} "
                              f"(source mean={mean_val:.2f}, std={std_val:.2f})"
                })
            if len(outliers) > 0:
                logger.info(f"  Intensity outliers in {source}: {len(outliers)}")

    # Check 6: Resolution mismatches within source
    for source in stats_df['source'].unique():
        src_df = stats_df[stats_df['source'] == source]
        unique_res = src_df.groupby(['width', 'height']).size()
        if len(unique_res) > 1:
            dominant_res = unique_res.idxmax()
            non_dominant = src_df[
                (src_df['width'] != dominant_res[0]) | (src_df['height'] != dominant_res[1])
            ]
            for _, row in non_dominant.iterrows():
                anomalies.append({
                    'check': 'resolution_mismatch',
                    'filepath': row['filepath'],
                    'filename': row['filename'],
                    'class': row['class'],
                    'source': row['source'],
                    'severity': 'MEDIUM',
                    'detail': f"Resolution {row['width']}x{row['height']} differs from "
                              f"dominant {dominant_res[0]}x{dominant_res[1]} in source {source}"
                })
            logger.info(f"  Resolution mismatches in {source}: {len(non_dominant)} "
                       f"(dominant: {dominant_res[0]}x{dominant_res[1]})")

    # Save report
    report_path = REPORTS_DIR / "mri_anomaly_report.csv"
    if anomalies:
        df = pd.DataFrame(anomalies)
        df.to_csv(report_path, index=False)
    else:
        pd.DataFrame(columns=['check', 'filepath', 'filename', 'class', 'source',
                               'severity', 'detail']).to_csv(report_path, index=False)
    logger.info(f"  MRI anomaly report saved to {report_path} ({len(anomalies)} issues)")

    return anomalies


# ===========================================================================
# STEP 9: SYNTHETIC ARTIFACT ANALYSIS (FFT)
# ===========================================================================
def step9_fft_artifact_analysis(all_files):
    """FFT-based artifact detection for synthetic images."""
    logger.info("=" * 70)
    logger.info("STEP 9: Synthetic Artifact Analysis (FFT)")
    logger.info("=" * 70)

    # We analyze FFT spectra for both real and fake to find distinguishing patterns
    fft_results = {'Real': {}, 'Fake': {}}

    # Sample images for FFT analysis (up to 50 per source)
    max_samples = 50

    for class_label in ['Real', 'Fake']:
        class_files = [f for f in all_files if f['class'] == class_label]
        sources = sorted(set(f['source'] for f in class_files))

        for source in sources:
            src_files = [f for f in class_files if f['source'] == source]
            sample_indices = np.linspace(0, len(src_files) - 1,
                                          min(max_samples, len(src_files)), dtype=int)
            sampled = [src_files[i] for i in sample_indices]

            azimuthal_profiles = []
            spectral_energies = []

            for finfo in tqdm(sampled, desc=f"FFT {class_label}/{source}", unit="img"):
                try:
                    img = cv2.imread(finfo['filepath'], cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    # Compute 2D FFT
                    f_transform = np.fft.fft2(img.astype(np.float64))
                    f_shift = np.fft.fftshift(f_transform)
                    magnitude = np.log1p(np.abs(f_shift))

                    # Azimuthally averaged power spectrum
                    cy, cx = magnitude.shape[0] // 2, magnitude.shape[1] // 2
                    Y, X = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
                    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
                    max_r = min(cx, cy)
                    radial_profile = np.zeros(max_r)
                    for r in range(max_r):
                        mask = R == r
                        if mask.any():
                            radial_profile[r] = magnitude[mask].mean()
                    azimuthal_profiles.append(radial_profile)

                    # Total spectral energy in different bands
                    low = magnitude[R < max_r * 0.25].mean() if (R < max_r * 0.25).any() else 0
                    mid = magnitude[(R >= max_r * 0.25) & (R < max_r * 0.5)].mean() if ((R >= max_r * 0.25) & (R < max_r * 0.5)).any() else 0
                    high = magnitude[R >= max_r * 0.5].mean() if (R >= max_r * 0.5).any() else 0
                    spectral_energies.append({'low_freq': low, 'mid_freq': mid, 'high_freq': high})

                except Exception as e:
                    logger.debug(f"  FFT error for {finfo['filepath']}: {e}")

            if azimuthal_profiles:
                # Truncate to common length
                min_len = min(len(p) for p in azimuthal_profiles)
                profiles_arr = np.array([p[:min_len] for p in azimuthal_profiles])
                mean_profile = profiles_arr.mean(axis=0)
                std_profile = profiles_arr.std(axis=0)

                fft_results[class_label][source] = {
                    'mean_radial_profile': mean_profile.tolist(),
                    'std_radial_profile': std_profile.tolist(),
                    'n_samples': len(azimuthal_profiles),
                    'spectral_energy': {
                        'low_freq_mean': float(np.mean([s['low_freq'] for s in spectral_energies])),
                        'mid_freq_mean': float(np.mean([s['mid_freq'] for s in spectral_energies])),
                        'high_freq_mean': float(np.mean([s['high_freq'] for s in spectral_energies])),
                    }
                }
                logger.info(f"  {class_label}/{source}: {len(azimuthal_profiles)} FFT profiles computed")

    # ---- Plot FFT comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Mean radial profiles by source
    ax = axes[0]
    color_cycle = plt.cm.tab10(np.linspace(0, 1, 10))
    ci = 0
    for class_label in ['Real', 'Fake']:
        for source, data in fft_results[class_label].items():
            profile = np.array(data['mean_radial_profile'])
            linestyle = '-' if class_label == 'Real' else '--'
            ax.plot(profile, label=f"{class_label}/{source}", linestyle=linestyle,
                    color=color_cycle[ci % 10], alpha=0.8)
            ci += 1
    ax.set_title("Mean Radial FFT Profile by Source", fontsize=13, fontweight='bold')
    ax.set_xlabel("Spatial Frequency (radius)")
    ax.set_ylabel("Log Magnitude")
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Spectral energy comparison (bar chart)
    ax = axes[1]
    sources_list = []
    low_vals, mid_vals, high_vals = [], [], []
    colors_bar = []
    for class_label in ['Real', 'Fake']:
        for source, data in fft_results[class_label].items():
            sources_list.append(f"{class_label}/{source}")
            se = data['spectral_energy']
            low_vals.append(se['low_freq_mean'])
            mid_vals.append(se['mid_freq_mean'])
            high_vals.append(se['high_freq_mean'])
            colors_bar.append('#4ecdc4' if class_label == 'Real' else '#ff6b6b')

    x = np.arange(len(sources_list))
    width = 0.25
    ax.bar(x - width, low_vals, width, label='Low Freq', alpha=0.8)
    ax.bar(x, mid_vals, width, label='Mid Freq', alpha=0.8)
    ax.bar(x + width, high_vals, width, label='High Freq', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sources_list, rotation=45, ha='right', fontsize=8)
    ax.set_title("Spectral Energy by Frequency Band", fontsize=13, fontweight='bold')
    ax.set_ylabel("Mean Log Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fft_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  FFT analysis plot saved")

    # Save a sample FFT magnitude spectrum comparison
    _plot_fft_sample_comparison(all_files)

    return fft_results


def _plot_fft_sample_comparison(all_files):
    """Plot side-by-side FFT magnitude spectra for one real and one fake image per source pair."""
    source_pairs = [
        ('cermep', 'MLS_CERMEP', 'GAN'),
        ('tcga', 'MLS_TCGA', 'LDM'),
        ('upenn', 'MLS_UPenn', None),
    ]

    n_rows = len(source_pairs)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    fig.suptitle("FFT Magnitude Spectrum: Real vs Fake", fontsize=16, fontweight='bold')

    for row, (real_src, fake_src, extra_fake) in enumerate(source_pairs):
        # Get a real sample
        real_files = [f for f in all_files if f['source'] == real_src]
        fake_files = [f for f in all_files if f['source'] == fake_src]

        pairs = [(real_files, real_src, 0), (fake_files, fake_src, 2)]

        for files, src_name, col_offset in pairs:
            if files:
                mid_idx = len(files) // 2
                fp = files[mid_idx]['filepath']
                try:
                    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Original
                        axes[row][col_offset].imshow(img, cmap='gray')
                        axes[row][col_offset].set_title(f"{src_name}\n(Original)", fontsize=9)
                        axes[row][col_offset].axis('off')

                        # FFT
                        f_transform = np.fft.fft2(img.astype(np.float64))
                        f_shift = np.fft.fftshift(f_transform)
                        magnitude = np.log1p(np.abs(f_shift))
                        axes[row][col_offset + 1].imshow(magnitude, cmap='hot')
                        axes[row][col_offset + 1].set_title(f"{src_name}\n(FFT Magnitude)", fontsize=9)
                        axes[row][col_offset + 1].axis('off')
                except Exception:
                    pass

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fft_sample_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  FFT sample comparison plot saved")


# ===========================================================================
# STEP 10: FINAL SUMMARY REPORT
# ===========================================================================
def step10_final_report(structure, integrity, stats_df, class_report,
                         duplicates, leakage, anomalies, fft_results):
    """Generate the final comprehensive audit report."""
    logger.info("=" * 70)
    logger.info("STEP 10: Final Summary Report")
    logger.info("=" * 70)

    # ---- JSON Summary ----
    summary = {
        "audit_timestamp": datetime.now().isoformat(),
        "dataset_root": to_relpath(DATASET_ROOT),
        "total_files": structure.get('total_files', 0),
        "total_sources": structure.get('total_dirs', 0),
        "integrity": {
            "valid_files": integrity['valid'],
            "corrupted": integrity['corrupted'],
            "zero_size": integrity['zero_size'],
            "load_errors": integrity['load_errors'],
        },
        "image_stats": {},
        "class_balance": {
            "class_counts": class_report.get('class_counts', {}),
            "imbalance_ratio": class_report.get('imbalance_ratio', 0),
        },
        "duplicates": duplicates,
        "leakage_issues_count": len(leakage),
        "mri_anomalies_count": len(anomalies),
        "fft_sources_analyzed": sum(len(v) for v in fft_results.values()),
    }

    if stats_df is not None and len(stats_df) > 0:
        summary["image_stats"] = {
            "total_analyzed": len(stats_df),
            "resolution_range": {
                "min_width": int(stats_df['width'].min()),
                "max_width": int(stats_df['width'].max()),
                "min_height": int(stats_df['height'].min()),
                "max_height": int(stats_df['height'].max()),
            },
            "unique_resolutions": int(stats_df.groupby(['width', 'height']).ngroups),
            "mean_intensity_overall": round(float(stats_df['mean_intensity'].mean()), 2),
            "std_intensity_overall": round(float(stats_df['mean_intensity'].std()), 2),
            "channels": dict(stats_df['channels'].value_counts()),
            "dtypes": dict(stats_df['dtype'].value_counts()),
        }

    summary_path = REPORTS_DIR / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  JSON summary saved to {summary_path}")

    # ---- Markdown Report ----
    md_lines = []
    md_lines.append("# MRI Dataset Audit Report")
    md_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"\n**Dataset Root:** `{to_relpath(DATASET_ROOT)}`\n")

    # Executive Summary
    md_lines.append("## Executive Summary\n")
    md_lines.append(f"| Metric | Value |")
    md_lines.append(f"|--------|-------|")
    md_lines.append(f"| Total Images | {structure.get('total_files', 0)} |")
    md_lines.append(f"| Total Sources | {structure.get('total_dirs', 0)} |")
    md_lines.append(f"| Valid Files | {integrity['valid']} |")
    md_lines.append(f"| Corrupted/Error Files | {integrity['corrupted'] + integrity['zero_size'] + integrity['load_errors']} |")
    md_lines.append(f"| Class Imbalance Ratio | {class_report.get('imbalance_ratio', 'N/A')} |")
    md_lines.append(f"| Exact Duplicate Pairs | {duplicates['exact_duplicate_pairs_total']} |")
    md_lines.append(f"| Near-Duplicate Pairs | {duplicates['near_duplicate_pairs']} |")
    md_lines.append(f"| Leakage Issues | {len(leakage)} |")
    md_lines.append(f"| MRI Anomalies | {len(anomalies)} |")
    md_lines.append("")

    # Overall Health Assessment
    issues_total = (integrity['corrupted'] + integrity['zero_size'] + integrity['load_errors']
                    + duplicates['exact_duplicate_pairs_total'] + len([l for l in leakage if l.get('severity') == 'CRITICAL']))
    if issues_total == 0:
        health = "HEALTHY"
        health_desc = "No critical issues detected."
    elif issues_total < 10:
        health = "MINOR ISSUES"
        health_desc = "A few issues found. Review recommended."
    else:
        health = "NEEDS ATTENTION"
        health_desc = "Multiple issues detected. Detailed review required."

    md_lines.append(f"### Overall Health: **{health}**")
    md_lines.append(f"{health_desc}\n")

    # Step 1: Structure
    md_lines.append("---\n## 1. Dataset Structure\n")
    md_lines.append("| Class | Source | Files | Subjects |")
    md_lines.append("|-------|--------|-------|----------|")
    if "sources" in structure:
        for key, info in sorted(structure["sources"].items()):
            md_lines.append(f"| {info['class']} | {info['source']} | {info['num_files']} | {info['num_subjects']} |")
    md_lines.append("")

    # Step 2: Integrity
    md_lines.append("---\n## 2. File Integrity\n")
    md_lines.append(f"- **Valid:** {integrity['valid']}")
    md_lines.append(f"- **Corrupted (PIL):** {integrity['corrupted']}")
    md_lines.append(f"- **Zero-size:** {integrity['zero_size']}")
    md_lines.append(f"- **Load errors (OpenCV):** {integrity['load_errors']}")
    if integrity['issues']:
        md_lines.append(f"\n**Issue details:** See `corrupted_files_report.csv`\n")
    else:
        md_lines.append(f"\nAll files passed integrity checks.\n")

    # Step 3: Image Statistics
    md_lines.append("---\n## 3. Image Statistics\n")
    if stats_df is not None and len(stats_df) > 0:
        md_lines.append(f"- **Images analyzed:** {len(stats_df)}")
        md_lines.append(f"- **Resolution range:** {stats_df['width'].min()}x{stats_df['height'].min()} "
                       f"to {stats_df['width'].max()}x{stats_df['height'].max()}")
        md_lines.append(f"- **Unique resolutions:** {stats_df.groupby(['width', 'height']).ngroups}")
        md_lines.append(f"- **Mean intensity:** {stats_df['mean_intensity'].mean():.2f} ± {stats_df['mean_intensity'].std():.2f}")
        md_lines.append(f"- **Channel counts:** {dict(stats_df['channels'].value_counts())}")

        # Per-source stats table
        md_lines.append("\n### Per-Source Statistics\n")
        md_lines.append("| Source | Count | Width (mean) | Height (mean) | Mean Intensity | Std Intensity |")
        md_lines.append("|--------|-------|-------------|--------------|----------------|---------------|")
        for source in sorted(stats_df['source'].unique()):
            sdf = stats_df[stats_df['source'] == source]
            md_lines.append(
                f"| {source} | {len(sdf)} | {sdf['width'].mean():.0f} | {sdf['height'].mean():.0f} | "
                f"{sdf['mean_intensity'].mean():.2f} | {sdf['std_intensity'].mean():.2f} |"
            )
        md_lines.append(f"\nSee `image_statistics.csv` for full details.\n")
        md_lines.append(f"\nPlots: `image_statistics_overview.png`, `intensity_by_source_boxplot.png`, "
                       f"`file_size_distribution.png`, `resolution_scatter.png`\n")

    # Step 4: Class Balance
    md_lines.append("---\n## 4. Class Balance\n")
    md_lines.append(f"| Class | Count | Percentage |")
    md_lines.append(f"|-------|-------|------------|")
    for cls, cnt in class_report.get('class_counts', {}).items():
        pct = class_report.get('class_percentages', {}).get(cls, 0)
        md_lines.append(f"| {cls} | {cnt} | {pct}% |")
    md_lines.append(f"\n**Imbalance Ratio:** {class_report.get('imbalance_ratio', 'N/A')}")
    if class_report.get('imbalance_ratio', 1) > 2:
        md_lines.append("\n> ⚠️ **Warning:** Significant class imbalance detected. "
                       "Consider oversampling the minority class or using weighted loss.\n")

    # Step 5: Duplicates
    md_lines.append("---\n## 5. Duplicate Detection\n")
    md_lines.append(f"- **Exact duplicate groups (MD5):** {duplicates['exact_duplicate_groups']}")
    md_lines.append(f"- **Exact duplicate pairs (total):** {duplicates['exact_duplicate_pairs_total']}")
    md_lines.append(f"- **Near-duplicate pairs (pHash, sampled):** {duplicates['near_duplicate_pairs']}")
    if duplicates['total_duplicate_pairs_sampled'] > 0:
        md_lines.append(f"\nSee `duplicates_report.csv` for details.\n")
    else:
        md_lines.append(f"\nNo duplicates detected.\n")

    # Step 6: Data Leakage
    md_lines.append("---\n## 6. Data Leakage Check\n")
    if leakage:
        severity_counts = Counter(l.get('severity', 'INFO') for l in leakage)
        md_lines.append(f"| Severity | Count |")
        md_lines.append(f"|----------|-------|")
        for sev in ['CRITICAL', 'HIGH', 'WARNING', 'INFO']:
            if sev in severity_counts:
                md_lines.append(f"| {sev} | {severity_counts[sev]} |")
        md_lines.append("")
        for issue in leakage:
            emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "WARNING": "🟡", "INFO": "ℹ️"}.get(issue['severity'], "")
            md_lines.append(f"- {emoji} **[{issue['severity']}]** {issue['check']}: {issue['detail']}")
        md_lines.append(f"\nSee `data_leakage_report.csv` for details.\n")
    else:
        md_lines.append("No leakage issues detected.\n")

    # Step 7: Visual Samples
    md_lines.append("---\n## 7. Visual Inspection\n")
    md_lines.append("Sample grids generated:")
    md_lines.append("- `samples/sample_real_grid.png`")
    md_lines.append("- `samples/sample_fake_grid.png`\n")

    # Step 8: MRI Anomalies
    md_lines.append("---\n## 8. MRI-Specific Anomalies\n")
    if anomalies:
        anomaly_counts = Counter(a['check'] for a in anomalies)
        md_lines.append(f"| Check | Count |")
        md_lines.append(f"|-------|-------|")
        for check, cnt in sorted(anomaly_counts.items()):
            md_lines.append(f"| {check} | {cnt} |")
        md_lines.append(f"\nSee `mri_anomaly_report.csv` for details.\n")
    else:
        md_lines.append("No MRI-specific anomalies detected.\n")

    # Step 9: FFT Analysis
    md_lines.append("---\n## 9. Synthetic Artifact Analysis (FFT)\n")
    md_lines.append("FFT-based frequency analysis was performed to detect synthetic artifacts.\n")
    for class_label in ['Real', 'Fake']:
        for source, data in fft_results.get(class_label, {}).items():
            se = data.get('spectral_energy', {})
            md_lines.append(f"- **{class_label}/{source}** ({data['n_samples']} samples): "
                           f"Low={se.get('low_freq_mean', 0):.2f}, "
                           f"Mid={se.get('mid_freq_mean', 0):.2f}, "
                           f"High={se.get('high_freq_mean', 0):.2f}")
    md_lines.append(f"\nPlots: `fft_analysis.png`, `fft_sample_comparison.png`\n")

    # Recommendations
    md_lines.append("---\n## 10. Recommendations\n")
    recommendations = []
    if class_report.get('imbalance_ratio', 1) > 2:
        recommendations.append("**Class Imbalance:** Apply oversampling (e.g., SMOTE), undersampling, "
                              "or use weighted loss functions to address the class imbalance.")
    if duplicates['exact_duplicate_pairs_total'] > 0:
        recommendations.append("**Duplicates:** Remove exact duplicate images to prevent data leakage and inflated metrics.")
    if any(l.get('severity') == 'CRITICAL' for l in leakage):
        recommendations.append("**Critical Leakage:** Identical images found in both Real and Fake sets. "
                              "These MUST be removed before training.")
    if any(l.get('check') == 'no_split_detected' for l in leakage):
        recommendations.append("**Train/Test Split:** No pre-defined split found. Implement **subject-level splitting** "
                              "to ensure all slices from the same subject are in the same split.")
    if any(a['check'] == 'nearly_blank' for a in anomalies):
        recommendations.append("**Blank Images:** Remove or review nearly-blank images that may harm training.")
    if any(a['check'] == 'resolution_mismatch' for a in anomalies):
        recommendations.append("**Resolution Consistency:** Standardize image resolutions within each source. "
                              "Apply consistent preprocessing (resize/crop).")

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            md_lines.append(f"{i}. {rec}")
    else:
        md_lines.append("No critical recommendations. Dataset appears ready for use.")

    md_lines.append(f"\n---\n*Report generated by MRI Dataset Audit Script*\n")

    # Write markdown report
    report_path = REPORTS_DIR / "dataset_audit_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    logger.info(f"  Markdown report saved to {report_path}")

    return summary


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================
def main():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("MRI DATASET COMPREHENSIVE AUDIT")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Dataset: {DATASET_ROOT}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("=" * 70)

    # Verify dataset exists
    if not DATASET_ROOT.exists():
        logger.error(f"Dataset root not found: {DATASET_ROOT}")
        sys.exit(1)

    # Step 1
    all_files, structure = step1_dataset_structure()
    if not all_files:
        logger.error("No files found in dataset!")
        sys.exit(1)

    # Step 2
    integrity = step2_file_integrity(all_files)

    # Step 3
    stats_df = step3_image_statistics(all_files)

    # Step 4
    class_report = step4_class_balance(all_files, structure)

    # Step 5
    duplicates = step5_duplicate_detection(all_files)

    # Step 6
    leakage = step6_data_leakage(all_files)

    # Step 7
    step7_visual_samples(all_files)

    # Step 8
    anomalies = step8_mri_checks(all_files, stats_df)

    # Step 9
    fft_results = step9_fft_artifact_analysis(all_files)

    # Step 10
    summary = step10_final_report(structure, integrity, stats_df, class_report,
                                   duplicates, leakage, anomalies, fft_results)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"AUDIT COMPLETE in {elapsed:.1f} seconds")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)

    return summary


if __name__ == "__main__":
    main()
