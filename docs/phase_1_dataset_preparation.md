# Phase 1: Dataset Preparation

> **Status**: Complete | **Dataset**: RGIIIT | **Output**: `RGIIIT_clean/`

## Overview

Phase 1 covers auditing, cleaning, and splitting the RGIIIT MRI brain image dataset into a reproducible train/val/test structure suitable for model training. The goal is to remove noise, eliminate shortcuts, and ensure clean evaluation.

## Initial Dataset

The raw RGIIIT dataset contains **6,735 grayscale MRI brain slice images** across 8 sources:

| Class | Sources | Images | Subjects |
|-------|---------|--------|----------|
| Real | cermep, tcga, upenn | 2,025 | 405 |
| Fake | GAN, LDM, MLS_CERMEP, MLS_TCGA, MLS_UPenn | 4,710 | 942 |

Each subject has 5 axial slices extracted at fixed intervals from a 3D volume.

### Key Issues Found

| Issue | Impact |
|-------|--------|
| **1,607 blank boundary slices** | Non-informative; inflate dataset; create false duplicates |
| **4 MD5 duplicate groups** (124K pairs) | All concentrated in blank images |
| **7 cross-class pHash matches** | GAN images perceptually identical to real cermep images |
| **4 different resolutions** | Resolution acts as class shortcut (e.g., 176×144 → always LDM/Fake) |
| **2.33:1 class imbalance** | Fake class overrepresented |

## Preparation Pipeline

```
┌─────────────────┐
│  Dataset Audit   │  scripts/audit/run_dataset_audit.py
│  (10 checks)     │  → dataset_audit/reports/
└────────┬────────┘
         ▼
┌─────────────────┐
│  Cleaning &      │  scripts/cleaning/clean_and_split_dataset.py
│  Splitting       │  → RGIIIT_clean/ + dataset_cleaning/
└────────┬────────┘
         ▼
┌─────────────────┐
│  Validation      │  scripts/validation/validate_clean_dataset.py
│  (10 checks)     │  → All checks PASS
└─────────────────┘
```

### Cleaning Steps

1. **Blank removal** — Images with mean intensity < 5, >95% black pixels, or std < 2 are removed (1,607 images)
2. **MD5 deduplication** — Keep 1 per group (0 remaining after blank removal)
3. **pHash quarantine** — 7 GAN images with cross-class perceptual hash matches excluded
4. **Subject-level split** — 70/15/15 stratified by (class, source), ensuring all slices of a subject stay in the same split
5. **Resize** — All images resized to 224×224 using Lanczos interpolation, saved as PNG

## How to Reproduce

> **Prerequisite**: Place the raw `RGIIIT/` folder in the project root.

```bash
# 1. Audit (optional — reports already committed)
python scripts/audit/run_dataset_audit.py

# 2. Clean, split, and resize
python scripts/cleaning/clean_and_split_dataset.py

# 3. Validate
python scripts/validation/validate_clean_dataset.py
```

**Requirements**: Python 3.11+, PIL, OpenCV, numpy, pandas, matplotlib, imagehash, tqdm

The cleaning pipeline uses `seed=42` for deterministic subject-level splitting.

## Final Dataset Structure

```
RGIIIT_clean/
├── train/
│   ├── Fake/    (2,731 images)
│   └── Real/    (856 images)
├── val/
│   ├── Fake/    (584 images)
│   └── Real/    (185 images)
└── test/
    ├── Fake/    (580 images)
    └── Real/    (185 images)
```

All images: **224×224 grayscale PNG**, uint8.

Filenames follow the pattern `{source}__{original_stem}.png` for provenance tracking.

## Dataset Statistics Summary

| Metric | Before | After |
|--------|--------|-------|
| Total images | 6,735 | 5,121 |
| Removed | — | 1,614 (24.0%) |
| Resolutions | 4 | 1 (224×224) |
| MD5 duplicates | 124,245 pairs | 0 |
| Cross-class pHash | 7 genuine | 0 |
| Blank images | ~1,600 | 0 |

| Split | Fake | Real | Total | % |
|-------|------|------|-------|---|
| Train | 2,731 | 856 | 3,587 | 70.0% |
| Val | 584 | 185 | 769 | 15.0% |
| Test | 580 | 185 | 765 | 14.9% |

## Metadata Files

All metadata is committed to Git under `dataset_cleaning/`:

| File | Description |
|------|-------------|
| `split_manifest.csv` | Per-image manifest (filepath, class, source, subject, split) |
| `cleaning_log.csv` | Every removal action with reason |
| `subject_split_map.json` | Subject → split assignments |
| `dataset_config.json` | Full pipeline config (thresholds, seed, ratios) |

Audit reports are committed under `dataset_audit/reports/`.

## Validation

10 automated checks — all passed:

1. File integrity — 5,121 files, 0 errors
2. Resolution uniformity — all 224×224
3. No blank images remain
4. No intra-split MD5 duplicates
5. No subject leakage across splits
6. Both classes present in all splits
7. All 8 sources represented in all splits
8. No cross-class pHash collisions
9. Split ratios within ±5% of 70/15/15
10. Manifest ↔ disk consistency

---

*Phase 2 will cover CLIP feature extraction and hyperbolic projection.*
