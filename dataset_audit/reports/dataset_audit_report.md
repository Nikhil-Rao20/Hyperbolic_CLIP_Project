# MRI Dataset Audit Report

**Generated:** 2026-03-04 15:10:59

**Dataset Root:** `RGIIIT`

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Images | 6735 |
| Total Sources | 8 |
| Valid Files | 6735 |
| Corrupted/Error Files | 0 |
| Class Imbalance Ratio | 2.33 |
| Exact Duplicate Pairs | 124245 |
| Near-Duplicate Pairs | 500 |
| Leakage Issues | 268 |
| MRI Anomalies | 3951 |

### Overall Health: **NEEDS ATTENTION**
Multiple issues detected. Detailed review required.

---
## 1. Dataset Structure

| Class | Source | Files | Subjects |
|-------|--------|-------|----------|
| Fake | GAN | 185 | 37 |
| Fake | LDM | 2500 | 500 |
| Fake | MLS_CERMEP | 90 | 18 |
| Fake | MLS_TCGA | 255 | 51 |
| Fake | MLS_UPenn | 1680 | 336 |
| Real | cermep | 95 | 19 |
| Real | tcga | 255 | 51 |
| Real | upenn | 1675 | 335 |

---
## 2. File Integrity

- **Valid:** 6735
- **Corrupted (PIL):** 0
- **Zero-size:** 0
- **Load errors (OpenCV):** 0

All files passed integrity checks.

---
## 3. Image Statistics

- **Images analyzed:** 6735
- **Resolution range:** 176x144 to 256x256
- **Unique resolutions:** 4
- **Mean intensity:** 31.62 ± 28.41
- **Channel counts:** {1: 6735}

### Per-Source Statistics

| Source | Count | Width (mean) | Height (mean) | Mean Intensity | Std Intensity |
|--------|-------|-------------|--------------|----------------|---------------|
| GAN | 185 | 256 | 256 | 30.08 | 47.52 |
| LDM | 2500 | 176 | 144 | 49.04 | 51.80 |
| MLS_CERMEP | 90 | 243 | 207 | 44.75 | 45.95 |
| MLS_TCGA | 255 | 243 | 207 | 33.70 | 40.86 |
| MLS_UPenn | 1680 | 243 | 207 | 27.30 | 34.21 |
| cermep | 95 | 243 | 207 | 23.39 | 41.58 |
| tcga | 255 | 240 | 240 | 16.76 | 35.52 |
| upenn | 1675 | 240 | 240 | 11.84 | 24.94 |

See `image_statistics.csv` for full details.


Plots: `image_statistics_overview.png`, `intensity_by_source_boxplot.png`, `file_size_distribution.png`, `resolution_scatter.png`

---
## 4. Class Balance

| Class | Count | Percentage |
|-------|-------|------------|
| Fake | 4710 | 69.93% |
| Real | 2025 | 30.07% |

**Imbalance Ratio:** 2.33

> ⚠️ **Warning:** Significant class imbalance detected. Consider oversampling the minority class or using weighted loss.

---
## 5. Duplicate Detection

- **Exact duplicate groups (MD5):** 4
- **Exact duplicate pairs (total):** 124245
- **Near-duplicate pairs (pHash, sampled):** 500

See `duplicates_report.csv` for details.

---
## 6. Data Leakage Check

| Severity | Count |
|----------|-------|
| HIGH | 266 |
| WARNING | 1 |
| INFO | 1 |

- 🟡 **[WARNING]** subject_count_match: Exact subject count match (51). Possible 1:1 real-to-fake mapping. Ensure no data leakage in train/test splits.
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=c03f2ff0348393ce)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=c03f2ff030c3934f)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=c03f2ff0348393ce)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=c03f2ff0348393ce)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=c03f2ff030c3934f)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=c03f2ff0348393ce)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=c03d2ff030c393cf)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=0000000000000000)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=939b6c64939b6c64)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=936c6c93936c6c93)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=93936c6c9393666c)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b6664999b666499)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=999b6664999b6664)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=999b6664999b6664)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b6664999b666499)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b92646d9b92646d)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=999b6664999b6664)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b92646d9992666d)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=999b6664999b6664)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9964669b996466)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9964669b996466)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9964669b996466)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=926d6d92926d6d92)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b6664999b666499)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=999b6664999b6664)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=936d6c92936d6c92)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b6964b69b496436)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b6d64929b6d6492)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9964669b996466)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=999b6664999b6664)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b6664999b666499)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b6664999b666499)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9999666699996666)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b64649b9b64649b)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=9b9b64649b9b6464)
- 🟠 **[HIGH]** cross_class_phash: Perceptual hash match (pHash=92926d6d92926d6d)
- ℹ️ **[INFO]** no_split_detected: No train/test/val directory split found. If splitting is done at runtime, ensure subject-level splitting to prevent data leakage (same subject slices in both train and test).

See `data_leakage_report.csv` for details.

---
## 7. Visual Inspection

Sample grids generated:
- `samples/sample_real_grid.png`
- `samples/sample_fake_grid.png`

---
## 8. MRI-Specific Anomalies

| Check | Count |
|-------|-------|
| intensity_outlier | 46 |
| low_variance | 606 |
| mostly_black | 1698 |
| nearly_blank | 1601 |

See `mri_anomaly_report.csv` for details.

---
## 9. Synthetic Artifact Analysis (FFT)

FFT-based frequency analysis was performed to detect synthetic artifacts.

- **Real/cermep** (50 samples): Low=5.90, Mid=5.01, High=4.03
- **Real/tcga** (50 samples): Low=5.72, Mid=4.91, High=4.24
- **Real/upenn** (50 samples): Low=6.82, Mid=5.92, High=5.08
- **Fake/GAN** (50 samples): Low=5.82, Mid=4.79, High=3.96
- **Fake/LDM** (50 samples): Low=9.07, Mid=7.76, High=5.69
- **Fake/MLS_CERMEP** (50 samples): Low=9.01, Mid=7.33, High=5.48
- **Fake/MLS_TCGA** (50 samples): Low=8.99, Mid=7.52, High=5.53
- **Fake/MLS_UPenn** (50 samples): Low=8.92, Mid=7.51, High=5.52

Plots: `fft_analysis.png`, `fft_sample_comparison.png`

---
## 10. Recommendations

1. **Class Imbalance:** Apply oversampling (e.g., SMOTE), undersampling, or use weighted loss functions to address the class imbalance.
2. **Duplicates:** Remove exact duplicate images to prevent data leakage and inflated metrics.
3. **Train/Test Split:** No pre-defined split found. Implement **subject-level splitting** to ensure all slices from the same subject are in the same split.
4. **Blank Images:** Remove or review nearly-blank images that may harm training.

---
*Report generated by MRI Dataset Audit Script*
