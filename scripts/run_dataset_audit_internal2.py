from __future__ import annotations

import hashlib
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import imagehash
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError


RANDOM_SEED = 1337
MAX_SAMPLE_PER_CLASS = 500
PHASH_NEAR_DUP_THRESHOLD = 4
SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
}
TARGET_CLASSES = ["Real", "GAN", "LDM", "MLS"]


@dataclass
class ImageRecord:
    path: Path
    rel_path: str
    class_name: Optional[str]
    extension: str
    file_size_bytes: int


@dataclass
class ValidImageRecord:
    record: ImageRecord
    mode: str
    size: Tuple[int, int]
    np_array: np.ndarray
    md5: str
    phash_int: int


def log(message: str) -> None:
    print(message, flush=True)


def classify_path(path: Path, dataset_root: Path) -> Optional[str]:
    rel_parts = [p.lower() for p in path.relative_to(dataset_root).parts]
    file_name = path.name.lower()

    if "real" in rel_parts:
        return "Real"

    if "gan" in rel_parts or file_name.startswith("gan__"):
        return "GAN"

    if "ldm" in rel_parts or file_name.startswith("ldm__"):
        return "LDM"

    if "mls" in rel_parts or file_name.startswith("mls_") or file_name.startswith("mls__"):
        return "MLS"

    if "fake" in rel_parts:
        if file_name.startswith("gan"):
            return "GAN"
        if file_name.startswith("ldm"):
            return "LDM"
        if file_name.startswith("mls"):
            return "MLS"

    return None


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def imagehash_to_int(h: imagehash.ImageHash) -> int:
    bits = h.hash.flatten().astype(np.uint8)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance_u64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def summarize_size_bytes(total_bytes: int) -> float:
    return total_bytes / (1024.0 * 1024.0)


def make_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, sep_row] + data_rows)


def generate_sample_grid(class_name: str, items: List[ValidImageRecord], output_path: Path) -> None:
    random.seed(RANDOM_SEED + hash(class_name) % 1000)
    k = min(16, len(items))
    selected = random.sample(items, k=k) if k > 0 else []

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes_flat = axes.flatten()

    for idx, ax in enumerate(axes_flat):
        ax.axis("off")
        if idx < len(selected):
            arr = selected[idx].np_array
            if arr.ndim == 2:
                ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(arr)
            ax.set_title(selected[idx].record.path.name[:30], fontsize=7)

    fig.suptitle(f"Random Samples: {class_name}", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def nearest_duplicate_pairs_by_lsh(
    records: List[ValidImageRecord], threshold: int
) -> List[Tuple[int, int, int]]:
    # 5-way chunking guarantees at least one exact chunk match for distances <= 4.
    boundaries = [13, 13, 13, 13, 12]
    offsets = []
    current = 64
    for width in boundaries:
        current -= width
        offsets.append((current, width))

    bucket_maps: List[Dict[int, List[int]]] = [defaultdict(list) for _ in boundaries]
    pairs: List[Tuple[int, int, int]] = []

    for i, rec in enumerate(records):
        value = rec.phash_int
        candidates = set()
        for m_idx, (offset, width) in enumerate(offsets):
            key = (value >> offset) & ((1 << width) - 1)
            candidates.update(bucket_maps[m_idx][key])

        for j in candidates:
            dist = hamming_distance_u64(value, records[j].phash_int)
            if dist <= threshold:
                pairs.append((j, i, dist))

        for m_idx, (offset, width) in enumerate(offsets):
            key = (value >> offset) & ((1 << width) - 1)
            bucket_maps[m_idx][key].append(i)

    return pairs


def main() -> int:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    project_root = Path(__file__).resolve().parent.parent
    dataset_root = project_root / "RGIIIT_clean"
    out_dir = project_root / "docs" / "internal2"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}", file=sys.stderr)
        return 1

    log("STEP 1/8 - Directory & file inventory")

    expected_structure = {
        "Real": dataset_root / "Real",
        "GAN": dataset_root / "GAN",
        "LDM": dataset_root / "LDM",
        "MLS": dataset_root / "MLS",
    }

    expected_structure_exists = {k: p.exists() for k, p in expected_structure.items()}

    all_files: List[Path] = [p for p in dataset_root.rglob("*") if p.is_file()]
    total_size_bytes = sum(p.stat().st_size for p in all_files)
    unique_extensions = sorted({p.suffix.lower() if p.suffix else "<no_ext>" for p in all_files})

    inventory_counts: Dict[str, int] = defaultdict(int)
    image_records: List[ImageRecord] = []
    non_image_files: List[str] = []
    unexpected_files: List[str] = []

    for p in all_files:
        rel = p.relative_to(dataset_root).as_posix()
        parent = p.parent.relative_to(dataset_root).as_posix()
        inventory_counts[parent] += 1

        ext = p.suffix.lower()
        class_name = classify_path(p, dataset_root)
        is_image_ext = ext in SUPPORTED_IMAGE_EXTENSIONS

        record = ImageRecord(
            path=p,
            rel_path=rel,
            class_name=class_name,
            extension=ext if ext else "<no_ext>",
            file_size_bytes=p.stat().st_size,
        )

        if is_image_ext:
            image_records.append(record)
        else:
            non_image_files.append(rel)

        if class_name is None:
            unexpected_files.append(rel)

    class_counts_from_paths = Counter(r.class_name for r in image_records if r.class_name is not None)

    log("STEP 2/8 - Image format & integrity")

    valid_records: List[ValidImageRecord] = []
    corrupted_by_class = Counter()
    corrupted_paths: List[str] = []
    mode_set_by_class: Dict[str, set] = defaultdict(set)
    size_set_by_class: Dict[str, set] = defaultdict(set)
    mode_counts_by_class: Dict[str, Counter] = defaultdict(Counter)

    for rec in image_records:
        cls = rec.class_name or "Unclassified"
        try:
            with Image.open(rec.path) as img:
                img.verify()

            with Image.open(rec.path) as img2:
                arr = np.array(img2)
                md5 = md5_file(rec.path)
                ph = imagehash.phash(img2)
                ph_int = imagehash_to_int(ph)
                mode = img2.mode
                size = img2.size

            vrec = ValidImageRecord(
                record=rec,
                mode=mode,
                size=size,
                np_array=arr,
                md5=md5,
                phash_int=ph_int,
            )
            valid_records.append(vrec)
            mode_set_by_class[cls].add(mode)
            size_set_by_class[cls].add(size)
            mode_counts_by_class[cls][mode] += 1
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            corrupted_by_class[cls] += 1
            corrupted_paths.append(rec.rel_path)

    class_valid_records: Dict[str, List[ValidImageRecord]] = defaultdict(list)
    for vr in valid_records:
        cls = vr.record.class_name or "Unclassified"
        class_valid_records[cls].append(vr)

    log("STEP 3/8 - Class balance analysis")

    class_counts = {cls: len(class_valid_records.get(cls, [])) for cls in TARGET_CLASSES}
    total_classified = sum(class_counts.values())
    class_percentages = {
        cls: (100.0 * count / total_classified if total_classified > 0 else 0.0)
        for cls, count in class_counts.items()
    }

    largest_class = max(class_counts.values()) if class_counts else 0
    smallest_nonzero = min([c for c in class_counts.values() if c > 0], default=0)
    imbalance_ratio = (largest_class / smallest_nonzero) if smallest_nonzero else math.inf

    min_class_size = smallest_nonzero

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(class_counts.keys()), list(class_counts.values()))
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Image Count")
    for i, v in enumerate(class_counts.values()):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    class_dist_plot = out_dir / "class_distribution.png"
    fig.savefig(class_dist_plot, dpi=150)
    plt.close(fig)

    log("STEP 4/8 - Pixel statistics")

    sampled_by_class: Dict[str, List[ValidImageRecord]] = {}
    stats_by_class: Dict[str, Dict[str, float]] = {}
    rgb_stats_by_class: Dict[str, Dict[str, float]] = {}
    grayscale_rgb_check: Dict[str, Dict[str, int]] = {}

    hist_values_by_class: Dict[str, np.ndarray] = {}

    for cls in TARGET_CLASSES:
        records = class_valid_records.get(cls, [])
        if not records:
            sampled_by_class[cls] = []
            continue

        sample_n = min(MAX_SAMPLE_PER_CLASS, len(records))
        sampled = random.sample(records, k=sample_n)
        sampled_by_class[cls] = sampled

        all_pixels = []
        rgb_pixels = []
        rgb_equal_counter = 0
        rgb_total_counter = 0

        for rec in sampled:
            arr = rec.np_array
            all_pixels.append(arr.reshape(-1))

            if arr.ndim == 3 and arr.shape[2] >= 3:
                rgb = arr[:, :, :3]
                rgb_pixels.append(rgb.reshape(-1, 3))
                rgb_total_counter += 1
                if np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 1], rgb[:, :, 2]):
                    rgb_equal_counter += 1

        flat = np.concatenate(all_pixels) if all_pixels else np.array([], dtype=np.uint8)
        if flat.size > 0:
            stats_by_class[cls] = {
                "mean": float(np.mean(flat)),
                "std": float(np.std(flat)),
                "min": float(np.min(flat)),
                "max": float(np.max(flat)),
            }
            hist_values_by_class[cls] = flat
        else:
            stats_by_class[cls] = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
            hist_values_by_class[cls] = np.array([], dtype=np.uint8)

        if rgb_pixels:
            rgb_flat = np.concatenate(rgb_pixels, axis=0)
            rgb_stats_by_class[cls] = {
                "mean_r": float(np.mean(rgb_flat[:, 0])),
                "mean_g": float(np.mean(rgb_flat[:, 1])),
                "mean_b": float(np.mean(rgb_flat[:, 2])),
                "std_r": float(np.std(rgb_flat[:, 0])),
                "std_g": float(np.std(rgb_flat[:, 1])),
                "std_b": float(np.std(rgb_flat[:, 2])),
            }
        else:
            rgb_stats_by_class[cls] = {}

        grayscale_rgb_check[cls] = {
            "rgb_images_sampled": rgb_total_counter,
            "rgb_equal_images": rgb_equal_counter,
        }

    fig, ax = plt.subplots(figsize=(10, 6))
    for cls in TARGET_CLASSES:
        values = hist_values_by_class.get(cls, np.array([], dtype=np.uint8))
        if values.size > 0:
            ax.hist(values, bins=64, alpha=0.35, label=cls, density=True)
    ax.set_title("Pixel Intensity Histogram (Sampled)")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    histogram_plot = out_dir / "pixel_intensity_histogram.png"
    fig.savefig(histogram_plot, dpi=150)
    plt.close(fig)

    log("STEP 5/8 - Duplicate detection")

    md5_groups: Dict[str, List[ValidImageRecord]] = defaultdict(list)
    for rec in valid_records:
        md5_groups[rec.md5].append(rec)

    exact_duplicate_groups = {k: v for k, v in md5_groups.items() if len(v) > 1}

    exact_dup_within_class_count = Counter()
    exact_dup_cross_class_count = 0
    exact_dup_cross_class_details: List[Tuple[str, List[str], List[str]]] = []

    for md5, group in exact_duplicate_groups.items():
        classes = [g.record.class_name or "Unclassified" for g in group]
        class_set = set(classes)

        class_counter = Counter(classes)
        for cls, count in class_counter.items():
            if count > 1:
                exact_dup_within_class_count[cls] += count

        if len(class_set) > 1:
            exact_dup_cross_class_count += len(group)
            exact_dup_cross_class_details.append(
                (
                    md5,
                    sorted(class_set),
                    [g.record.rel_path for g in group],
                )
            )

    near_pairs = nearest_duplicate_pairs_by_lsh(valid_records, PHASH_NEAR_DUP_THRESHOLD)

    near_dup_within_class: List[Tuple[str, int, str, str]] = []
    near_dup_cross_class: List[Tuple[str, str, int, str, str]] = []

    for i, j, dist in near_pairs:
        a = valid_records[i]
        b = valid_records[j]
        a_cls = a.record.class_name or "Unclassified"
        b_cls = b.record.class_name or "Unclassified"
        if a_cls == b_cls:
            near_dup_within_class.append((a_cls, dist, a.record.rel_path, b.record.rel_path))
        else:
            near_dup_cross_class.append((a_cls, b_cls, dist, a.record.rel_path, b.record.rel_path))

    near_dup_within_class_counts = Counter(item[0] for item in near_dup_within_class)

    log("STEP 6/8 - Visual sample grids")

    sample_grid_paths = {}
    for cls in TARGET_CLASSES:
        out_path = out_dir / f"samples_{cls}.png"
        generate_sample_grid(cls, class_valid_records.get(cls, []), out_path)
        sample_grid_paths[cls] = out_path

    log("STEP 7/8 - Train/test split feasibility")

    total_real = class_counts.get("Real", 0)
    real_train_target = 500
    real_test_pool = max(total_real - real_train_target, 0)

    fake_counts = {
        "GAN": class_counts.get("GAN", 0),
        "LDM": class_counts.get("LDM", 0),
        "MLS": class_counts.get("MLS", 0),
    }
    fake_balanced_size_each = min(fake_counts.values()) if fake_counts and min(fake_counts.values()) > 0 else 0
    fake_balanced_total = fake_balanced_size_each * 3

    k = 5
    fold_val = real_train_target // k
    fold_train = real_train_target - fold_val

    log("STEP 8/8 - Anomaly flags")

    anomaly_counts = {
        "extreme_mean": Counter(),
        "non_square": Counter(),
        "small_lt_64": Counter(),
    }

    anomaly_details = {
        "extreme_mean": [],
        "non_square": [],
        "small_lt_64": [],
    }

    all_square = True
    for rec in valid_records:
        w, h = rec.size
        if w != h:
            all_square = False
            break

    for rec in valid_records:
        cls = rec.record.class_name or "Unclassified"
        arr = rec.np_array
        mean_val = float(np.mean(arr))
        w, h = rec.size

        if mean_val < 5 or mean_val > 250:
            anomaly_counts["extreme_mean"][cls] += 1
            anomaly_details["extreme_mean"].append((cls, rec.record.rel_path, mean_val))

        if all_square and w != h:
            anomaly_counts["non_square"][cls] += 1
            anomaly_details["non_square"].append((cls, rec.record.rel_path, f"{w}x{h}"))

        if w < 64 or h < 64:
            anomaly_counts["small_lt_64"][cls] += 1
            anomaly_details["small_lt_64"].append((cls, rec.record.rel_path, f"{w}x{h}"))

    log("Generating markdown report")

    rel = lambda p: p.relative_to(project_root).as_posix()

    inventory_rows = [[folder, str(count)] for folder, count in sorted(inventory_counts.items(), key=lambda x: x[0])]
    class_count_rows = [[cls, str(class_counts.get(cls, 0))] for cls in TARGET_CLASSES]

    format_rows = []
    for cls in TARGET_CLASSES:
        modes = sorted(mode_set_by_class.get(cls, set()))
        sizes = sorted(size_set_by_class.get(cls, set()))
        format_rows.append(
            [
                cls,
                ", ".join(modes) if modes else "-",
                str(len(sizes)),
                "Variable" if len(sizes) > 1 else (f"Consistent ({sizes[0][0]}x{sizes[0][1]})" if len(sizes) == 1 else "-"),
            ]
        )

    mode_count_rows = []
    for cls in TARGET_CLASSES:
        counter = mode_counts_by_class.get(cls, Counter())
        if not counter:
            mode_count_rows.append([cls, "-", "0"])
            continue
        for mode, cnt in sorted(counter.items()):
            mode_count_rows.append([cls, mode, str(cnt)])

    balance_rows = [
        [
            cls,
            str(class_counts[cls]),
            f"{class_percentages[cls]:.2f}%",
        ]
        for cls in TARGET_CLASSES
    ]

    pixel_rows = []
    for cls in TARGET_CLASSES:
        s = stats_by_class.get(cls, {})
        pixel_rows.append(
            [
                cls,
                str(len(sampled_by_class.get(cls, []))),
                f"{s.get('mean', float('nan')):.3f}",
                f"{s.get('std', float('nan')):.3f}",
                f"{s.get('min', float('nan')):.1f}",
                f"{s.get('max', float('nan')):.1f}",
            ]
        )

    rgb_rows = []
    for cls in TARGET_CLASSES:
        rs = rgb_stats_by_class.get(cls, {})
        if rs:
            rgb_rows.append(
                [
                    cls,
                    f"{rs['mean_r']:.3f}",
                    f"{rs['mean_g']:.3f}",
                    f"{rs['mean_b']:.3f}",
                    f"{rs['std_r']:.3f}",
                    f"{rs['std_g']:.3f}",
                    f"{rs['std_b']:.3f}",
                ]
            )
        else:
            rgb_rows.append([cls, "-", "-", "-", "-", "-", "-"])

    gray_check_rows = []
    for cls in TARGET_CLASSES:
        g = grayscale_rgb_check.get(cls, {"rgb_images_sampled": 0, "rgb_equal_images": 0})
        pct = (100.0 * g["rgb_equal_images"] / g["rgb_images_sampled"]) if g["rgb_images_sampled"] > 0 else 0.0
        gray_check_rows.append([cls, str(g["rgb_images_sampled"]), str(g["rgb_equal_images"]), f"{pct:.2f}%"])

    split_rows = [
        ["Real total", str(total_real)],
        ["Real train target", str(real_train_target)],
        ["Real test pool", str(real_test_pool)],
        ["GAN test pool", str(fake_counts["GAN"])],
        ["LDM test pool", str(fake_counts["LDM"])],
        ["MLS test pool", str(fake_counts["MLS"])],
        ["Balanced fake each (GAN=LDM=MLS)", str(fake_balanced_size_each)],
        ["Balanced fake total", str(fake_balanced_total)],
        ["5-fold CV (on 500 real): train/fold", str(fold_train)],
        ["5-fold CV (on 500 real): val/fold", str(fold_val)],
    ]

    anomaly_rows = []
    for cls in TARGET_CLASSES:
        anomaly_rows.append(
            [
                cls,
                str(anomaly_counts["extreme_mean"].get(cls, 0)),
                str(anomaly_counts["non_square"].get(cls, 0)),
                str(anomaly_counts["small_lt_64"].get(cls, 0)),
            ]
        )

    report_lines: List[str] = []
    report_lines.append("# Dataset Audit Report: RGIIIT_clean")
    report_lines.append("")
    report_lines.append("## Executive summary")

    exec_points = [
        f"Expected top-level class folders present status: Real={expected_structure_exists['Real']}, GAN={expected_structure_exists['GAN']}, LDM={expected_structure_exists['LDM']}, MLS={expected_structure_exists['MLS']}",
        f"Total files scanned: {len(all_files)}; total size: {summarize_size_bytes(total_size_bytes):.2f} MB",
        f"Class image counts (valid images): Real={class_counts['Real']}, GAN={class_counts['GAN']}, LDM={class_counts['LDM']}, MLS={class_counts['MLS']}",
        f"Corrupted/unreadable images: {sum(corrupted_by_class.values())}",
        f"Class imbalance ratio (largest/smallest non-zero): {imbalance_ratio:.4f}" if math.isfinite(imbalance_ratio) else "Class imbalance ratio: infinite (one or more classes are zero)",
        f"Cross-class exact duplicates found: {len(exact_dup_cross_class_details)} MD5 groups; near-duplicate cross-class pairs (pHash <= {PHASH_NEAR_DUP_THRESHOLD}): {len(near_dup_cross_class)}",
    ]
    report_lines.extend([f"- {line}" for line in exec_points])
    report_lines.append("")

    report_lines.append("## Step 1 - Directory & file inventory")
    report_lines.append("### Structure verification")
    structure_rows = [[cls, str(path.relative_to(project_root).as_posix()), str(exists)] for cls, path in expected_structure.items() for exists in [expected_structure_exists[cls]]]
    report_lines.append(make_markdown_table(["Class", "Expected path", "Exists"], structure_rows))
    report_lines.append("")
    report_lines.append("### Files per subfolder")
    report_lines.append(make_markdown_table(["Subfolder (relative to RGIIIT_clean)", "File count"], inventory_rows))
    report_lines.append("")
    report_lines.append("### Class counts from discovered image files")
    report_lines.append(make_markdown_table(["Class", "Image count"], class_count_rows))
    report_lines.append("")
    report_lines.append(f"Total dataset size: {summarize_size_bytes(total_size_bytes):.2f} MB")
    report_lines.append("")
    report_lines.append("Unique file extensions found:")
    report_lines.append("")
    report_lines.append(", ".join(unique_extensions) if unique_extensions else "-")
    report_lines.append("")

    report_lines.append("Non-image files detected:")
    report_lines.append("")
    if non_image_files:
        for p in non_image_files[:200]:
            report_lines.append(f"- {p}")
        if len(non_image_files) > 200:
            report_lines.append(f"- ... truncated, total non-image files: {len(non_image_files)}")
    else:
        report_lines.append("- None")
    report_lines.append("")

    report_lines.append("Unexpected/unclassified files:")
    report_lines.append("")
    if unexpected_files:
        for p in unexpected_files[:200]:
            report_lines.append(f"- {p}")
        if len(unexpected_files) > 200:
            report_lines.append(f"- ... truncated, total unexpected files: {len(unexpected_files)}")
    else:
        report_lines.append("- None")
    report_lines.append("")

    report_lines.append("## Step 2 - Image format & integrity")
    report_lines.append("Corrupted/unreadable images per class:")
    report_lines.append("")
    corr_rows = [[cls, str(corrupted_by_class.get(cls, 0))] for cls in TARGET_CLASSES]
    report_lines.append(make_markdown_table(["Class", "Corrupted count"], corr_rows))
    report_lines.append("")
    if corrupted_paths:
        report_lines.append("Corrupted file paths:")
        report_lines.append("")
        for p in corrupted_paths[:200]:
            report_lines.append(f"- {p}")
        if len(corrupted_paths) > 200:
            report_lines.append(f"- ... truncated, total corrupted files: {len(corrupted_paths)}")
    else:
        report_lines.append("No corrupted files detected.")
    report_lines.append("")

    report_lines.append("Image mode and size summary by class:")
    report_lines.append("")
    report_lines.append(make_markdown_table(["Class", "Modes", "Unique sizes", "Size consistency"], format_rows))
    report_lines.append("")
    report_lines.append("Mode counts per class:")
    report_lines.append("")
    report_lines.append(make_markdown_table(["Class", "Mode", "Count"], mode_count_rows))
    report_lines.append("")

    report_lines.append("## Step 3 - Class balance analysis")
    report_lines.append(make_markdown_table(["Class", "Count", "Percentage"], balance_rows))
    report_lines.append("")
    report_lines.append(f"Imbalance ratio (largest/smallest non-zero): {imbalance_ratio:.4f}" if math.isfinite(imbalance_ratio) else "Imbalance ratio: infinite")
    report_lines.append("")
    report_lines.append(f"Balanced subset size per class (limited by smallest class): {min_class_size}")
    report_lines.append("")
    report_lines.append(f"![Class distribution]({rel(class_dist_plot)})")
    report_lines.append("")

    report_lines.append("## Step 4 - Pixel statistics (sample up to 500 per class)")
    report_lines.append(make_markdown_table(["Class", "Sample size", "Mean", "Std", "Min", "Max"], pixel_rows))
    report_lines.append("")
    report_lines.append("Per-channel RGB statistics (where RGB data exists):")
    report_lines.append("")
    report_lines.append(make_markdown_table(["Class", "Mean R", "Mean G", "Mean B", "Std R", "Std G", "Std B"], rgb_rows))
    report_lines.append("")
    report_lines.append("Grayscale-stored-as-RGB check (R==G==B on sampled RGB images):")
    report_lines.append("")
    report_lines.append(make_markdown_table(["Class", "RGB sampled", "R==G==B count", "R==G==B %"], gray_check_rows))
    report_lines.append("")
    report_lines.append(f"![Pixel intensity histogram]({rel(histogram_plot)})")
    report_lines.append("")

    report_lines.append("## Step 5 - Duplicate detection")
    exact_rows = [[cls, str(exact_dup_within_class_count.get(cls, 0))] for cls in TARGET_CLASSES]
    report_lines.append("Exact duplicate file count contribution (same MD5, within class):")
    report_lines.append("")
    report_lines.append(make_markdown_table(["Class", "Duplicate file count"], exact_rows))
    report_lines.append("")
    report_lines.append(f"Cross-class exact duplicate groups: {len(exact_dup_cross_class_details)}")
    report_lines.append("")

    if exact_dup_cross_class_details:
        report_lines.append("Cross-class exact duplicate groups (critical):")
        report_lines.append("")
        for md5, classes, paths in exact_dup_cross_class_details:
            report_lines.append(f"- MD5 `{md5}` | Classes: {', '.join(classes)}")
            for p in paths:
                report_lines.append(f"  - {p}")
    else:
        report_lines.append("No cross-class exact duplicates found.")
    report_lines.append("")

    report_lines.append(f"Near-duplicate threshold: pHash distance <= {PHASH_NEAR_DUP_THRESHOLD}")
    report_lines.append("")
    near_rows = [[cls, str(near_dup_within_class_counts.get(cls, 0))] for cls in TARGET_CLASSES]
    report_lines.append("Near-duplicate pairs within class:")
    report_lines.append("")
    report_lines.append(make_markdown_table(["Class", "Near-duplicate pair count"], near_rows))
    report_lines.append("")
    report_lines.append(f"Near-duplicate cross-class pair count: {len(near_dup_cross_class)}")
    report_lines.append("")

    if near_dup_cross_class:
        report_lines.append("Near-duplicate cross-class pairs (critical):")
        report_lines.append("")
        for a_cls, b_cls, dist, a_path, b_path in near_dup_cross_class[:500]:
            report_lines.append(f"- [{a_cls}] {a_path} <-> [{b_cls}] {b_path} | pHash distance={dist}")
        if len(near_dup_cross_class) > 500:
            report_lines.append(f"- ... truncated, total cross-class near-duplicate pairs: {len(near_dup_cross_class)}")
    else:
        report_lines.append("No cross-class near-duplicate pairs found.")
    report_lines.append("")

    report_lines.append("## Step 6 - Visual sample grids")
    for cls in TARGET_CLASSES:
        report_lines.append(f"### {cls}")
        report_lines.append(f"![Samples {cls}]({rel(sample_grid_paths[cls])})")
        report_lines.append("")

    report_lines.append("## Step 7 - Train/test split feasibility")
    report_lines.append(make_markdown_table(["Metric", "Value"], split_rows))
    report_lines.append("")

    report_lines.append("## Step 8 - Anomaly flags")
    report_lines.append(make_markdown_table(["Class", "Extreme mean (<5 or >250)", "Non-square (if expected square)", "Smaller than 64x64"], anomaly_rows))
    report_lines.append("")

    if anomaly_details["extreme_mean"]:
        report_lines.append("Extreme mean flagged images:")
        report_lines.append("")
        for cls, p, v in anomaly_details["extreme_mean"][:500]:
            report_lines.append(f"- [{cls}] {p} | mean={v:.3f}")
        if len(anomaly_details["extreme_mean"]) > 500:
            report_lines.append(f"- ... truncated, total extreme-mean flags: {len(anomaly_details['extreme_mean'])}")
        report_lines.append("")

    if anomaly_details["non_square"]:
        report_lines.append("Non-square flagged images:")
        report_lines.append("")
        for cls, p, sz in anomaly_details["non_square"][:500]:
            report_lines.append(f"- [{cls}] {p} | size={sz}")
        if len(anomaly_details["non_square"]) > 500:
            report_lines.append(f"- ... truncated, total non-square flags: {len(anomaly_details['non_square'])}")
        report_lines.append("")

    if anomaly_details["small_lt_64"]:
        report_lines.append("Small (<64x64) flagged images:")
        report_lines.append("")
        for cls, p, sz in anomaly_details["small_lt_64"][:500]:
            report_lines.append(f"- [{cls}] {p} | size={sz}")
        if len(anomaly_details["small_lt_64"]) > 500:
            report_lines.append(f"- ... truncated, total small-size flags: {len(anomaly_details['small_lt_64'])}")
        report_lines.append("")

    report_lines.append("## Recommendations for experiment design")
    recommendations = [
        "Use strict hash-based deduplication before any train/test split, and remove cross-class duplicates first to avoid label leakage.",
        "If class imbalance is present, use class-balanced sampling or weighted loss during training and report macro-averaged metrics.",
        "If image sizes or modes vary, add a deterministic preprocessing pipeline (mode conversion + resize + normalization) and version it.",
        "Exclude corrupted files and maintain a machine-generated exclusion manifest so future runs are reproducible.",
        "For generator attribution experiments, build fake test subsets that are balanced across GAN/LDM/MLS to prevent generator-prior bias.",
        "If many RGB images are effectively grayscale (R==G==B), consider single-channel pipelines to reduce compute and avoid redundant channels.",
    ]
    report_lines.extend([f"- {r}" for r in recommendations])
    report_lines.append("")

    report_path = out_dir / "dataset_audit.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    log(f"Audit complete. Report written to: {report_path}")
    log(f"Class distribution chart: {class_dist_plot}")
    log(f"Pixel histogram chart: {histogram_plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
