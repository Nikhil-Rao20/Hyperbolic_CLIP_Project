from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "experiments" / "source_specific_ood_analysis"

DOMAINS = ["Real", "GAN", "LDM", "MLS"]
GEOMETRIES = ["euclidean", "hyperbolic"]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / n) + ((z**2) / (4 * n**2)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(values: List[float], n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(np.mean(sample)))
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def read_results(exp_root: Path, geometry: str) -> List[Dict]:
    out: List[Dict] = []
    if not exp_root.exists():
        return out

    seed_dirs = sorted([p for p in exp_root.glob("seed_*") if p.is_dir()])
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_")[-1])
        for domain in DOMAINS:
            rpath = seed_dir / f"domain_{domain}" / "results.json"
            if not rpath.exists():
                continue
            data = json.loads(rpath.read_text(encoding="utf-8"))
            cm = data["confusion_matrix"]
            tn, fp = cm[0]
            fn, tp = cm[1]
            out.append(
                {
                    "seed": seed,
                    "domain": domain,
                    "geometry": geometry,
                    "accuracy": float(data["accuracy"]),
                    "precision": float(data["precision"]),
                    "recall": float(data["recall"]),
                    "specificity": float(data["specificity"]),
                    "PPV": float(data["PPV"]),
                    "NPV": float(data["NPV"]),
                    "f1": float(data["f1"]),
                    "auroc": float(data["auroc"]),
                    "auprc": float(data["auprc"]),
                    "default_threshold": float(data["default_threshold"]),
                    "calibrated_threshold": float(data["calibrated_threshold"]),
                    "split_hash": str(data["split_hash"]),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "per_source_accuracy": data.get("per_source_accuracy", {}),
                }
            )
    return out


def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def aggregate(rows: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        key = (r["domain"], r["geometry"])
        grouped.setdefault(key, []).append(r)

    summary_rows = []
    threshold_rows = []
    subgroup_rows = []

    for (domain, geometry), grp in sorted(grouped.items()):
        def m(key: str):
            vals = [x[key] for x in grp]
            return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)

        auroc_vals = [x["auroc"] for x in grp]
        auprc_vals = [x["auprc"] for x in grp]
        auroc_ci = bootstrap_ci(auroc_vals)
        auprc_ci = bootstrap_ci(auprc_vals)

        tp = sum(x["tp"] for x in grp)
        tn = sum(x["tn"] for x in grp)
        fp = sum(x["fp"] for x in grp)
        fn = sum(x["fn"] for x in grp)

        sens_ci = wilson_interval(tp, tp + fn)
        spec_ci = wilson_interval(tn, tn + fp)
        ppv_ci = wilson_interval(tp, tp + fp)
        npv_ci = wilson_interval(tn, tn + fn)

        acc_mu, acc_sd = m("accuracy")
        rec_mu, rec_sd = m("recall")
        spe_mu, spe_sd = m("specificity")
        ppv_mu, ppv_sd = m("PPV")
        npv_mu, npv_sd = m("NPV")
        f1_mu, f1_sd = m("f1")
        auroc_mu, auroc_sd = m("auroc")
        auprc_mu, auprc_sd = m("auprc")

        summary_rows.append(
            {
                "domain": domain,
                "geometry": geometry,
                "n_seeds": len(grp),
                "accuracy_mean": round(acc_mu, 6),
                "accuracy_std": round(acc_sd, 6),
                "recall_mean": round(rec_mu, 6),
                "recall_std": round(rec_sd, 6),
                "specificity_mean": round(spe_mu, 6),
                "specificity_std": round(spe_sd, 6),
                "PPV_mean": round(ppv_mu, 6),
                "PPV_std": round(ppv_sd, 6),
                "NPV_mean": round(npv_mu, 6),
                "NPV_std": round(npv_sd, 6),
                "f1_mean": round(f1_mu, 6),
                "f1_std": round(f1_sd, 6),
                "auroc_mean": round(auroc_mu, 6),
                "auroc_std": round(auroc_sd, 6),
                "auroc_ci_low": round(auroc_ci[0], 6),
                "auroc_ci_high": round(auroc_ci[1], 6),
                "auprc_mean": round(auprc_mu, 6),
                "auprc_std": round(auprc_sd, 6),
                "auprc_ci_low": round(auprc_ci[0], 6),
                "auprc_ci_high": round(auprc_ci[1], 6),
                "sensitivity_ci_low": round(sens_ci[0], 6),
                "sensitivity_ci_high": round(sens_ci[1], 6),
                "specificity_ci_low": round(spec_ci[0], 6),
                "specificity_ci_high": round(spec_ci[1], 6),
                "PPV_ci_low": round(ppv_ci[0], 6),
                "PPV_ci_high": round(ppv_ci[1], 6),
                "NPV_ci_low": round(npv_ci[0], 6),
                "NPV_ci_high": round(npv_ci[1], 6),
            }
        )

        cal = [x["calibrated_threshold"] for x in grp]
        dft = [x["default_threshold"] for x in grp]
        threshold_rows.append(
            {
                "domain": domain,
                "geometry": geometry,
                "n_seeds": len(grp),
                "default_threshold_mean": round(mean(dft), 6),
                "default_threshold_std": round(stdev(dft) if len(dft) > 1 else 0.0, 6),
                "calibrated_threshold_mean": round(mean(cal), 6),
                "calibrated_threshold_std": round(stdev(cal) if len(cal) > 1 else 0.0, 6),
                "split_hash_unique_count": len(set(x["split_hash"] for x in grp)),
            }
        )

        # Per-source subgroup summary across seeds.
        source_vals: Dict[str, List[float]] = {}
        for x in grp:
            for s, v in x["per_source_accuracy"].items():
                source_vals.setdefault(s, []).append(float(v))
        for src, vals in sorted(source_vals.items()):
            subgroup_rows.append(
                {
                    "domain": domain,
                    "geometry": geometry,
                    "source": src,
                    "mean_accuracy": round(mean(vals), 6),
                    "std_accuracy": round(stdev(vals) if len(vals) > 1 else 0.0, 6),
                    "n_seeds": len(vals),
                }
            )

    return summary_rows, threshold_rows, subgroup_rows


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed source-specific OOD results")
    parser.add_argument(
        "--euclidean-root",
        type=str,
        default="experiments/source_specific_ood_euclidean_reruns",
    )
    parser.add_argument(
        "--hyperbolic-root",
        type=str,
        default="experiments/source_specific_ood_hyperbolic_reruns",
    )
    args = parser.parse_args()

    rows = []
    rows.extend(read_results(PROJECT_ROOT / args.euclidean_root, "euclidean"))
    rows.extend(read_results(PROJECT_ROOT / args.hyperbolic_root, "hyperbolic"))

    if not rows:
        raise RuntimeError("No seed-based results found. Expected seed_* folders with results.json files.")

    write_csv(OUT_DIR / "multi_seed_per_run_metrics.csv", sorted(rows, key=lambda r: (r["domain"], r["geometry"], r["seed"])))

    summary_rows, threshold_rows, subgroup_rows = aggregate(rows)
    write_csv(OUT_DIR / "multi_seed_summary_with_ci.csv", summary_rows)
    write_csv(OUT_DIR / "multi_seed_threshold_stability.csv", threshold_rows)
    write_csv(OUT_DIR / "multi_seed_subgroup_summary.csv", subgroup_rows)

    print("Saved:")
    print(OUT_DIR / "multi_seed_per_run_metrics.csv")
    print(OUT_DIR / "multi_seed_summary_with_ci.csv")
    print(OUT_DIR / "multi_seed_threshold_stability.csv")
    print(OUT_DIR / "multi_seed_subgroup_summary.csv")


if __name__ == "__main__":
    main()
