from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
EUC_DIR = EXPERIMENTS_ROOT / "source_specific_ood_euclidean"
HYP_DIR = EXPERIMENTS_ROOT / "source_specific_ood_hyperbolic"
ANALYSIS_DIR = EXPERIMENTS_ROOT / "source_specific_ood_analysis"
SNAPSHOT_ROOT = EXPERIMENTS_ROOT / "snapshots"

DOMAINS = ["Real", "GAN", "LDM", "MLS"]
GEOMETRY_DIRS = {
    "euclidean": EUC_DIR,
    "hyperbolic": HYP_DIR,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_domain_paths(base_dir: Path, domain: str) -> Dict[str, Path]:
    run_dir = base_dir / f"domain_{domain}"
    return {
        "run_dir": run_dir,
        "results": run_dir / "results.json",
        "calibration": run_dir / "calibration.json",
        "split_manifest": run_dir / "split_manifest.json",
        "training_log": run_dir / "training_log.csv",
    }


def ensure_expected_files():
    missing: List[str] = []
    for geom, base_dir in GEOMETRY_DIRS.items():
        for domain in DOMAINS:
            paths = collect_domain_paths(base_dir, domain)
            for key, p in paths.items():
                if key == "run_dir":
                    continue
                if not p.exists():
                    missing.append(f"{geom}/{domain}: {p}")
        summary = base_dir / "summary_results.csv"
        if not summary.exists():
            missing.append(str(summary))

    if missing:
        raise FileNotFoundError("Missing expected experiment files:\n" + "\n".join(missing))


def build_rows():
    master_rows = []
    appendix_rows = []
    threshold_modes = set()

    for geometry, base_dir in GEOMETRY_DIRS.items():
        for domain in DOMAINS:
            p = collect_domain_paths(base_dir, domain)
            results = json.loads(p["results"].read_text(encoding="utf-8"))
            calibration = json.loads(p["calibration"].read_text(encoding="utf-8"))

            threshold_mode = str(results.get("threshold_mode", ""))
            threshold_modes.add(threshold_mode)

            if threshold_mode == "calibrated_f1":
                primary_threshold = float(results["calibrated_threshold"])
            else:
                primary_threshold = float(results["default_threshold"])

            master_rows.append(
                {
                    "domain": domain,
                    "geometry": geometry,
                    "model": results.get("model", ""),
                    "method": results.get("method", ""),
                    "n_samples": int(results.get("n_samples", 0)),
                    "n_real": int(results.get("n_real", 0)),
                    "n_fake": int(results.get("n_fake", 0)),
                    "accuracy": float(results["accuracy"]),
                    "precision": float(results["precision"]),
                    "recall": float(results["recall"]),
                    "f1": float(results["f1"]),
                    "specificity": float(results["specificity"]),
                    "sensitivity": float(results["sensitivity"]),
                    "PPV": float(results["PPV"]),
                    "NPV": float(results["NPV"]),
                    "auroc": float(results["auroc"]),
                    "auprc": float(results["auprc"]),
                    "threshold_mode_primary": threshold_mode,
                    "primary_threshold": primary_threshold,
                    "default_threshold": float(results["default_threshold"]),
                    "calibrated_threshold": float(results["calibrated_threshold"]),
                    "fake_positive_if_high": bool(results["fake_positive_if_high"]),
                    "split_hash": str(results["split_hash"]),
                    "training_time_sec": float(results.get("training_time_sec", 0.0)),
                    "epochs": int(results.get("epochs", 0)),
                    "batch_size": int(results.get("batch_size", 0)),
                }
            )

            d = calibration["test_metrics_default"]
            appendix_rows.append(
                {
                    "domain": domain,
                    "geometry": geometry,
                    "threshold_policy": "default_threshold_appendix",
                    "default_threshold": float(calibration["default_threshold"]),
                    "accuracy": float(d["accuracy"]),
                    "precision": float(d["precision"]),
                    "recall": float(d["recall"]),
                    "f1": float(d["f1"]),
                    "specificity": float(d["specificity"]),
                    "sensitivity": float(d["sensitivity"]),
                    "PPV": float(d["PPV"]),
                    "NPV": float(d["NPV"]),
                    "auroc": float(d["auroc"]),
                    "auprc": float(d["auprc"]),
                }
            )

    return master_rows, appendix_rows, threshold_modes


def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_snapshot(snapshot_name: str) -> Path:
    snapshot_dir = SNAPSHOT_ROOT / snapshot_name
    if snapshot_dir.exists():
        raise FileExistsError(
            f"Snapshot already exists: {snapshot_dir}. Use a new snapshot name or remove existing one explicitly."
        )

    for geometry, base_dir in GEOMETRY_DIRS.items():
        for domain in DOMAINS:
            src_run = base_dir / f"domain_{domain}"
            dst_run = snapshot_dir / base_dir.name / f"domain_{domain}"
            dst_run.mkdir(parents=True, exist_ok=True)

            # Snapshot only core reproducibility/reporting artifacts.
            for fname in [
                "results.json",
                "calibration.json",
                "split_manifest.json",
                "training_log.csv",
                "summary_results.csv",
                "roc_curve.png",
                "pr_curve.png",
                "confusion_matrix.png",
                "confusion_matrix_default.png",
                "confusion_matrix_calibrated.png",
                "score_distribution.png",
                "loss_curve.png",
            ]:
                src = src_run / fname
                if fname == "summary_results.csv":
                    src = base_dir / fname
                if src.exists():
                    dst = dst_run / fname if fname != "summary_results.csv" else (snapshot_dir / base_dir.name / fname)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

    manifest = {
        "snapshot_name": snapshot_name,
        "created_on": str(date.today()),
        "source_dirs": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in GEOMETRY_DIRS.items()},
        "files": [],
    }

    for p in sorted(snapshot_dir.rglob("*")):
        if p.is_file():
            manifest["files"].append(
                {
                    "path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "sha256": sha256_file(p),
                    "size_bytes": p.stat().st_size,
                }
            )

    manifest_path = snapshot_dir / "baseline_snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return snapshot_dir


def write_policy(threshold_modes):
    policy_path = ANALYSIS_DIR / "reporting_policy.json"
    mode = "calibrated_f1" if "calibrated_f1" in threshold_modes else sorted(threshold_modes)[0]
    payload = {
        "primary_reporting_threshold_policy": mode,
        "appendix_threshold_policy": "default_threshold",
        "notes": [
            "Primary tables/charts must use the primary policy only.",
            "Default-threshold results are retained for appendix and sensitivity analysis.",
        ],
    }
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    ensure_expected_files()

    master_rows, appendix_rows, threshold_modes = build_rows()
    master_rows = sorted(master_rows, key=lambda r: (r["domain"], r["geometry"]))
    appendix_rows = sorted(appendix_rows, key=lambda r: (r["domain"], r["geometry"]))

    write_csv(ANALYSIS_DIR / "master_metrics_primary.csv", master_rows)
    write_csv(ANALYSIS_DIR / "appendix_default_threshold_metrics.csv", appendix_rows)
    write_policy(threshold_modes)

    snapshot_name = f"source_specific_ood_baseline_{date.today().isoformat()}"
    snapshot_dir = create_snapshot(snapshot_name)

    print("Built reporting bundle successfully")
    print(f"Master metrics: {ANALYSIS_DIR / 'master_metrics_primary.csv'}")
    print(f"Appendix metrics: {ANALYSIS_DIR / 'appendix_default_threshold_metrics.csv'}")
    print(f"Policy file: {ANALYSIS_DIR / 'reporting_policy.json'}")
    print(f"Snapshot: {snapshot_dir}")


if __name__ == "__main__":
    main()
