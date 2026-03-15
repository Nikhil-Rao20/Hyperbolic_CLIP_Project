from __future__ import annotations

import argparse
import zipfile
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def add_path(zf: zipfile.ZipFile, path: Path):
    if path.is_file():
        zf.write(path, arcname=str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
        return
    for p in path.rglob("*"):
        if p.is_file():
            zf.write(p, arcname=str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"))


def main():
    parser = argparse.ArgumentParser(description="Package source-specific OOD artifacts into a zip archive")
    parser.add_argument(
        "--output",
        type=str,
        default=f"experiments/source_specific_ood_bundle_{date.today().isoformat()}.zip",
    )
    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    include_paths = [
        PROJECT_ROOT / "experiments" / "source_specific_ood_analysis",
        PROJECT_ROOT / "experiments" / "source_specific_ood_euclidean_reruns",
        PROJECT_ROOT / "experiments" / "source_specific_ood_hyperbolic_reruns",
        PROJECT_ROOT / "docs" / "internal" / "source_specific_ood_results_audit.md",
        PROJECT_ROOT / "docs" / "internal" / "source_specific_ood_professor_brief.md",
        PROJECT_ROOT / "configs" / "source_specific_ood_euclidean.yaml",
        PROJECT_ROOT / "configs" / "source_specific_ood_hyperbolic.yaml",
        PROJECT_ROOT / "scripts" / "generate_source_specific_manifests.py",
        PROJECT_ROOT / "scripts" / "run_source_specific_multi_seed.py",
        PROJECT_ROOT / "scripts" / "analyze_source_specific_multi_seed.py",
        PROJECT_ROOT / "scripts" / "build_source_specific_reporting_bundle.py",
    ]

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in include_paths:
            if p.exists():
                add_path(zf, p)

    print(f"Created archive: {out_path}")


if __name__ == "__main__":
    main()
