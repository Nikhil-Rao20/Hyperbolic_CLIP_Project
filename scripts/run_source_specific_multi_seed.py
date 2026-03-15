from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = Path(sys.executable)

SCRIPT_BY_GEOMETRY = {
    "euclidean": PROJECT_ROOT / "scripts" / "run_source_specific_euclidean_ood.py",
    "hyperbolic": PROJECT_ROOT / "scripts" / "run_source_specific_hyperbolic_ood.py",
}
CFG_BY_GEOMETRY = {
    "euclidean": PROJECT_ROOT / "configs" / "source_specific_ood_euclidean.yaml",
    "hyperbolic": PROJECT_ROOT / "configs" / "source_specific_ood_hyperbolic.yaml",
}


def run_cmd(cmd, cwd: Path):
    print(" ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_seed(geometry: str, seed: int):
    cfg_path = CFG_BY_GEOMETRY[geometry]
    script_path = SCRIPT_BY_GEOMETRY[geometry]

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_exp = str(cfg.get("experiment_dir", f"experiments/source_specific_ood_{geometry}_reruns"))
    cfg["seed"] = int(seed)
    cfg["experiment_dir"] = f"{base_exp}/seed_{seed}"

    with tempfile.TemporaryDirectory(prefix=f"ood_{geometry}_seed_{seed}_") as td:
        temp_cfg = Path(td) / f"{geometry}_seed_{seed}.yaml"
        with open(temp_cfg, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        cmd = [str(PYTHON_EXE), str(script_path), "--config", str(temp_cfg)]
        run_cmd(cmd, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run source-specific OOD experiments across multiple seeds")
    parser.add_argument(
        "--geometry",
        type=str,
        default="both",
        choices=["euclidean", "hyperbolic", "both"],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 777],
        help="Seed list for repeated runs",
    )
    args = parser.parse_args()

    geometries = ["euclidean", "hyperbolic"] if args.geometry == "both" else [args.geometry]

    for geometry in geometries:
        for seed in args.seeds:
            print(f"\n=== Running {geometry} seed={seed} ===")
            run_seed(geometry, seed)

    print("\nMulti-seed run complete.")


if __name__ == "__main__":
    main()
