from __future__ import annotations

import argparse
import subprocess
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"

WINCLIP_REPO = "https://github.com/mala-lab/WinCLIP.git"
ANOMALYCLIP_REPO = "https://github.com/zqhang/AnomalyCLIP.git"

WINCLIP_DIR = THIRD_PARTY / "WinCLIP"
ANOMALYCLIP_DIR = THIRD_PARTY / "AnomalyCLIP"

WINCLIP_CHECKPOINT_URL = (
    "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
    "vit_b_16_plus_240-laion400m_e31-8fb26589.pt"
)
WINCLIP_CHECKPOINT = WINCLIP_DIR / "vit_b_16_plus_240-laion400m_e31-8fb26589.pt"
ANOMALYCLIP_CHECKPOINT = ANOMALYCLIP_DIR / "checkpoints" / "9_12_4_multiscale_visa" / "epoch_15.pth"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _ensure_repo(path: Path, url: str, depth: int | None) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    if depth is not None:
        cmd.extend(["--depth", str(depth)])
    cmd.extend([url, path.as_posix()])
    _run(cmd)


def _ensure_winclip_datasets_package_marker() -> None:
    marker = WINCLIP_DIR / "datasets" / "__init__.py"
    if marker.exists():
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        "# Auto-created by setup_official_baselines.py for deterministic local imports.\n",
        encoding="utf-8",
    )


def _ensure_winclip_checkpoint(download_if_missing: bool) -> None:
    WINCLIP_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if WINCLIP_CHECKPOINT.exists():
        return
    if not download_if_missing:
        raise FileNotFoundError(
            f"WinCLIP checkpoint missing at {WINCLIP_CHECKPOINT}. "
            "Re-run without --skip-checkpoint-download or place the file manually."
        )
    urllib.request.urlretrieve(WINCLIP_CHECKPOINT_URL, WINCLIP_CHECKPOINT.as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up official WinCLIP and AnomalyCLIP baseline dependencies")
    parser.add_argument("--clone-depth", type=int, default=1, help="Depth for git clone (use 0 for full history)")
    parser.add_argument(
        "--skip-checkpoint-download",
        action="store_true",
        help="Do not download WinCLIP checkpoint if missing",
    )
    args = parser.parse_args()

    depth = None if args.clone_depth == 0 else args.clone_depth

    _ensure_repo(WINCLIP_DIR, WINCLIP_REPO, depth)
    _ensure_repo(ANOMALYCLIP_DIR, ANOMALYCLIP_REPO, depth)
    _ensure_winclip_datasets_package_marker()
    _ensure_winclip_checkpoint(download_if_missing=not args.skip_checkpoint_download)

    if not ANOMALYCLIP_CHECKPOINT.exists():
        print(
            "Warning: AnomalyCLIP checkpoint not found at "
            f"{ANOMALYCLIP_CHECKPOINT}.\n"
            "Please obtain the official checkpoint and place it at that path "
            "or override --checkpoint-path when running the AnomalyCLIP baseline."
        )

    print("Official baseline setup completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
