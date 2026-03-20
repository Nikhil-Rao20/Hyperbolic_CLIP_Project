from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset
from transformers import (
    CLIPModel,
    CLIPProcessor,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output if hasattr(output, "pooler_output") else output[0]


def _rel_path_to_label_source(rel_path: str) -> Tuple[int, str]:
    path = Path(rel_path)
    parts_lower = [p.lower() for p in path.parts]
    is_real = any("real" in p for p in parts_lower) and not any("fake" in p for p in parts_lower)
    label = 0 if is_real else 1

    stem = path.stem
    if "__" in stem:
        source = stem.split("__", 1)[0]
    else:
        source = "unknown"
        for key in ["cermep", "tcga", "upenn", "gan", "ldm", "mls_cermep", "mls_tcga", "mls_upenn", "mls"]:
            if any(key in p for p in parts_lower):
                source = key.upper() if key in {"gan", "ldm"} else key
                break

    return label, source


class ImagePathDataset(Dataset):
    def __init__(self, dataset_root: Path, rel_paths: Sequence[str]):
        self.dataset_root = dataset_root
        self.rel_paths = list(rel_paths)

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        rel = self.rel_paths[idx]
        abs_path = self.dataset_root / rel
        image = Image.open(abs_path).convert("RGB")
        label, source = _rel_path_to_label_source(rel)
        return image, label, source, rel


class PromptZeroShotModel:
    def __init__(self, model_id: str, model_type: str, device: torch.device, prompt_question: str):
        self.model_id = model_id
        self.model_type = model_type
        self.device = device
        self.prompt_question = prompt_question

        if self.model_type == "clip":
            self.model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_id)
        elif self.model_type == "instructblip":
            self.model = InstructBlipForConditionalGeneration.from_pretrained(model_id).to(device)
            self.processor = InstructBlipProcessor.from_pretrained(model_id)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def build_class_text_embeddings(self, real_prompts: List[str], fake_prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model_type != "clip":
            raise RuntimeError("Class text embeddings are only for CLIP models")

        real_inputs = self.processor(text=real_prompts, return_tensors="pt", padding=True)
        fake_inputs = self.processor(text=fake_prompts, return_tensors="pt", padding=True)
        real_inputs = {k: v.to(self.device) for k, v in real_inputs.items()}
        fake_inputs = {k: v.to(self.device) for k, v in fake_inputs.items()}

        real_feats = _to_tensor(self.model.get_text_features(**real_inputs))
        fake_feats = _to_tensor(self.model.get_text_features(**fake_inputs))
        real_mean = F.normalize(real_feats.mean(dim=0), dim=-1)
        fake_mean = F.normalize(fake_feats.mean(dim=0), dim=-1)
        return real_mean, fake_mean


@torch.no_grad()
def infer_clip_probs(
    zs_model: PromptZeroShotModel,
    dataset: ImagePathDataset,
    batch_size: int,
    real_emb: torch.Tensor,
    fake_emb: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels = []
    probs = []
    sources = []

    class_embs = torch.stack([real_emb, fake_emb], dim=0)  # [2, D]

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = [dataset[i] for i in range(start, end)]
        images = [x[0] for x in batch]
        lbs = [x[1] for x in batch]
        srcs = [x[2] for x in batch]

        inputs = zs_model.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(zs_model.device) for k, v in inputs.items()}

        image_features = _to_tensor(zs_model.model.get_image_features(**inputs))
        image_features = F.normalize(image_features.float(), dim=-1)

        logits = image_features @ class_embs.T
        probs_fake = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        labels.extend(lbs)
        probs.extend(probs_fake.tolist())
        sources.extend(srcs)

    return np.array(labels), np.array(probs), sources


@torch.no_grad()
def infer_instructblip_probs(
    zs_model: PromptZeroShotModel,
    dataset: ImagePathDataset,
    batch_size: int,
    max_new_tokens: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels = []
    probs = []
    sources = []

    # For InstructBLIP in zero-shot classification mode, we parse generated answer and map to hard score.
    # score(fake)=1 if output mentions fake/synthetic/artificial else 0.
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = [dataset[i] for i in range(start, end)]
        images = [x[0] for x in batch]
        lbs = [x[1] for x in batch]
        srcs = [x[2] for x in batch]

        prompts = [zs_model.prompt_question] * len(images)
        inputs = zs_model.processor(images=images, text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(zs_model.device) for k, v in inputs.items()}

        generated = zs_model.model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = zs_model.processor.batch_decode(generated, skip_special_tokens=True)

        for text in decoded:
            t = text.strip().lower()
            is_fake = ("fake" in t) or ("synthetic" in t) or ("artificial" in t) or ("generated" in t)
            probs.append(1.0 if is_fake else 0.0)

        labels.extend(lbs)
        sources.extend(srcs)

    return np.array(labels), np.array(probs, dtype=float), sources


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict:
    preds = (probs >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    try:
        out["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        out["auroc"] = 0.0
    try:
        out["auprc"] = float(average_precision_score(labels, probs))
    except ValueError:
        out["auprc"] = 0.0

    cm = confusion_matrix(labels, preds)
    out["confusion_matrix"] = cm.tolist()
    tn, fp = cm[0]
    fn, tp = cm[1]
    out["specificity"] = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
    out["sensitivity"] = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    out["balanced_mean_accuracy"] = round((out["specificity"] + out["sensitivity"]) / 2.0, 4)
    return out


def _save_confusion_matrix(labels: np.ndarray, probs: np.ndarray, out_path: Path):
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "test_set",
        "n_real",
        "n_fake",
        "auroc",
        "auprc",
        "accuracy",
        "f1",
        "precision",
        "sensitivity",
        "specificity",
        "balanced_mean_accuracy",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pure prompt-based zero-shot protocol evaluation")
    parser.add_argument("--config", type=str, default="configs/prompt_zero_shot_protocol.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = _load_config(cfg_path)

    dataset_root = Path(cfg["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root

    manifest_path = Path(cfg["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    output_root = Path(cfg.get("output_root", "experiments_zero_shot_protocol"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    run_name = cfg.get("run_name", f"prompt_zero_shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg.get("batch_size", 16))
    max_new_tokens = int(cfg.get("instructblip_max_new_tokens", 6))

    real_prompts = cfg.get("real_prompts", ["A Real MRI image"])
    fake_prompts = cfg.get("fake_prompts", ["A Fake MRI image"])
    instruct_prompt = cfg.get(
        "instruct_prompt",
        "Classify this brain MRI image as Real or Fake. Answer with one word: Real or Fake.",
    )

    summary_rows: List[Dict] = []

    for model_cfg in cfg.get("models", []):
        model_name = model_cfg["name"]
        model_id = model_cfg["model_id"]
        model_type = model_cfg.get("type", "clip")

        print(f"\\n[MODEL] {model_name} ({model_id})", flush=True)
        model_dir = run_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        zs_model = PromptZeroShotModel(
            model_id=model_id,
            model_type=model_type,
            device=device,
            prompt_question=instruct_prompt,
        )

        if model_type == "clip":
            real_emb, fake_emb = zs_model.build_class_text_embeddings(real_prompts, fake_prompts)
        else:
            real_emb = None
            fake_emb = None

        tests_payload = {}
        for test_name, test_spec in manifest["test_sets"].items():
            test_dir = model_dir / test_name
            test_dir.mkdir(parents=True, exist_ok=True)

            rel_paths = sorted(test_spec["real_ids"] + test_spec["fake_ids"])
            ds = ImagePathDataset(dataset_root, rel_paths)

            if model_type == "clip":
                labels, probs, _ = infer_clip_probs(
                    zs_model=zs_model,
                    dataset=ds,
                    batch_size=batch_size,
                    real_emb=real_emb,
                    fake_emb=fake_emb,
                )
            else:
                labels, probs, _ = infer_instructblip_probs(
                    zs_model=zs_model,
                    dataset=ds,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                )

            metrics = compute_metrics(labels, probs)
            _save_confusion_matrix(labels, probs, test_dir / "confusion_matrix.png")

            payload = {
                "model_name": model_name,
                "model_type": model_type,
                "test_set": test_name,
                "n_real": int(test_spec["n_real"]),
                "n_fake": int(test_spec["n_fake"]),
                "real_prompts": real_prompts if model_type == "clip" else None,
                "fake_prompts": fake_prompts if model_type == "clip" else None,
                "instruct_prompt": instruct_prompt if model_type == "instructblip" else None,
                "metrics": metrics,
                "decision_threshold": 0.5,
            }
            with (test_dir / "results.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            tests_payload[test_name] = payload

            summary_rows.append(
                {
                    "model_name": model_name,
                    "test_set": test_name,
                    "n_real": int(test_spec["n_real"]),
                    "n_fake": int(test_spec["n_fake"]),
                    "auroc": round(metrics["auroc"], 6),
                    "auprc": round(metrics["auprc"], 6),
                    "accuracy": round(metrics["accuracy"], 6),
                    "f1": round(metrics["f1"], 6),
                    "precision": round(metrics["precision"], 6),
                    "sensitivity": round(metrics["sensitivity"], 6),
                    "specificity": round(metrics["specificity"], 6),
                    "balanced_mean_accuracy": round(metrics["balanced_mean_accuracy"], 6),
                }
            )

        with (model_dir / "model_summary.json").open("w", encoding="utf-8") as f:
            json.dump({"model_name": model_name, "tests": tests_payload}, f, indent=2)

    write_summary_csv(summary_rows, run_dir / "final_prompt_zero_shot_summary.csv")
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"run_dir": run_dir.as_posix(), "n_rows": len(summary_rows)}, f, indent=2)

    print("\\n[INFO] Completed pure prompt-based zero-shot evaluation.")
    print("[INFO] Summary:", run_dir / "final_prompt_zero_shot_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
