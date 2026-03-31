"""
Supervised CLIP ViT B16 Baseline
Binary cross-entropy classifier for supervised upper bound comparison.
Mirrors architecture and hyperparameters from one_class_svdd_clip_v2.py exactly.
"""
from __future__ import annotations

import argparse
import json
import csv
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, confusion_matrix, precision_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ManifestSupervisedDataset(Dataset):
    """
    Loads images from explicit path lists.
    Real images -> label 0, Fake images -> label 1.
    Returns raw RGB PIL images (CLIPProcessor handles preprocessing).
    Mirrors ImagePathDataset from run_one_class_svdd_clip_v2.py.
    """
    def __init__(self, dataset_root: Path, real_ids: list, fake_ids: list):
        self.dataset_root = dataset_root
        self.samples = []  # (rel_path, label)
        for rel in real_ids:
            self.samples.append((rel, 0))
        for rel in fake_ids:
            self.samples.append((rel, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel, label = self.samples[idx]
        image = Image.open(self.dataset_root / rel).convert("RGB")
        return image, label


def collate_fn(batch):
    """Collate function for DataLoader: return list of images and tensor of labels."""
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)


def split_real_fake(id_list: list):
    """Split a list of paths into real and fake based on /Real/ or /Fake/ in path."""
    real_ids = [p for p in id_list if "/Real/" in p.replace("\\", "/")]
    fake_ids = [p for p in id_list if "/Fake/" in p.replace("\\", "/")]
    return real_ids, fake_ids


def encode_image_features(clip_model, processor, images, device):
    """
    Takes a list of raw RGB PIL images.
    Passes through CLIPProcessor then extracts CLIP image features.
    Mirrors _encode_image_features (HuggingFace branch) from SVDD script.
    """
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features


class CLIPClassificationHead(nn.Module):
    """
    Binary classification head.
    Mirrors EuclideanProjectionHead: Linear->ReLU->Linear.
    Xavier uniform init, zeros bias. No dropout.
    """
    def __init__(self, input_dim: int = 512,
                 projection_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, num_classes),
        )
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_optimizer(clip_model, head, cfg, freeze_backbone: bool):
    """Build optimizer. Mirrors build_optimizer from SVDD exactly."""
    lr_backbone = float(cfg.get("lr_backbone", 1e-5))
    lr_head = float(cfg.get("lr_head", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-4))

    if freeze_backbone:
        return torch.optim.AdamW(
            head.parameters(),
            lr=lr_head,
            weight_decay=wd,
        )
    trainable_backbone = [p for p in clip_model.parameters()
                          if p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": trainable_backbone, "lr": lr_backbone},
            {"params": head.parameters(), "lr": lr_head},
        ],
        weight_decay=wd,
    )


def compute_metrics(labels, probs_fake):
    """
    Compute all metrics for classification.
    labels: list/array of 0/1 ground truth
    probs_fake: list/array of predicted probability for class 1 (fake)
    Returns dict of all 8 metrics.
    """
    labels = np.array(labels)
    probs_fake = np.array(probs_fake)

    auroc = float(roc_auc_score(labels, probs_fake))
    auprc = float(average_precision_score(labels, probs_fake))

    preds = (probs_fake >= 0.5).astype(int)
    acc = float(accuracy_score(labels, preds))
    bal_acc = float(balanced_accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, zero_division=0))
    prec = float(precision_score(labels, preds, zero_division=0))

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "precision": prec,
    }


def run_fold(fold: dict, manifest: dict, cfg: dict, freeze_backbone: bool,
             dataset_root: Path, device: torch.device, fold_dir: Path,
             fold_index: int):
    """
    Train and evaluate on a single fold.
    Returns (fold_test_results, best_val_auroc, log_rows)
    """
    batch_size = int(cfg.get("batch_size", 32))
    epochs = int(cfg.get("epochs", 10))
    projection_dim = int(cfg.get("projection_dim", 256))
    clip_model_name = cfg.get("clip_model_name", "openai/clip-vit-base-patch16")

    # --- STEP A: Build datasets ---
    train_real_ids = fold["train_ids"]
    train_fake_ids = manifest["supervised_baseline"]["fake_train_ids"]
    train_ds = ManifestSupervisedDataset(dataset_root, train_real_ids, train_fake_ids)

    val_real_ids, val_fake_ids = split_real_fake(fold["val_eval_ids"])
    val_ds = ManifestSupervisedDataset(dataset_root, val_real_ids, val_fake_ids)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0,
                            collate_fn=collate_fn)

    print(f"[INFO]   Train: {len(train_ds)} images "
          f"({len(train_real_ids)} real + {len(train_fake_ids)} fake)", flush=True)
    print(f"[INFO]   Val:   {len(val_ds)} images "
          f"({len(val_real_ids)} real + {len(val_fake_ids)} fake)", flush=True)

    # --- STEP B: Build model fresh per fold ---
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    head = CLIPClassificationHead(input_dim=512,
                                  projection_dim=projection_dim,
                                  num_classes=2).to(device)
    if freeze_backbone:
        for p in clip_model.parameters():
            p.requires_grad = False
    else:
        for p in clip_model.parameters():
            p.requires_grad = True

    # --- STEP C: Build optimizer and scheduler ---
    optimizer = build_optimizer(clip_model, head, cfg, freeze_backbone)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(steps_per_epoch * epochs, 1)
    )
    criterion = nn.CrossEntropyLoss()

    # --- STEP D: Training loop ---
    best_val_auroc = -1.0
    best_checkpoint = None
    log_rows = []

    for epoch in range(1, epochs + 1):

        # --- TRAIN ---
        clip_model.train()
        head.train()
        total_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad()

            # Encode via CLIPProcessor
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = clip_model.get_image_features(**inputs)

            logits = head(features)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(clip_model.parameters()) + list(head.parameters()),
                max_norm=1.0
            )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # --- VALIDATE ---
        clip_model.eval()
        head.eval()
        val_labels = []
        val_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.to(device)
                inputs = processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                features = clip_model.get_image_features(**inputs)
                logits = head(features)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_labels.extend(labels.cpu().numpy().tolist())
                val_probs.extend(probs.tolist())

        val_auroc = float(roc_auc_score(val_labels, val_probs))
        improved = val_auroc > best_val_auroc
        marker = " ✓ (best)" if improved else ""

        print(f"[INFO]   Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {avg_loss:.4f} | Val AUROC: {val_auroc:.4f}{marker}",
              flush=True)

        if improved:
            best_val_auroc = val_auroc
            best_checkpoint = {
                "clip_model": {k: v.cpu() for k, v in clip_model.state_dict().items()},
                "head": {k: v.cpu() for k, v in head.state_dict().items()},
                "fold_index": fold_index,
                "val_auroc": best_val_auroc,
                "epoch": epoch,
            }
            torch.save(best_checkpoint, fold_dir / "best_model.pth")

        log_rows.append({
            "fold": fold_index,
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_auroc": val_auroc,
        })

    # --- STEP E: Load best checkpoint and evaluate on 4 test sets ---
    if best_checkpoint is None:
        print(f"[ERROR] No valid checkpoint found for fold {fold_index}", flush=True)
        return {}, 0.0, log_rows

    clip_model.load_state_dict(best_checkpoint["clip_model"])
    head.load_state_dict(best_checkpoint["head"])
    clip_model.to(device).eval()
    head.to(device).eval()

    test_set_names = ["test_allfake", "test_gan", "test_ldm", "test_mls"]
    fold_test_results = {}

    for test_name in test_set_names:
        test_spec = manifest["test_sets"][test_name]
        test_ds = ManifestSupervisedDataset(
            dataset_root,
            real_ids=test_spec["real_ids"],
            fake_ids=test_spec["fake_ids"],
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=0,
                                 collate_fn=collate_fn)
        test_labels = []
        test_probs = []
        with torch.no_grad():
            for images, labels in test_loader:
                inputs = processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                features = clip_model.get_image_features(**inputs)
                logits = head(features)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                test_labels.extend(labels.numpy().tolist())
                test_probs.extend(probs.tolist())

        fold_test_results[test_name] = compute_metrics(test_labels, test_probs)

    return fold_test_results, best_val_auroc, log_rows


def main():
    parser = argparse.ArgumentParser(
        description="Supervised CLIP ViT B16 Baseline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/supervised_clip_baseline.yaml",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["linear_probe", "full_finetune"],
        required=True,
        help="Which variant to run: linear_probe or full_finetune",
    )
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    variant_cfg = cfg["variants"][args.variant]
    freeze_backbone = variant_cfg["freeze_backbone"]
    display_name = variant_cfg["display_name"]

    manifest_path = PROJECT_ROOT / cfg["protocol_manifest_path"]
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if "supervised_baseline" not in manifest:
        raise KeyError(
            "[ERROR] supervised_baseline key not found in manifest. "
            "Please run Step 1 (manifest update) first."
        )

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = PROJECT_ROOT / cfg["dataset_root"]
    output_dir = PROJECT_ROOT / cfg["output_root"] / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading manifest: {manifest_path}", flush=True)
    print(f"[INFO] Supervised fake train set: "
          f"{manifest['supervised_baseline']['total_fake_train']} images "
          f"({manifest['supervised_baseline']['n_ldm']} LDM + "
          f"{manifest['supervised_baseline']['n_mls']} MLS)", flush=True)
    print(f"[INFO] Variant: {args.variant} | Freeze backbone: {freeze_backbone}",
          flush=True)
    print(f"[INFO] Device: {device}", flush=True)

    all_fold_results = []
    all_log_rows = []
    best_fold_index = -1
    best_fold_val_auroc = -1.0

    for fold in manifest["cv_folds"]:
        fold_index = fold["fold_index"]
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[INFO] ── Fold {fold_index}/4 "
              f"──────────────────────────────────────────",
              flush=True)

        fold_results, fold_val_auroc, log_rows = run_fold(
            fold=fold,
            manifest=manifest,
            cfg=cfg,
            freeze_backbone=freeze_backbone,
            dataset_root=dataset_root,
            device=device,
            fold_dir=fold_dir,
            fold_index=fold_index,
        )

        all_fold_results.append(fold_results)
        all_log_rows.extend(log_rows)

        if fold_val_auroc > best_fold_val_auroc:
            best_fold_val_auroc = fold_val_auroc
            best_fold_index = fold_index

        print(f"[INFO] Fold {fold_index} | Best Val AUROC: {fold_val_auroc:.4f}",
              flush=True)
        print(f"[INFO] Fold {fold_index} | Test Results:", flush=True)
        for test_name, metrics in fold_results.items():
            print(f"       {test_name:<15} → "
                  f"AUROC: {metrics['auroc']:.4f} | "
                  f"ACC: {metrics['accuracy']:.4f} | "
                  f"F1: {metrics['f1']:.4f}", flush=True)

    # Cross-fold aggregation
    print(f"\n[INFO] ── All Folds Complete "
          f"─────────────────────────────────────────────",
          flush=True)

    test_set_names = ["test_allfake", "test_gan", "test_ldm", "test_mls"]
    metric_keys = ["auroc", "auprc", "accuracy", "balanced_accuracy",
                   "f1", "specificity", "sensitivity", "precision"]

    aggregated = {}
    for test_name in test_set_names:
        aggregated[test_name] = {}
        for metric in metric_keys:
            values = [fold_res[test_name][metric]
                      for fold_res in all_fold_results]
            aggregated[test_name][f"{metric}_mean"] = float(np.mean(values))
            aggregated[test_name][f"{metric}_std"] = float(np.std(values))

    # Print summary
    allfake = aggregated["test_allfake"]
    print(f"[INFO] Mean Results (test_allfake):", flush=True)
    print(f"       AUROC: {allfake['auroc_mean']:.4f} ± {allfake['auroc_std']:.4f} | "
          f"ACC: {allfake['accuracy_mean']:.4f} ± {allfake['accuracy_std']:.4f} | "
          f"F1: {allfake['f1_mean']:.4f} ± {allfake['f1_std']:.4f}", flush=True)

    # Save results_summary.json
    results_summary = {
        "variant": args.variant,
        "display_name": display_name,
        "freeze_backbone": freeze_backbone,
        "config": {
            "clip_model_name": cfg["clip_model_name"],
            "batch_size": cfg["batch_size"],
            "epochs": cfg["epochs"],
            "lr_backbone": cfg["lr_backbone"],
            "lr_head": cfg["lr_head"],
            "weight_decay": cfg["weight_decay"],
            "projection_dim": cfg["projection_dim"],
            "seed": cfg["seed"],
        },
        "best_fold": {
            "fold_index": best_fold_index,
            "val_auroc": float(best_fold_val_auroc),
        },
        "test_results": aggregated,
    }
    with open(output_dir / "results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    # Save fold_results.json
    with open(output_dir / "fold_results.json", "w") as f:
        json.dump(all_fold_results, f, indent=2)

    # Save training_log.csv
    if all_log_rows:
        with open(output_dir / "training_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["fold", "epoch", "train_loss", "val_auroc"]
            )
            writer.writeheader()
            writer.writerows(all_log_rows)

    print(f"[INFO] Results saved to {output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(str(e), flush=True)
        raise SystemExit(1)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
