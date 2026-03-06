"""Inference script for ResNet50 Baseline.

Usage
-----
Single image:
    python src/inference_resnet.py --model experiments/resnet_baseline/best_model.pth --image path/to/image.png

Batch folder:
    python src/inference_resnet.py --model experiments/resnet_baseline/best_model.pth --folder path/to/folder/
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.mri_dataset import get_transforms  # noqa: E402
from src.models.resnet50_baseline import build_resnet50_baseline  # noqa: E402

CLASS_NAMES = {0: "real", 1: "fake"}


def load_model(model_path: str, device: torch.device):
    model = build_resnet50_baseline(pretrained=False)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    return model


def predict_single(model, image_path: str, device: torch.device):
    transform = get_transforms("test")
    image = Image.open(image_path).convert("L")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor).squeeze()
        prob = torch.sigmoid(logit).item()

    predicted_label = 1 if prob >= 0.5 else 0
    return CLASS_NAMES[predicted_label], prob


def predict_folder(model, folder_path: str, device: torch.device):
    folder = Path(folder_path)
    results = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for img_path in sorted(folder.glob(ext)):
            cls, prob = predict_single(model, str(img_path), device)
            results.append({"file": img_path.name, "prediction": cls,
                            "probability_fake": round(prob, 4)})
    return results


def main():
    parser = argparse.ArgumentParser(description="ResNet50 baseline inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to best_model.pth")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image")
    parser.add_argument("--folder", type=str, default=None,
                        help="Path to a folder of images")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide --image or --folder.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    if args.image:
        cls, prob = predict_single(model, args.image, device)
        print(f"Prediction: {cls}  (P(fake) = {prob:.4f})")

    if args.folder:
        results = predict_folder(model, args.folder, device)
        print(f"\n{'File':<60s} {'Prediction':<12s} {'P(fake)'}")
        print("-" * 80)
        for r in results:
            print(f"{r['file']:<60s} {r['prediction']:<12s} "
                  f"{r['probability_fake']:.4f}")
        print(f"\nTotal: {len(results)} images")


if __name__ == "__main__":
    main()
