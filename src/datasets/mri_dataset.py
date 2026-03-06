"""MRI Dataset loader for Real vs Synthetic classification."""

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# Label mapping: real → 0, fake → 1
LABEL_MAP = {"Real": 0, "Fake": 1}


def get_transforms(split: str = "train"):
    """Return transforms for the given split.

    All splits: grayscale → tensor → 3-channel → ImageNet normalisation.
    Train split additionally applies augmentations.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),            # [1, 224, 224]
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # → [3, 224, 224]
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            normalize,
        ])


class MRIDataset(Dataset):
    """PyTorch Dataset for the cleaned MRI real/fake dataset.

    Expected directory layout
    -------------------------
    root/
        train/ | val/ | test/
            real/
            fake/

    Filenames encode the source before the first ``__`` separator,
    e.g. ``cermep__sub-0023_...`` → source = ``cermep``.
    """

    def __init__(self, root: str, split: str = "train", transform=None,
                 include_sources=None, exclude_sources=None):
        self.root = Path(root) / split
        self.split = split
        self.transform = transform or get_transforms(split)
        self.samples = []  # list of (path, label)
        self.sources = []  # parallel list of source strings

        for class_name, label in LABEL_MAP.items():
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.glob("*.png")):
                # Extract source from filename (text before first "__")
                name = img_path.stem
                source = name.split("__")[0] if "__" in name else "unknown"

                # Apply source filters
                if include_sources and source not in include_sources:
                    continue
                if exclude_sources and source in exclude_sources:
                    continue

                self.samples.append((img_path, label))
                self.sources.append(source)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("L")  # ensure grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self):
        """Compute inverse-frequency weights for Binary CE (weight for pos class)."""
        labels = [s[1] for s in self.samples]
        n_real = labels.count(0)
        n_fake = labels.count(1)
        # pos_weight = n_negative / n_positive  (used by BCEWithLogitsLoss)
        if n_fake == 0:
            return torch.tensor([1.0])
        return torch.tensor([n_real / n_fake])

    def get_source_for_index(self, idx):
        return self.sources[idx]
