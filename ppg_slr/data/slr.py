"""Sign-language dataset loader (Arabic Sign Language "Mosl_alphabet", 32 classes).

NOTE: the filename stays `breakhis.py` for import compatibility - train.py,
run_ablations.py and the notebooks all do `from data.breakhis import
make_datasets`. This project was ported from BreakHis to sign language.

Leakage-free by construction via two physically separate folders under
`cfg.data_root`:
  - `cfg.train_subdir` : class-subfolder images -> stratified 80/20 train/val
  - `cfg.test_subdir`  : held-out test set (class subfolders OR a flat folder of
                         class-named files, e.g. `aleff.png`)

Transforms are ORIENTATION-PRESERVING. Unlike histopathology (flip/rotation
invariant), hand signs are orientation-sensitive, so NO horizontal or vertical
flips are used - only crops, small rotations, colour jitter and RandAugment.
"""
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff", ".webp"}
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def scan_folder(root: str, class_to_idx) -> Tuple[List[str], List[int]]:
    """Scan an image folder -> (paths, integer labels).

    Handles a class-subfolder layout (root/<class>/<imgs>) or a flat folder of
    class-named files (root/<class>*.<ext>). Longest-prefix match disambiguates
    names such as 'al' vs 'aleff'.
    """
    root = Path(root)
    if not root.exists():
        raise RuntimeError(f"Dataset folder not found: {root}")
    classes = list(class_to_idx)
    by_len = sorted(classes, key=len, reverse=True)
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    paths, labels = [], []
    if subdirs:                                   # class-subfolder layout
        for cls in classes:
            cdir = root / cls
            if not cdir.is_dir():
                continue
            for f in sorted(cdir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    paths.append(str(f))
                    labels.append(class_to_idx[cls])
    else:                                         # flat layout: label from filename
        cset = set(classes)
        for f in sorted(root.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                stem = f.stem.lower()
                lbl = stem if stem in cset else next(
                    (c for c in by_len if stem.startswith(c.lower())), None)
                if lbl is not None:
                    paths.append(str(f))
                    labels.append(class_to_idx[lbl])
    if not paths:
        raise RuntimeError(f"No images found under {root}.")
    return paths, labels


def build_transforms(image_size: int, train: bool):
    """Orientation-preserving transforms (NO flips: signs are orientation-sensitive)."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.0),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    resize = int(round(image_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


class SignDataset(Dataset):
    def __init__(self, paths, labels, image_size, train):
        self.paths = list(paths)
        self.labels = list(labels)
        self.tf = build_transforms(image_size, train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), int(self.labels[i])


def make_datasets(cfg):
    """{train,val,test}: 80/20 split of the train folder + the separate test folder."""
    class_to_idx = {c: i for i, c in enumerate(cfg.class_names)}   # class_names is sorted
    train_root = os.path.join(cfg.data_root, cfg.train_subdir)
    test_root = os.path.join(cfg.data_root, cfg.test_subdir)

    tr_paths, tr_labels = scan_folder(train_root, class_to_idx)
    te_paths, te_labels = scan_folder(test_root, class_to_idx)

    p_tr, p_va, y_tr, y_va = train_test_split(
        tr_paths, tr_labels, test_size=cfg.val_frac, stratify=tr_labels,
        random_state=cfg.seed)

    return {
        "train": SignDataset(p_tr, y_tr, cfg.image_size, train=True),
        "val": SignDataset(p_va, y_va, cfg.image_size, train=False),
        "test": SignDataset(te_paths, te_labels, cfg.image_size, train=False),
    }
