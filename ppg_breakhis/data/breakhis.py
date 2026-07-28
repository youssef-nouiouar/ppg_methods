"""BreakHis dataset loader with a PATIENT-LEVEL split.

The single biggest mistake with BreakHis is splitting by image: the same
patient's slides then appear in both train and test, and accuracy becomes
meaningless. BreakHis filenames encode the patient/slide id, e.g.

    SOB_B_TA-14-4659-400-001.png
        |  |     |  |    |   |
        |  |     |  |    |   +-- sequence number
        |  |     |  |    +------ magnification (40 / 100 / 200 / 400)
        |  |     |  +----------- slide id      -> used as the PATIENT group
        |  |     +-------------- year
        |  +-------------------- tumor subtype (TA, DC, ...)
        +----------------------- class: B (benign) or M (malignant)

We group by "<year>-<slideid>" and split those groups, so no patient leaks.
"""
import os
import glob
import re
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit


_FNAME_RE = re.compile(
    r"SOB_(?P<cls>[BM])_[A-Za-z]+-(?P<year>\d+)-(?P<slide>[A-Za-z0-9]+)-(?P<mag>\d+)-\d+"
)


def _parse(fname: str):
    """Return (class_idx, patient_group, magnification) or None if unparseable."""
    m = _FNAME_RE.search(os.path.basename(fname))
    if m is None:
        return None
    cls = 0 if m.group("cls") == "B" else 1        # benign=0, malignant=1
    patient = f"{m.group('year')}-{m.group('slide')}"
    mag = m.group("mag") + "X"
    return cls, patient, mag


def scan_breakhis(root: str, magnification: str) -> Tuple[List[str], List[int], List[str]]:
    """Walk the BreakHis folder and collect paths/labels/patient-groups."""
    paths, labels, groups = [], [], []
    for p in glob.glob(os.path.join(root, "**", "*.png"), recursive=True):
        parsed = _parse(p)
        if parsed is None:
            continue
        cls, patient, mag = parsed
        if mag != magnification:
            continue
        paths.append(p)
        labels.append(cls)
        groups.append(patient)
    if not paths:
        raise RuntimeError(
            f"No {magnification} images found under {root}. "
            "Check data_root and that filenames follow the SOB_* convention."
        )
    return paths, labels, groups


def patient_level_split(paths, labels, groups, val_frac, test_frac, seed):
    """Two nested GroupShuffleSplits -> train / val / test with disjoint patients."""
    paths = np.array(paths)
    labels = np.array(labels)
    groups = np.array(groups)

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(paths, labels, groups))

    val_ratio = val_frac / (1.0 - test_frac)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_rel, va_rel = next(
        gss_val.split(paths[trainval_idx], labels[trainval_idx], groups[trainval_idx])
    )
    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[va_rel]

    # sanity: no patient overlap
    for a, b in [(train_idx, val_idx), (train_idx, test_idx), (val_idx, test_idx)]:
        assert set(groups[a]).isdisjoint(set(groups[b])), "patient leak between splits!"

    return {
        "train": (paths[train_idx], labels[train_idx]),
        "val": (paths[val_idx], labels[val_idx]),
        "test": (paths[test_idx], labels[test_idx]),
    }


_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


class BreakHisDataset(Dataset):
    def __init__(self, paths, labels, image_size, train):
        self.paths = list(paths)
        self.labels = list(labels)
        self.tf = build_transforms(image_size, train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.tf(img)
        y = int(self.labels[i])
        return x, y


def make_datasets(cfg):
    """Convenience: returns train/val/test BreakHisDataset objects from a Config."""
    paths, labels, groups = scan_breakhis(cfg.data_root, cfg.magnification)
    split = patient_level_split(paths, labels, groups,
                                cfg.val_frac, cfg.test_frac, cfg.seed)
    ds = {
        name: BreakHisDataset(p, y, cfg.image_size, train=(name == "train"))
        for name, (p, y) in split.items()
    }
    return ds
