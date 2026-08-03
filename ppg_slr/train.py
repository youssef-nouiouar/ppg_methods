"""Train a single model on BreakHis.

Examples
--------
# the proposed model
python train.py --exp_name ppg_swint

# baselines (exp_name prefix selects the architecture)
python train.py --exp_name plain
python train.py --exp_name proto          # deterministic prototypes (HierViT-style)
python train.py --exp_name mcdropout

# a quick smoke test on CPU without touching real data
python train.py --smoke
"""
import argparse
import dataclasses
import json
import os
import random

import numpy as np
import torch

from config import Config
from models.backbone import SwinFeatureExtractor
from models.baselines import build_model
from engine.trainer import train


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        if f.type is bool:
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"),
                           default=None)
        else:
            p.add_argument(f"--{f.name}", type=type(f.default), default=None)
    p.add_argument("--smoke", action="store_true", help="tiny run on synthetic data")
    return p.parse_args()


def apply_overrides(cfg, args):
    for f in dataclasses.fields(Config):
        v = getattr(args, f.name)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg


def smoke_test(cfg):
    """Verify the whole model wiring on random tensors (no data, no GPU needed)."""
    cfg.device = "cpu"
    cfg.pretrained = False
    backbone = SwinFeatureExtractor(cfg.backbone, pretrained=False, stage=cfg.feature_stage)
    model = build_model(cfg, backbone).to(cfg.device)
    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    out = model(x)
    print("smoke OK -> logits", tuple(out["logits"].shape),
          "| keys:", list(out.keys()))


def main():
    args = parse_args()
    cfg = apply_overrides(Config(), args)
    set_seed(cfg.seed)

    if args.smoke:
        smoke_test(cfg)
        return

    if not torch.cuda.is_available():
        cfg.device = "cpu"

    from ppg_slr.data.slr import make_datasets
    datasets = make_datasets(cfg)
    print({k: len(v) for k, v in datasets.items()})

    backbone = SwinFeatureExtractor(cfg.backbone, pretrained=cfg.pretrained,
                                    stage=cfg.feature_stage)
    model = build_model(cfg, backbone)
    model, test_metrics = train(model, datasets, cfg)

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, f"{cfg.exp_name}_seed{cfg.seed}.json"), "w") as fh:
        json.dump({"config": dataclasses.asdict(cfg), "test": test_metrics}, fh, indent=2)


if __name__ == "__main__":
    main()
