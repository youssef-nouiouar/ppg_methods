"""Run the ablation grid + baselines over several seeds and print a summary table.

Each ablation isolates one design choice flagged in review. Run:

    python run_ablations.py --seeds 0 1 2
"""
import argparse
import copy
import dataclasses
import json
import os

import numpy as np
import torch

from config import Config
from models.backbone import SwinFeatureExtractor
from models.baselines import build_model
from engine.trainer import train
from train import set_seed
from data.breakhis import make_datasets


# name -> dict of Config overrides. exp_name prefix picks the architecture.
ABLATIONS = {
    # --- baselines ---
    "plain_swin":        dict(exp_name="plain"),
    "proto_det":         dict(exp_name="proto"),
    "mcdropout_vit":     dict(exp_name="mcdropout"),
    # --- proposed model, full ---
    "ppg_full":          dict(exp_name="ppg_swint"),
    # --- ablations on the proposed model ---
    "ppg_gate_original": dict(exp_name="ppg_gateorig", gate_direction="original"),
    "ppg_no_g_gate":     dict(exp_name="ppg_nog", use_uncertainty_gate=False),
    "ppg_no_z_gate":     dict(exp_name="ppg_noz", use_bernoulli_gate=False),
    "ppg_maxpool":       dict(exp_name="ppg_maxpool", pooling="maxpool"),
    "ppg_M1":            dict(exp_name="ppg_m1", mc_samples=1),
    "ppg_M3":            dict(exp_name="ppg_m3", mc_samples=3),
    "ppg_learnable_head":dict(exp_name="ppg_learnhead", head="learnable"),
}


def run_one(base_cfg, overrides, datasets, seed):
    cfg = copy.deepcopy(base_cfg)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.seed = seed
    set_seed(seed)
    backbone = SwinFeatureExtractor(cfg.backbone, pretrained=cfg.pretrained,
                                    stage=cfg.feature_stage)
    model = build_model(cfg, backbone)
    _, test = train(model, datasets, cfg, verbose=False)
    return test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data_root", type=str, default=None)
    args = ap.parse_args()

    base = Config()
    if args.data_root:
        base.data_root = args.data_root
    if not torch.cuda.is_available():
        base.device = "cpu"

    # build datasets once (patient split is seeded by base.seed; keep fixed across
    # ablations so differences come from the METHOD, not the split)
    datasets = make_datasets(base)

    results = {name: [] for name in ABLATIONS}
    for name, ov in ABLATIONS.items():
        for seed in args.seeds:
            m = run_one(base, ov, datasets, seed)
            results[name].append(m)
            print(f"{name:20s} seed={seed} acc={m['accuracy']:.3f} "
                  f"auc={m['auc']:.3f} ece={m['ece']:.3f}")

    # summary: mean +/- std
    print("\n=== SUMMARY (mean +/- std over seeds) ===")
    header = f"{'ablation':20s} {'acc':>14s} {'auc':>14s} {'ece':>14s}"
    print(header)
    print("-" * len(header))
    summary = {}
    for name, runs in results.items():
        row = {}
        line = f"{name:20s}"
        for key in ("accuracy", "auc", "ece"):
            vals = np.array([r[key] for r in runs], dtype=float)
            row[key] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
            line += f" {row[key][0]:.3f}+/-{row[key][1]:.3f}"
        summary[name] = row
        print(line)

    os.makedirs(base.out_dir, exist_ok=True)
    with open(os.path.join(base.out_dir, "ablation_summary.json"), "w") as fh:
        json.dump({"raw": results, "summary": summary}, fh, indent=2)


if __name__ == "__main__":
    main()
