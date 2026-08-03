"""Baselines. Each isolates one of PPG's two claims.

  PlainSwin        : no prototypes, no uncertainty  -> is any machinery needed?
  DeterministicProto: prototypes + hard max, no uncertainty -> does the
                       PROBABILISTIC part beat plain prototypes?
  MCDropoutViT     : uncertainty but no prototypes -> does the PROTOTYPE part
                       add anything over plain uncertainty?

All return the same dict shape as PPGSwinT for a uniform training/eval loop.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainSwin(nn.Module):
    def __init__(self, cfg, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.out_channels, cfg.num_classes)

    def forward(self, x):
        feat = self.backbone(x)                       # [B, C, H, W]
        pooled = feat.mean(dim=(2, 3))                # global average pool
        return {"logits": self.head(pooled)}


class DeterministicProto(nn.Module):
    """ProtoPNet / HierViT-style: cosine prototypes + hard max, learnable head."""
    def __init__(self, cfg, backbone):
        super().__init__()
        self.backbone = backbone
        P = cfg.total_protos()
        self.prototypes = nn.Parameter(torch.randn(P, backbone.out_channels) * 0.02)
        self.head = nn.Linear(P, cfg.num_classes, bias=False)

    def forward(self, x):
        feat = F.normalize(self.backbone(x), dim=1)
        p = F.normalize(self.prototypes, dim=1)
        S = torch.einsum("pc,bchw->bphw", p, feat)    # [B, P, H, W]
        act = S.flatten(2).max(dim=2).values          # hard max pooling
        return {"logits": self.head(F.relu(act)), "alpha": S}


class MCDropoutViT(nn.Module):
    """Backbone + dropout + linear head; M passes at eval for uncertainty."""
    def __init__(self, cfg, backbone):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.drop = nn.Dropout(cfg.mc_dropout_p)
        self.head = nn.Linear(backbone.out_channels, cfg.num_classes)

    def _once(self, feat):
        pooled = self.drop(feat.mean(dim=(2, 3)))
        return self.head(pooled)

    def forward(self, x):
        feat = self.backbone(x)
        M = self.cfg.mc_samples if not self.training else 1
        logits = torch.stack([self._once(feat) for _ in range(max(M, 1))], 0).mean(0)
        return {"logits": logits}


def build_model(cfg, backbone):
    """Factory used by train.py."""
    name = cfg.exp_name.lower()
    if name.startswith("plain"):
        return PlainSwin(cfg, backbone)
    if name.startswith("proto") or name.startswith("hiervit"):
        return DeterministicProto(cfg, backbone)
    if name.startswith("mcdropout"):
        return MCDropoutViT(cfg, backbone)
    # default -> the proposed model
    from .ppg import PPGSwinT
    return PPGSwinT(cfg, backbone)
