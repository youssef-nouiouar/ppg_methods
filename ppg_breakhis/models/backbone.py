"""Swin-Tiny backbone that returns a spatial feature map [B, C, H, W].

timm's `features_only` output layout differs across versions (some return
channels-last NHWC for Swin). We detect the channel dim and normalise to NCHW
so the rest of the code never has to care.
"""
import timm
import torch
import torch.nn as nn


class SwinFeatureExtractor(nn.Module):
    def __init__(self, name="swin_tiny_patch4_window7_224", pretrained=True, stage=3):
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=pretrained, features_only=True, out_indices=(stage,)
        )
        self.out_channels = self.backbone.feature_info.channels()[-1]

    @staticmethod
    def _to_nchw(x, c):
        # already NCHW
        if x.dim() == 4 and x.shape[1] == c:
            return x
        # NHWC -> NCHW
        if x.dim() == 4 and x.shape[-1] == c:
            return x.permute(0, 3, 1, 2).contiguous()
        raise RuntimeError(f"Unexpected feature shape {tuple(x.shape)} for C={c}")

    def forward(self, x):
        feats = self.backbone(x)[-1]
        return self._to_nchw(feats, self.out_channels)   # [B, C, H, W]
