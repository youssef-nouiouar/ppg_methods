"""Backbones that return a spatial feature map [B, C, H, W].

  SwinFeatureExtractor : Swin-Tiny, last stage  -> [B, 768,  7,  7]   (unchanged)
  ViTFeatureExtractor  : DeiT-Small (any plain timm ViT/DeiT)
                         -> [B, 384, 14, 14] at 224x224 input

make_backbone(cfg) picks one from cfg.backbone, so the rest of the code
(PPG layer, baselines, notebook) never has to care which one is used.

Why ViTFeatureExtractor does NOT use timm's `features_only`:
  older timm versions refuse features_only for plain ViTs, and newer ones
  change the return layout. Instead we call `forward_features` (token
  sequence [B, prefix + N, C]), drop the prefix tokens (CLS, and the
  distillation token for *_distilled models) and fold the N patch tokens
  back into an H x W grid.
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


class ViTFeatureExtractor(nn.Module):
    """DeiT / ViT patch tokens as a feature map.

    block : index of the transformer block whose output is used.
            -1 (default) = last block.  Plain supervised ViTs only train the CLS
            token for classification, so the patch tokens of the LAST block can be
            less spatially meaningful than those of an earlier block (e.g. 9 or 10
            for the 12-block DeiT-Small). Treat `block` as a hyper-parameter.
    drop_path_rate : stochastic depth while fine-tuning (0.0 = off, as in the Swin extractor).
    """
    def __init__(self, name="deit_small_patch16_224", pretrained=True, block=-1,
                 drop_path_rate=0.0):
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate
        )
        n_blocks = len(self.backbone.blocks)
        block = n_blocks + block if block < 0 else block
        assert 0 <= block < n_blocks, f"block must be in [0, {n_blocks - 1}]"
        # keep only the blocks we need (faster, and no unused parameters)
        self.backbone.blocks = nn.Sequential(*list(self.backbone.blocks)[: block + 1])

        self.out_channels = self.backbone.embed_dim                    # 384 for DeiT-Small
        self.n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)  # 1 (CLS) or 2 (distilled)
        ps = self.backbone.patch_embed.patch_size
        self.patch = ps[0] if isinstance(ps, (tuple, list)) else ps    # 16

    def forward(self, x):
        B, _, H, W = x.shape
        tokens = self.backbone.forward_features(x)           # [B, prefix + N, C]
        tokens = tokens[:, self.n_prefix:, :]                # drop CLS (and dist) token(s)
        h, w = H // self.patch, W // self.patch
        if tokens.size(1) != h * w:
            raise RuntimeError(
                f"{tokens.size(1)} patch tokens but input {H}x{W} / patch {self.patch} "
                f"gives {h}x{w}={h * w}. Use the model's native resolution (224).")
        return tokens.transpose(1, 2).reshape(B, self.out_channels, h, w).contiguous()


def make_backbone(cfg):
    """Factory: Swin if cfg.backbone contains 'swin', otherwise a plain ViT/DeiT.

    Optional cfg fields (all have defaults):
      cfg.feature_stage  Swin stage index                (default 3)
      cfg.feature_block  ViT/DeiT block index, -1 = last (default -1)
      cfg.drop_path_rate ViT/DeiT stochastic depth        (default 0.0)
    """
    name = cfg.backbone
    pretrained = getattr(cfg, "pretrained", True)
    if "swin" in name.lower():
        return SwinFeatureExtractor(name, pretrained=pretrained,
                                    stage=getattr(cfg, "feature_stage", 3))
    return ViTFeatureExtractor(name, pretrained=pretrained,
                               block=getattr(cfg, "feature_block", -1),
                               drop_path_rate=getattr(cfg, "drop_path_rate", 0.0))