"""Backbones that return a spatial feature map [B, C, H, W].

  CNNFeatureExtractor  : any timm CNN (resnet34 / resnet50 / convnext_tiny / densenet121 / vgg19 ...)
                         trunk (pretrained) + optional ProtoPNet-style 1x1 "add-on" layers
                         resnet34 @224 -> [B, 128, 7, 7]  (512 trunk channels -> 128 add-on)
  SwinFeatureExtractor : Swin-Tiny, last stage -> [B, 768, 7, 7]
  ViTFeatureExtractor  : DeiT-Small patch tokens -> [B, 384, 14, 14]

make_backbone(cfg) picks one from cfg.backbone, so the rest of the code never has to
care which one is used. split_param_groups(model) separates the PRETRAINED trunk from
everything else (add-on, prototypes, heads) for freezing / learning-rate groups.

Why an add-on layer for CNNs: the trunk output (e.g. ResNet layer4) is post-ReLU, i.e.
all-positive, and 512-2048 dimensional. A 1x1 conv maps it to a small prototype space
(ProtoPNet uses 64-128 channels). It is RANDOMLY initialised, so it must be trained at
the full learning rate and must not be frozen with the pretrained trunk: that is why it
is excluded from `pretrained_parameters()`.
"""
import timm
import torch
import torch.nn as nn


def _to_nchw(x, c):
    if x.dim() == 4 and x.shape[1] == c:      # already NCHW
        return x
    if x.dim() == 4 and x.shape[-1] == c:     # NHWC -> NCHW
        return x.permute(0, 3, 1, 2).contiguous()
    raise RuntimeError(f"Unexpected feature shape {tuple(x.shape)} for C={c}")


def make_addon(c_in, kind="linear", dim=128):
    """kind: 'linear'    1x1 conv, signed features (cosine uses its full [-1,1] range)   <- default
             'protopnet' conv-ReLU-conv-Sigmoid as in the original ProtoPNet (designed for
                         L2 distances; with cosine, all features are positive so the
                         similarities are compressed towards the top of the range)
             None        no add-on (prototypes live in the raw trunk space)"""
    if kind in (None, "none"):
        return nn.Identity(), c_in
    if kind == "linear":
        return nn.Conv2d(c_in, dim, kernel_size=1), dim
    if kind == "protopnet":
        return nn.Sequential(nn.Conv2d(c_in, dim, 1), nn.ReLU(inplace=True),
                             nn.Conv2d(dim, dim, 1), nn.Sigmoid()), dim
    raise ValueError(f"unknown add-on kind: {kind!r}")


class CNNFeatureExtractor(nn.Module):
    """timm CNN trunk (+ add-on). stage=None -> last stage (ResNet layer4, stride 32 -> 7x7 at 224).
    stage=3 on a ResNet gives layer3 (stride 16 -> 14x14: finer maps, less semantic)."""
    def __init__(self, name="resnet34", pretrained=True, stage=None, addon="linear", addon_dim=128):
        super().__init__()
        kw = dict(pretrained=pretrained, features_only=True)
        if stage is not None:
            kw["out_indices"] = (stage,)
        self.trunk = timm.create_model(name, **kw)
        self.trunk_channels = self.trunk.feature_info.channels()[-1]
        self.reduction = self.trunk.feature_info.reduction()[-1]
        self.addon, self.out_channels = make_addon(self.trunk_channels, addon, addon_dim)

    def pretrained_parameters(self):
        return self.trunk.parameters()

    def forward(self, x):
        f = _to_nchw(self.trunk(x)[-1], self.trunk_channels)
        return self.addon(f)                                   # [B, C', H, W]


class SwinFeatureExtractor(nn.Module):
    def __init__(self, name="swin_tiny_patch4_window7_224", pretrained=True, stage=3):
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=pretrained, features_only=True, out_indices=(stage,)
        )
        self.out_channels = self.backbone.feature_info.channels()[-1]

    def pretrained_parameters(self):
        return self.backbone.parameters()

    def forward(self, x):
        return _to_nchw(self.backbone(x)[-1], self.out_channels)   # [B, C, H, W]


class ViTFeatureExtractor(nn.Module):
    """DeiT / ViT patch tokens as a feature map (block=-1 -> last block)."""
    def __init__(self, name="deit_small_patch16_224", pretrained=True, block=-1, drop_path_rate=0.0):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                          drop_path_rate=drop_path_rate)
        n_blocks = len(self.backbone.blocks)
        block = n_blocks + block if block < 0 else block
        assert 0 <= block < n_blocks, f"block must be in [0, {n_blocks - 1}]"
        self.backbone.blocks = nn.Sequential(*list(self.backbone.blocks)[: block + 1])
        self.out_channels = self.backbone.embed_dim
        self.n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        ps = self.backbone.patch_embed.patch_size
        self.patch = ps[0] if isinstance(ps, (tuple, list)) else ps

    def pretrained_parameters(self):
        return self.backbone.parameters()

    def forward(self, x):
        B, _, H, W = x.shape
        tokens = self.backbone.forward_features(x)[:, self.n_prefix:, :]
        h, w = H // self.patch, W // self.patch
        if tokens.size(1) != h * w:
            raise RuntimeError(f"{tokens.size(1)} patch tokens but {H}x{W}/{self.patch} = {h}x{w}")
        return tokens.transpose(1, 2).reshape(B, self.out_channels, h, w).contiguous()


def _family(cfg):
    t = getattr(cfg, "backbone_type", None)
    if t:
        return t.lower()
    n = cfg.backbone.lower()
    if "swin" in n:
        return "swin"
    if n.startswith(("vit", "deit", "beit", "eva")):
        return "vit"
    return "cnn"


def make_backbone(cfg):
    """Factory. Family is inferred from cfg.backbone (or forced with cfg.backbone_type).

    CNN   : cfg.backbone='resnet34'   cfg.cnn_stage=None (last)   cfg.addon_type='linear'   cfg.addon_dim=128
    Swin  : cfg.backbone='swin_tiny_patch4_window7_224'           cfg.feature_stage=3
    ViT   : cfg.backbone='deit_small_patch16_224'                 cfg.feature_block=-1
    (every field except cfg.backbone is optional)
    """
    fam, name = _family(cfg), cfg.backbone
    pretrained = getattr(cfg, "pretrained", True)
    if fam == "cnn":
        return CNNFeatureExtractor(name, pretrained=pretrained,
                                   stage=getattr(cfg, "cnn_stage", None),
                                   addon=getattr(cfg, "addon_type", "linear"),
                                   addon_dim=getattr(cfg, "addon_dim", 128))
    if fam == "swin":
        return SwinFeatureExtractor(name, pretrained=pretrained, stage=getattr(cfg, "feature_stage", 3))
    if fam == "vit":
        return ViTFeatureExtractor(name, pretrained=pretrained, block=getattr(cfg, "feature_block", -1),
                                   drop_path_rate=getattr(cfg, "drop_path_rate", 0.0))
    raise ValueError(f"unknown backbone family {fam!r}")


def split_param_groups(model):
    """(pretrained_trunk_params, all_other_params). Use for the freeze schedule and LR groups."""
    trunk = list(model.backbone.pretrained_parameters())
    ids = {id(p) for p in trunk}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return trunk, rest