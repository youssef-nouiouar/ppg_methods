"""Pointing Game interpretability metric.

IMPORTANT CAVEAT FOR BreakHis: this metric needs a ground-truth localisation
(a bounding box or segmentation mask of the pathology). BreakHis provides only
image-level benign/malignant labels - it has NO such boxes. So Pointing Game is
*not directly runnable on BreakHis* out of the box.

Options:
  (a) skip Pointing Game for the BreakHis pilot (report only acc/AUC/ECE), or
  (b) evaluate it on a dataset that ships masks (e.g. ISIC provides lesion
      segmentations), or
  (c) have an expert annotate a small held-out subset of BreakHis.

The implementation below is generic: give it upsampled prototype attention maps
and binary masks and it computes the hit rate. It is here so the code is ready
the moment masks are available.
"""
import numpy as np
import torch
import torch.nn.functional as F


def upsample_attention(alpha, size):
    """alpha: [B, P, h, w] -> [B, P, H, W] bilinearly upsampled to `size`=(H,W)."""
    return F.interpolate(alpha, size=size, mode="bilinear", align_corners=False)


def pointing_game(alpha_upsampled, masks, class_of_proto, labels):
    """Hit if the argmax location of the most-activated prototype for the true
    class falls inside the ground-truth mask.

    alpha_upsampled : [B, P, H, W] tensor
    masks           : [B, H, W] binary tensor (1 = pathology region)
    class_of_proto  : [P] long tensor mapping prototype -> class index
    labels          : [B] long tensor of true classes
    returns         : hit rate in [0, 1]
    """
    B, P, H, W = alpha_upsampled.shape
    hits = 0
    for b in range(B):
        cls = int(labels[b])
        proto_idx = (class_of_proto == cls).nonzero(as_tuple=True)[0]
        if len(proto_idx) == 0:
            continue
        # pick the prototype (of the true class) with the strongest peak
        sub = alpha_upsampled[b, proto_idx]              # [k, H, W]
        peaks = sub.flatten(1).max(dim=1).values
        best = proto_idx[int(peaks.argmax())]
        flat = alpha_upsampled[b, best].flatten().argmax()
        y, x = divmod(int(flat), W)
        if masks[b, y, x] > 0.5:
            hits += 1
    return hits / max(B, 1)


def class_of_proto_tensor(protos_per_class, num_classes, device="cpu"):
    return torch.arange(num_classes, device=device).repeat_interleave(protos_per_class)
