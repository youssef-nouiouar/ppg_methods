"""Explainability evaluation for PPG and the baselines.

Two kinds of models here:
  * interpretable-by-design (PPG, DeterministicProto) -> expose prototype
    attention maps `alpha`; the explanation IS the mechanism.
  * black-box (PlainSwin, MCDropoutViT)             -> no built-in explanation;
    we fall back to a model-agnostic gradient saliency so every model gets a
    comparable heatmap. This contrast is itself a result: the black boxes need
    an external, post-hoc method to be explained at all.

Quantitative explainability metrics used here (all runnable on BreakHis, which
has NO ground-truth lesion masks - so we use mask-free proxies):

  * Attention concentration (entropy): how focused is the explanation? A sharp,
    low-entropy map points at a specific region; a diffuse map explains little.
  * Deletion AUC: mask out the top-k% most-important pixels and measure how fast
    the predicted probability for the true class drops. A faithful explanation
    causes a FAST drop (low deletion-AUC).
  * Insertion AUC: start from a blurred image and add back the top pixels; a
    faithful explanation makes probability rise FAST (high insertion-AUC).

Deletion/Insertion (Petsiuk et al., 2018) need no ground-truth boxes, so they
are the honest choice for BreakHis. The Pointing Game (needs boxes) stays in
metrics/pointing_game.py for datasets that provide masks.
"""
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- heatmaps ----
def prototype_saliency(model, x, true_class, protos_per_class):
    """For prototype models: upsample the attention of the most-activated
    prototype belonging to the predicted class. Returns [B, H, W] in [0,1]."""
    out = model(x)
    alpha = out["alpha"]                       # [B, P, h, w]
    B, P, h, w = alpha.shape
    K = protos_per_class
    logits = out["logits"]
    pred = logits.argmax(1)
    sal = []
    for b in range(B):
        c = int(pred[b])
        idx = slice(c * K, (c + 1) * K)
        sub = alpha[b, idx]                     # [K, h, w]
        peak = sub.flatten(1).max(1).values
        best = sub[int(peak.argmax())]         # [h, w]
        sal.append(best)
    sal = torch.stack(sal).unsqueeze(1)        # [B,1,h,w]
    sal = F.interpolate(sal, size=x.shape[-2:], mode="bilinear", align_corners=False)
    sal = sal.squeeze(1)
    # normalise per-image to [0,1]
    flat = sal.flatten(1)
    mn = flat.min(1).values[:, None, None]
    mx = flat.max(1).values[:, None, None]
    return ((sal - mn) / (mx - mn + 1e-8))


def gradient_saliency(model, x, true_class):
    """Model-agnostic fallback for black boxes: |d logit_true / d input|,
    summed over channels. Returns [B, H, W] in [0,1]."""
    x = x.clone().requires_grad_(True)
    out = model(x)
    logits = out["logits"]
    score = logits.gather(1, true_class.view(-1, 1)).sum()
    grad = torch.autograd.grad(score, x, create_graph=False)[0]
    sal = grad.abs().sum(1)                     # [B, H, W]
    flat = sal.flatten(1)
    mn = flat.min(1).values[:, None, None]
    mx = flat.max(1).values[:, None, None]
    return ((sal - mn) / (mx - mn + 1e-8)).detach()


def get_saliency(model, x, y, cfg):
    """Dispatch: prototype attention if available, else gradient saliency."""
    model.eval()
    with torch.no_grad():
        has_alpha = "alpha" in model(x)
    if has_alpha:
        with torch.no_grad():
            return prototype_saliency(model, x, y, cfg.protos_per_class), "prototype-attention"
    return gradient_saliency(model, x, y), "gradient-saliency"


# ---------------------------------------------------------------- metrics -----
def attention_entropy(sal):
    """Mean normalised entropy of the saliency maps (lower = more focused)."""
    p = sal.flatten(1)
    p = p / (p.sum(1, keepdim=True) + 1e-8)
    ent = -(p * (p + 1e-12).log()).sum(1)
    ent = ent / np.log(p.shape[1])             # normalise to [0,1]
    return float(ent.mean())


@torch.no_grad()
def deletion_insertion(model, x, y, sal, steps=20, mode="deletion"):
    """Faithfulness. Progressively remove (deletion) or add (insertion) the most
    salient pixels and track prob of the true class. Returns the area under the
    prob-vs-fraction curve, averaged over the batch."""
    model.eval()
    B, C, H, W = x.shape
    n = H * W
    order = sal.flatten(1).argsort(dim=1, descending=True)   # most salient first
    if mode == "deletion":
        cur = x.clone()
        baseline = torch.zeros_like(x)
    else:  # insertion: start from a blurred image
        k = 11
        blur = F.avg_pool2d(x, k, stride=1, padding=k // 2)
        cur = blur.clone()
        baseline = x
    fracs = np.linspace(0, 1, steps + 1)
    curves = np.zeros((B, steps + 1))
    for s, frac in enumerate(fracs):
        probs = F.softmax(model(cur)["logits"], dim=1)
        curves[:, s] = probs.gather(1, y.view(-1, 1)).squeeze(1).cpu().numpy()
        if s == steps:
            break
        # reveal/remove the next chunk of pixels
        kk = int((fracs[s + 1]) * n) - int(frac * n)
        if kk <= 0:
            continue
        idx = order[:, int(frac * n):int(frac * n) + kk]     # [B, kk]
        for b in range(B):
            flat_cur = cur[b].view(C, -1)
            flat_base = baseline[b].view(C, -1)
            flat_cur[:, idx[b]] = flat_base[:, idx[b]]
            cur[b] = flat_cur.view(C, H, W)
    auc = curves.mean(axis=1).mean()           # area under prob curve, mean over batch
    return float(auc)


def explainability_report(model, loader, cfg, max_batches=8):
    """Aggregate explainability metrics over a few batches of a data loader."""
    model.eval()
    ents, dels, inss, method = [], [], [], None
    seen = 0
    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        sal, method = get_saliency(model, x, y, cfg)
        ents.append(attention_entropy(sal))
        dels.append(deletion_insertion(model, x, y, sal, mode="deletion"))
        inss.append(deletion_insertion(model, x, y, sal, mode="insertion"))
        seen += 1
        if seen >= max_batches:
            break
    return {
        "explanation_type": method,
        "attention_entropy": float(np.mean(ents)),      # lower = more focused
        "deletion_auc": float(np.mean(dels)),           # lower = more faithful
        "insertion_auc": float(np.mean(inss)),          # higher = more faithful
    }


# ---------------------------------------------------------------- viz ---------
def denormalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (x.cpu() * std + mean).clamp(0, 1)


def overlay_grid(models_dict, x, y, cfg, n=4):
    """Return a matplotlib figure: rows = examples, cols = original + each model's
    saliency overlay. Import matplotlib lazily so the module has no hard dep."""
    import matplotlib.pyplot as plt
    x, y = x[:n].to(cfg.device), y[:n].to(cfg.device)
    names = list(models_dict.keys())
    fig, axes = plt.subplots(n, len(names) + 1, figsize=(3 * (len(names) + 1), 3 * n))
    if n == 1:
        axes = axes[None, :]
    imgs = denormalize(x)
    for r in range(n):
        axes[r, 0].imshow(imgs[r].permute(1, 2, 0))
        axes[r, 0].set_title("input" if r == 0 else "")
        axes[r, 0].axis("off")
        for c, name in enumerate(names):
            sal, method = get_saliency(models_dict[name], x[r:r+1], y[r:r+1], cfg)
            axes[r, c + 1].imshow(imgs[r].permute(1, 2, 0))
            axes[r, c + 1].imshow(sal[0].cpu(), cmap="jet", alpha=0.5)
            if r == 0:
                axes[r, c + 1].set_title(f"{name}\n({method})", fontsize=9)
            axes[r, c + 1].axis("off")
    plt.tight_layout()
    return fig
