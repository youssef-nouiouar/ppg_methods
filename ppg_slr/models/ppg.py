"""PPG v2 -- Uncertainty-Gated Prototype Evidence.

Pipeline (per image, M stochastic passes through the prototype layer):

  s_kij^(m) = cosine(p_k, F_ij^(m))                       similarity
  alpha     = softmax_ij(s / tau)   (or hard max)         spatial attention
  r_k^(m)   = sum_ij alpha * s                            prototype response
  e_k       = mean_m relu(r_k^(m))                        EVIDENCE
  u_k       = std_m  relu(r_k^(m))   (unbiased)           UNCERTAINTY
  u~_k      = u_k / (median_train(u) + eps)               normalised uncertainty
  g_k       = exp(-lambda * u~_k)                         ONE deterministic gate
  a_k       = g_k * e_k                                   gated evidence
  logit_c   = b_c + s * sum_k W_ck a_k,  W_ck >= 0, s > 0 positive-evidence head
              (s = one learnable positive scalar, see logit_scale below)

REMOVED compared with the previous version:
  * Gumbel-sigmoid Bernoulli gate z   (gate_logit, gumbel_sigmoid, gumbel_tau, ...)
  * learned monotonic gate g          (gate_theta, gate_bias)
  * Student t-interval                (prototype_confidence_interval)

Two-stage training (see PPGSwinT.set_stage / calibrate_u_median):
  stage 1: gate OFF, a_k = e_k computed from a single pass
  stage 2: gate ON,  M passes, u_median calibrated at the start of the stage
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class PPGLayer(nn.Module):
    def __init__(self, cfg, feat_dim):
        super().__init__()
        self.cfg = cfg
        self.P = cfg.total_protos()
        self.K = cfg.protos_per_class
        self.C = cfg.num_classes
        P, C, K = self.P, self.C, self.K

        # ---- prototypes (prototype p belongs to class p // K) ----
        self.prototypes = nn.Parameter(torch.randn(P, feat_dim) * 0.02)
        self.log_tau = nn.Parameter(torch.tensor(math.log(cfg.temperature)))

        # ---- MC dropout on the feature map ----
        self.p_drop = cfg.mc_dropout_p
        self.mc_samples = int(cfg.mc_samples)

        # ---- the single deterministic gate: hyper-parameters only, nothing learned ----
        self.gate_lambda = float(getattr(cfg, "gate_lambda", 0.5))
        self.gate_eps = float(getattr(cfg, "gate_eps", 1e-6))
        # detach u inside the gate so the network cannot "game" the gate by
        # shrinking u (Goodhart). Set cfg.gate_detach_u=False to ablate.
        self.gate_detach_u = bool(getattr(cfg, "gate_detach_u", True))
        self.use_gate = True                                   # toggled by set_stage()
        self.register_buffer("u_median", torch.ones(()))       # filled by calibrate_u_median()

        # ---- positive class head: logit = b + s * (W a), W >= 0, s > 0 ----
        # s is ONE positive scalar multiplying all evidence. It does not change the
        # interpretation (still purely positive evidence) but lets logits reach a
        # useful range quickly: a_k <= 1, so without it W would need to grow to ~10
        # through a softplus at AdamW speed (thousands of steps).
        self.log_scale = nn.Parameter(torch.tensor(math.log(float(getattr(cfg, "logit_scale_init", 10.0)))))
        own = F.one_hot(torch.arange(P) // K, C).t().float()   # [C, P]
        self.register_buffer("own_mask", own)
        if cfg.head == "fixed":
            self.register_buffer("W_fixed", own / K)
            self.W_raw = None
            self.bias = None
        else:
            init = torch.where(own.bool(),
                               torch.full_like(own, _inv_softplus(1.0 / K)),  # own class ~ 1/K
                               torch.full_like(own, -5.0))                    # others ~ 0.007
            self.W_raw = nn.Parameter(init)
            self.bias = nn.Parameter(torch.zeros(C))

    # ------------------------------------------------------------------ head
    def W(self):
        """Non-negative prototype->class matrix [C, P]."""
        return self.W_fixed if self.W_raw is None else F.softplus(self.W_raw)

    def l1_penalty(self, off_class_only: bool = True):
        """L1 on the head for the stage-1 loss (lambda_W * ||W||_1).
        ProtoPNet only penalises OFF-class connections; that is the default."""
        W = self.W()
        if off_class_only:
            W = W * (1.0 - self.own_mask)
        return W.abs().sum()

    # ------------------------------------------------------------ one pass
    def _score_once(self, feat):
        """feat [B,C,H,W] -> (r [B,P], alpha [B,P,H,W])."""
        drop_on = self.training or self.use_gate          # MC dropout is ON whenever the gate is on
        f = F.dropout2d(feat, p=self.p_drop, training=drop_on)
        f = F.normalize(f, dim=1)
        p = F.normalize(self.prototypes, dim=1)
        S = torch.einsum("pc,bchw->bphw", p, f)           # cosine similarity map

        if self.cfg.pooling == "maxpool":
            r = S.flatten(2).max(dim=2).values
            alpha = torch.zeros_like(S)
            idx = S.flatten(2).argmax(dim=2)
            alpha.flatten(2).scatter_(2, idx.unsqueeze(-1), 1.0)
        else:
            tau = self.log_tau.exp().clamp(min=1e-3)
            alpha = F.softmax(S.flatten(2) / tau, dim=2).view_as(S)
            r = (alpha * S).flatten(2).sum(dim=2)
        return r, alpha

    # --------------------------------------------------------------- forward
    def forward(self, feat):
        M = self.mc_samples if self.use_gate else 1       # stage 1 -> a single pass
        rs, alpha_sum = [], 0.0
        for _ in range(M):
            r, a = self._score_once(feat)
            rs.append(r)
            alpha_sum = alpha_sum + a
        samples = F.relu(torch.stack(rs, 0))              # [M,B,P]  relu BEFORE aggregating
        alpha = alpha_sum / M                             # mean attention for visualisation

        e = samples.mean(0)                               # evidence
        if M > 1:
            u = samples.var(0, unbiased=True).clamp_min(1e-12).sqrt()   # uncertainty
        else:
            u = torch.zeros_like(e)

        if self.use_gate and M > 1:
            u_in = u.detach() if self.gate_detach_u else u
            u_tilde = u_in / (self.u_median + self.gate_eps)
            g = torch.exp(-self.gate_lambda * u_tilde)    # the ONE gate, in (0, 1]
        else:
            g = torch.ones_like(e)

        a = g * e                                         # gated evidence

        logits = self.log_scale.exp() * (a @ self.W().t())
        if self.bias is not None:
            logits = logits + self.bias

        return {"logits": logits, "e": e, "u": u, "g": g, "a": a,
                "alpha": alpha, "samples": samples,
                # backward-compatible aliases for existing train/eval code
                "mu": e, "sigma": u}

    # ------------------------------------------- explanation & uncertainty
    @torch.no_grad()
    def explanation_map(self, out, class_idx=None):
        """E_c(i,j) = sum_k W_ck a_k alpha_bar_kij  -> [B,H,W].
        class_idx: LongTensor [B]; defaults to the predicted class."""
        if class_idx is None:
            class_idx = out["logits"].argmax(1)
        coef = self.log_scale.exp() * self.W()[class_idx] * out["a"]   # [B,P]; sum(E) = logit_c - b_c
        return torch.einsum("bp,bphw->bhw", coef, out["alpha"])

    @torch.no_grad()
    def sample_uncertainty(self, out, class_idx=None, eps=1e-6):
        """U(x) = sum_k W_ck e_k u_k / (sum_k W_ck e_k + eps)  -> [B]."""
        if class_idx is None:
            class_idx = out["logits"].argmax(1)
        w = self.W()[class_idx]
        return (w * out["e"] * out["u"]).sum(1) / ((w * out["e"]).sum(1) + eps)


@torch.no_grad()
def prototype_mc_interval(samples, level=0.95):
    """Empirical MC interval per prototype from samples [M,B,P] (replaces the
    Student t-interval). Returns (low, median, high), each [B,P]."""
    q = torch.tensor([(1 - level) / 2, 0.5, 1 - (1 - level) / 2],
                     device=samples.device, dtype=samples.dtype)
    lo, med, hi = torch.quantile(samples, q, dim=0)
    return lo, med, hi


class PPGSwinT(nn.Module):
    """Backbone + PPG layer."""
    def __init__(self, cfg, backbone):
        super().__init__()
        self.backbone = backbone
        self.ppg = PPGLayer(cfg, feat_dim=backbone.out_channels)

    def forward(self, x):
        return self.ppg(self.backbone(x))

    def set_stage(self, stage: int):
        """stage 1: gate off, single pass.  stage 2: gate on, M MC passes."""
        assert stage in (1, 2)
        self.ppg.use_gate = (stage == 2)

    @torch.no_grad()
    def calibrate_u_median(self, loader, device="cuda", max_batches=None):
        """Estimate median_train(u) once, at the start of stage 2.

        Uses only entries with u > 0: relu() can make a prototype exactly 0 in
        every pass (u = 0), and including those would drag the median to ~0 and
        make every normalised uncertainty explode. Batches may be tensors or
        (x, y, ...) tuples.
        """
        was_training, was_gate = self.training, self.ppg.use_gate
        self.eval()
        self.ppg.use_gate = True                      # force M passes with dropout
        us = []
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            us.append(self(x.to(device))["u"].flatten().cpu())
        u = torch.cat(us)
        u = u[u > 0]
        med = u.median() if u.numel() else torch.tensor(1.0)
        self.ppg.u_median.copy_(med.to(self.ppg.u_median.device))
        self.train(was_training)
        self.ppg.use_gate = was_gate
        return float(med)