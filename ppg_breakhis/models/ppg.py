"""Probabilistic Prototype Gating (PPG) layer and PPG-SwinT model.

This implements the CORRECTED design discussed in review:

  1. cosine similarity between prototypes and every spatial location
  2. temperature-scaled spatial softmax  -> attention  (or hard max, ablation)
  3. M MC-dropout passes -> mean mu and UNBIASED variance -> uncertainty u = sigma
  4. gate that CLOSES on high uncertainty:  p = sigmoid(beta * (gamma - u))
        (the 'original' ablation flips the sign, reproducing the buggy version)
  5. Gumbel-sigmoid (binary concrete) sample z  (differentiable)
  6. TRUE multiplicative uncertainty gate g = sigmoid(b - softplus(theta) . u),
        monotonically decreasing in u by construction (non-negative weights)
  7. rectified mean so activations are non-negative:  a = z * g * relu(mu)
  8. head:  y = W a,  with P = K * C total prototypes (fixed block weights or learnable)

Set mc_samples=1 to disable uncertainty (u=0, gates open) for the ablation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_sigmoid(logit_p, tau, training):
    """Binary-concrete relaxation of Bernoulli(sigmoid(logit_p)).

    z = sigmoid((logit_p + L) / tau), L ~ Logistic(0,1) = log U - log(1-U).
    At eval we drop the noise and return the expected (deterministic) gate.
    """
    if not training:
        return torch.sigmoid(logit_p)
    u = torch.rand_like(logit_p).clamp_(1e-6, 1 - 1e-6)
    noise = torch.log(u) - torch.log1p(-u)          # logistic noise = g1 - g2
    return torch.sigmoid((logit_p + noise) / tau)


class PPGLayer(nn.Module):
    def __init__(self, cfg, feat_dim):
        super().__init__()
        self.cfg = cfg
        P = cfg.total_protos()
        self.P = P
        self.K = cfg.protos_per_class
        self.C = cfg.num_classes

        # prototypes: [P, feat_dim]
        self.prototypes = nn.Parameter(torch.randn(P, feat_dim) * 0.02)
        self.log_tau = nn.Parameter(torch.tensor(math.log(cfg.temperature)))

        # uncertainty gate g: non-negative weights via softplus(theta)
        self.gate_theta = nn.Parameter(torch.zeros(P, P))
        self.gate_bias = nn.Parameter(torch.ones(P) * 2.0)   # start with gates open

        self.dropout = nn.Dropout2d(cfg.mc_dropout_p)

        # classification head
        if cfg.head == "fixed":
            W = torch.zeros(self.C, P)
            for c in range(self.C):
                W[c, c * self.K:(c + 1) * self.K] = 1.0 / self.K
            self.register_buffer("W_head", W)
            self.head = None
        else:
            self.head = nn.Linear(P, self.C, bias=False)

    # ---- one stochastic scoring pass over a feature map ----
    def _score_once(self, feat):
        """feat: [B, C, H, W] -> (score [B,P], alpha [B,P,H,W])."""
        B, Cc, Hh, Ww = feat.shape
        f = self.dropout(feat)                                   # MC-dropout perturbation
        f = F.normalize(f, dim=1)                                # unit feature vectors
        p = F.normalize(self.prototypes, dim=1)                  # unit prototypes [P, C]
        # cosine similarity map S: [B, P, H, W]
        S = torch.einsum("pc,bchw->bphw", p, f)

        if self.cfg.pooling == "maxpool":
            score = S.flatten(2).max(dim=2).values               # [B, P]
            alpha = torch.zeros_like(S)                          # placeholder for viz
            idx = S.flatten(2).argmax(dim=2)
            alpha.flatten(2).scatter_(2, idx.unsqueeze(-1), 1.0)
        else:
            tau = self.log_tau.exp().clamp(min=1e-3)
            alpha = F.softmax(S.flatten(2) / tau, dim=2).view_as(S)  # [B,P,H,W], sums to 1
            score = (alpha * S).flatten(2).sum(dim=2)                # attention-weighted mean

        return score, alpha

    def forward(self, feat):
        """feat: [B, C, H, W]. Returns dict with logits, mu, sigma, alpha."""
        M = self.cfg.mc_samples
        # run M stochastic passes (dropout differs each time)
        scores, alphas = [], []
        for _ in range(max(M, 1)):
            s, a = self._score_once(feat)
            scores.append(s)
            alphas.append(a)
        scores = torch.stack(scores, dim=0)      # [M, B, P]
        alpha = torch.stack(alphas, dim=0).mean(0)  # [B, P, H, W] for visualisation

        mu = scores.mean(dim=0)                   # [B, P]
        if M > 1:
            sigma = scores.var(dim=0, unbiased=True).clamp_min(1e-12).sqrt()  # [B, P]
        else:
            sigma = torch.zeros_like(mu)          # no uncertainty available

        u = sigma                                 # epistemic proxy

        # ---- Bernoulli gate z ----
        if self.cfg.use_bernoulli_gate and M > 1:
            if self.cfg.gate_direction == "corrected":
                gate_logit = self.cfg.beta * (self.cfg.gamma - u)   # closes on high u
            else:  # 'original' -> the buggy sign that OPENS on high u
                gate_logit = self.cfg.beta * (u - self.cfg.gamma)
            z = gumbel_sigmoid(gate_logit, self.cfg.gumbel_tau, self.training)
        else:
            z = torch.ones_like(mu)

        # ---- uncertainty gate g (monotone decreasing in u) ----
        if self.cfg.use_uncertainty_gate and M > 1:
            W_tilde = F.softplus(self.gate_theta)                  # [P, P] >= 0
            g = torch.sigmoid(self.gate_bias - u @ W_tilde.t())    # [B, P]
        else:
            g = torch.ones_like(mu)

        a = z * g * F.relu(mu)                    # [B, P]

        if self.head is None:
            logits = a @ self.W_head.t()          # fixed positive block weights
        else:
            logits = self.head(a)

        return {"logits": logits, "mu": mu, "sigma": sigma, "alpha": alpha, "a": a}


class PPGSwinT(nn.Module):
    """Backbone + PPG head."""
    def __init__(self, cfg, backbone):
        super().__init__()
        self.backbone = backbone
        self.ppg = PPGLayer(cfg, feat_dim=backbone.out_channels)

    def forward(self, x):
        feat = self.backbone(x)
        return self.ppg(feat)


def prototype_confidence_interval(mu, sigma, M, level=0.95):
    """Per-prototype CI using a t-interval (correct for small M), not 1.96*z.

    Returns (low, high) for the MEAN activation. For M=5, t_{0.975,4}=2.776.
    """
    from scipy import stats
    if M <= 1:
        return mu, mu
    t = stats.t.ppf(0.5 + level / 2.0, df=M - 1)
    half = t * sigma / math.sqrt(M)
    return mu - half, mu + half
