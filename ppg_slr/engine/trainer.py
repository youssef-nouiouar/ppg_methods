"""Training / evaluation loops shared by every model."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from metrics.classification import classification_metrics
from metrics.calibration import calibration_metrics


def make_loaders(datasets, cfg):
    def dl(ds, shuffle):
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    return {
        "train": dl(datasets["train"], True),
        "val": dl(datasets["val"], False),
        "test": dl(datasets["test"], False),
    }


def cosine_warmup(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * prog))


def class_weights(dataset, num_classes, device):
    """Inverse-frequency weights so the loss is not dominated by the majority class."""
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    w = counts.sum() / (num_classes * counts)      # mean weight ~ 1
    return torch.tensor(w, dtype=torch.float32, device=device)


def anneal_gumbel_tau(model, step, total, tau0, tau_min):
    """Cosine anneal the Gumbel temperature high->low; hard-switch late = stabler."""
    prog = min(step / max(total, 1), 1.0)
    tau = tau_min + 0.5 * (tau0 - tau_min) * (1 + math.cos(math.pi * prog))
    # set on the PPG layer if present
    ppg = getattr(model, "ppg", None)
    if ppg is not None and hasattr(ppg, "current_gumbel_tau"):
        ppg.current_gumbel_tau = tau


@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(cfg.device)
        out = model(x)
        probs = F.softmax(out["logits"], dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    m = {}
    m.update(classification_metrics(probs, labels, cfg.num_classes))
    m.update(calibration_metrics(probs, labels))
    return m


def train(model, datasets, cfg, verbose=True):
    loaders = make_loaders(datasets, cfg)
    model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if getattr(cfg, "class_balanced_loss", False):
        w = class_weights(datasets["train"], cfg.num_classes, cfg.device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    total_steps = cfg.epochs * len(loaders["train"])
    warmup_steps = cfg.warmup_epochs * len(loaders["train"])
    step = 0
    select_on = getattr(cfg, "select_metric", "f1")   # f1 is robust under imbalance
    best_score, best_state = -1.0, None

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(cfg.device), y.to(cfg.device)
            for g in opt.param_groups:
                g["lr"] = cosine_warmup(step, total_steps, warmup_steps, cfg.lr)
            anneal_gumbel_tau(model, step, total_steps,
                              cfg.gumbel_tau, getattr(cfg, "gumbel_tau_min", cfg.gumbel_tau))
            out = model(x)
            loss = criterion(out["logits"], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            step += 1

        val = evaluate(model, loaders["val"], cfg)
        if verbose:
            print(f"[{cfg.exp_name}] epoch {epoch+1:02d}/{cfg.epochs} "
                  f"loss={running/len(datasets['train']):.4f} "
                  f"val_acc={val['accuracy']:.4f} val_auc={val['auc']:.4f} "
                  f"val_f1={val['f1']:.4f} val_ece={val['ece']:.4f}")
        # Select on accuracy/F1, NOT AUC: a collapsed epoch can have high AUC but
        # terrible decisions, and selecting on AUC would save that broken model.
        score = val.get(select_on, val["accuracy"])
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, loaders["test"], cfg)
    if verbose:
        print(f"[{cfg.exp_name}] TEST  acc={test['accuracy']:.4f} auc={test['auc']:.4f} "
              f"f1={test['f1']:.4f} ece={test['ece']:.4f} nll={test['nll']:.4f}")
    return model, test
