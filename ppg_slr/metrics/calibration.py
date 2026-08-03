"""Calibration: Expected Calibration Error (ECE) and Negative Log-Likelihood.

This is where the uncertainty claim is actually tested - keep it in every run.
"""
import numpy as np


def expected_calibration_error(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / N) * abs(acc_bin - conf_bin)
    return float(ece)


def negative_log_likelihood(probs, labels, eps=1e-12):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    p_true = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(p_true, eps, 1.0)).mean())


def calibration_metrics(probs, labels):
    return {
        "ece": expected_calibration_error(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
    }
