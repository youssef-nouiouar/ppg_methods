"""Accuracy, AUC-ROC and macro-F1."""
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def classification_metrics(probs, labels, num_classes):
    """probs: [N, C] softmax probs. labels: [N] ints."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    try:
        if num_classes == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")   # happens if a class is absent from a tiny split

    return {"accuracy": acc, "auc": auc, "f1": f1}
