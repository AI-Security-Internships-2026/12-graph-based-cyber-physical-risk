"""
One metrics implementation, used by every experiment. Several notebook
cells (Part C, Part G, Part M, Part X ...) had their own slightly
different inline accuracy/precision/recall blocks — this replaces all
of them so a metric name always means the same computation.
"""
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def compute_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
) -> Dict:
    """Binary classification metrics.

    pred, true: 1D LongTensor of {0,1}
    probs: optional 1D FloatTensor of P(class=1), needed for AUPRC.
    """
    pred = pred.detach().cpu()
    true = true.detach().cpu()

    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())

    accuracy = (pred == true).float().mean().item()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    auprc = None
    if probs is not None:
        probs_np = probs.detach().cpu().numpy()
        true_np = true.numpy()
        if len(np.unique(true_np)) > 1:
            auprc = float(average_precision_score(true_np, probs_np))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": auprc,
        "fpr": fpr,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
