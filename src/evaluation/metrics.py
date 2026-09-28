"""One metrics implementation, used by every experiment. Several notebook cells had their
own slightly different inline accuracy/precision/recall blocks — this replaces all of
them so a metric name always means the same computation.
"""
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
) -> Dict:
    """Binary classification metrics.

    pred, true: 1D LongTensor of {0,1}
    probs: optional 1D FloatTensor of P(class=1), needed for AUPRC/AUROC.
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
    auroc = None
    if probs is not None:
        probs_np = probs.detach().cpu().numpy()
        true_np = true.numpy()
        if len(np.unique(true_np)) > 1:
            auprc = float(average_precision_score(true_np, probs_np))
            # Undefined (and sklearn raises) with a single class present, same guard as
            # AUPRC above; left as None rather than raising.
            auroc = float(roc_auc_score(true_np, probs_np))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": auprc,
        "auroc": auroc,
        "fpr": fpr,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def per_attack_recall(
    pred: torch.Tensor,
    true: torch.Tensor,
    attack_labels: Sequence[str],
) -> Dict[str, Optional[float]]:
    """Recall broken out by attack subtype.

    attack_labels: the raw Attack_Type string for each row in `true`'s
    positive class AND every other class (e.g. 'normal') — same length
    and same row order as `true`/`pred`. Rows whose true label is a class
    with is_attack==0 (normal traffic) are ignored; only subtypes that
    are actually attacks get a recall entry. A subtype with zero test
    rows (e.g. excluded by a split's IP-diversity filter) is reported as
    None with a reason, matching the "applicable: False" philosophy in
    src/evaluation/audit.py rather than silently omitting the key.
    """
    pred = pred.detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    attack_labels = np.asarray(attack_labels)
    if len(attack_labels) != len(true):
        raise ValueError(
            f"attack_labels length {len(attack_labels)} != true length {len(true)}"
        )

    result: Dict[str, Optional[float]] = {}
    for label in sorted(set(attack_labels[true == 1].tolist())):
        mask = (attack_labels == label) & (true == 1)
        n = int(mask.sum())
        if n == 0:
            result[label] = None
            continue
        result[label] = float((pred[mask] == 1).sum() / n)
    return result


def count_trainable_params(model: torch.nn.Module) -> int:
    """Not meaningful for sklearn/XGBoost baselines — those report None with a reason in
    `notes` instead (see src/models/baselines.py), rather than an invented parameter
    count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_latency(
    predict_fn: Callable[[], None],
    n_repeats: int = 20,
    warmup: int = 3,
) -> Dict[str, float]:
    """`predict_fn` must run exactly one forward/inference pass with no side effects other
    than producing a result (i.e. wrap the already- built inputs in a closure; do not
    rebuild tensors/features inside the timed call, or you are timing feature
    engineering, not inference).

    Returns per-call mean/std in milliseconds over `n_repeats` calls
    after `warmup` untimed calls. This is single-process CPU wall time on
    whatever machine actually runs the experiment — comparable across
    models run in the same process/session, NOT an absolute cross-machine
    benchmark; the run's `library_versions`/environment in the result
    JSON is the record of what machine produced it.
    """
    for _ in range(warmup):
        predict_fn()

    times_ms: List[float] = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        predict_fn()
        times_ms.append((time.perf_counter() - start) * 1000.0)

    times_ms = np.array(times_ms)
    return {
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "n_repeats": n_repeats,
    }
