"""
Result-writing helpers. Every experiment produces exactly one JSON file
matching REQUIRED_FIELDS (Issue #1, section 5). If a field genuinely does
not apply to an experiment, it is still present in the JSON, set to null,
with a one-line reason in `notes`.
"""
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REQUIRED_FIELDS = [
    "dataset", "model", "split_protocol", "seed",
    "train_size", "validation_size", "test_size",
    "feature_set", "graph_mode", "threshold",
    "accuracy", "precision", "recall", "f1", "auprc", "fpr",
    "confusion_matrix", "runtime", "library_versions", "git_commit",
]


def get_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def get_library_versions() -> Dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for pkg in ("torch", "torch_geometric", "numpy", "pandas", "sklearn", "networkx"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


def write_result(result: Dict[str, Any], output_path: str) -> None:
    """Fill in environment metadata, validate schema, write JSON."""
    result.setdefault("git_commit", get_git_commit())
    result.setdefault("library_versions", get_library_versions())

    missing = [f for f in REQUIRED_FIELDS if f not in result]
    if missing:
        raise ValueError(
            f"Result is missing required fields: {missing}. "
            f"Set them to null explicitly with a reason in `notes` "
            f"if not applicable — do not silently omit them."
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[io] wrote {out}")


class Timer:
    """with Timer() as t: ...  -> t.elapsed (seconds)"""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
