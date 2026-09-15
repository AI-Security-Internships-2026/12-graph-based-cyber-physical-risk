"""
Central seed control. Every experiment MUST call set_seed() before any
data split, model init, or training loop. Do not scatter torch.manual_seed()
calls across notebooks/scripts — call this once, here.
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, PyTorch (CPU+CUDA) seeds.

    deterministic=True trades a little speed for exact reproducibility
    (forces deterministic cuDNN kernels). Keep True for paper experiments;
    only set False for quick throwaway exploration.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Some PyG scatter ops are nondeterministic on GPU without this.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seeded_generator(seed: int) -> torch.Generator:
    """Use for torch.randperm / DataLoader shuffling so split generation
    is independent of (and doesn't consume) the global RNG state."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
