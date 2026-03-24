"""
Centralized reproducibility seed setup.

Import and call set_seeds(42) at the top of every training/evaluation script
to ensure deterministic results across runs.

Reference: https://docs.pytorch.org/docs/stable/notes/randomness.html
"""
import random

import numpy as np
import torch


def set_seeds(seed: int = 42) -> None:
    """Fix all stochastic sources for reproducible training runs.

    Sets seeds for: Python random, numpy, PyTorch CPU, PyTorch CUDA.
    Also disables cuDNN non-deterministic algorithms.

    Note: cudnn.deterministic=True + benchmark=False trades ~10-15% training
    speed for full determinism. Acceptable cost for a research project.

    Args:
        seed: Integer seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
