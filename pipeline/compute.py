"""Compute backend detection. Every model script imports this so it runs unchanged on a
local CPU, a Colab/Kaggle GPU, or with no torch at all (CPU-only numpy/scipy/sklearn path)."""
import os
import random

import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:  # noqa: BLE001 - blocked DLLs raise ImportError/OSError
    torch = None
    HAS_TORCH = False


def device(prefer=None):
    """'cuda' if available (Colab/Kaggle), else 'cpu'; None when torch is unavailable."""
    if not HAS_TORCH:
        return None
    prefer = prefer or os.environ.get("SIH_DEVICE")
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)


def describe():
    if not HAS_TORCH:
        return "torch unavailable -> CPU-only numpy/scipy/sklearn path"
    d = device()
    extra = f" ({torch.cuda.get_device_name(0)})" if d.type == "cuda" else \
        f" ({torch.get_num_threads()} threads)"
    return f"torch {torch.__version__} on {d}{extra}"


if __name__ == "__main__":
    print(describe())
