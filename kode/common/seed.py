"""Seed global agar hasil reproducible."""
import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cudnn deterministik: sedikit lebih lambat, tapi angka bisa diulang
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass


def worker_init_fn(worker_id: int) -> None:
    """Untuk DataLoader dengan num_workers > 0."""
    import torch
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
