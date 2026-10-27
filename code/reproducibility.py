"""Seed every training RNG; deterministic kernels are opt-in and recorded."""
import os
import random

import numpy as np
import torch


def seed_everything(seed, deterministic=False):
    seed = int(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))


def runtime_metadata():
    import platform
    return dict(python=platform.python_version(), numpy=np.__version__,
                torch=torch.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                deterministic=torch.are_deterministic_algorithms_enabled())
