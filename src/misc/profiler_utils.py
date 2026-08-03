"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from typing import Tuple


def stats(
    cfg,
    input_shape: Tuple = (1, 3, 640, 640),
) -> Tuple[int, dict]:
    # FLOPs/MACs profiling is not required for training.  The third-party
    # calflops package imports Hugging Face Transformers at module import time,
    # which produces irrelevant PyTorch-version warnings for D-FINE.  Keep a
    # concise parameter-count message and profile FLOPs separately only when
    # preparing the paper.
    params = sum(p.numel() for p in cfg.model.parameters())
    return params, {"Model Params": f"{params / 1e6:.2f} M"}
