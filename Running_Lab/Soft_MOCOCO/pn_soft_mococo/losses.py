from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
from torch import Tensor

from Base.Code_Snippet.loss_code import (
    build_target_orthogonal_leakage_kwargs,
    target_orthogonal_leakage_loss,
)

__all__ = [
    "build_target_orthogonal_leakage_kwargs",
    "target_orthogonal_leakage_loss",
]
