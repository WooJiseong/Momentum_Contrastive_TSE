from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from pn_soft_mococo.paths import add_repo_paths

add_repo_paths()

from pn_mococo.moco_encoder import (
    MomentumContrastivePNLearner,
    encode_fused_grad,
)


class SoftMomentumContrastivePNLearner(MomentumContrastivePNLearner):
    """Standard PN_MOCOCO learner with a frozen Teacher representation anchor."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        ccfg = cfg.get("contrastive", {})
        ckpt = ccfg.get("initial_pn_ckpt")
        if not ckpt:
            raise ValueError("contrastive.initial_pn_ckpt is required")
        self.teacher = copy.deepcopy(self.student_encoder)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        self.teacher_loss_weight = float(ccfg.get("teacher_loss_weight", 0.1))

    def teacher_loss(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        student_dense = encode_fused_grad(self.student_encoder, pos, neg)
        with torch.no_grad():
            teacher_dense = encode_fused_grad(self.teacher, pos, neg)
        student_dense = F.normalize(student_dense.float(), dim=1)
        teacher_dense = F.normalize(teacher_dense.float(), dim=1)
        return (1.0 - (student_dense * teacher_dense).sum(dim=1)).mean()

