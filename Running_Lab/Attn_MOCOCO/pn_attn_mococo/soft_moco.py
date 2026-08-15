from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from pn_mococo.moco_encoder import MomentumContrastivePNLearner, encode_fused_grad


class SoftMomentumContrastivePNLearner(MomentumContrastivePNLearner):
    """Soft_MOCOCO learner with the fixed original PN representation anchor."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.teacher = copy.deepcopy(self.student_encoder)
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher.eval()
        self.teacher_loss_weight = float(cfg.get("contrastive", {}).get("teacher_loss_weight", 0.1))

    def teacher_loss(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        student_dense = encode_fused_grad(self.student_encoder, pos, neg)
        with torch.no_grad():
            teacher_dense = encode_fused_grad(self.teacher, pos, neg)
        student_dense = F.normalize(student_dense.float(), dim=1)
        teacher_dense = F.normalize(teacher_dense.float(), dim=1)
        return (1.0 - (student_dense * teacher_dense).sum(dim=1)).mean()
