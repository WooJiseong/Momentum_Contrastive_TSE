from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from pn_soft_indiv_mococo.paths import add_repo_paths

add_repo_paths()

from pn_indiv_mococo.moco_encoder import IndividualNegativeMoCo, encode_fused


class SoftIndividualNegativeMoCo(IndividualNegativeMoCo):
    """Individual-negative MoCo with a frozen initial-PN representation anchor."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.teacher = copy.deepcopy(self.student)
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher.eval()
        self.teacher_loss_weight = float(cfg["contrastive"].get("teacher_loss_weight", 0.1))

    def teacher_loss(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        student_dense = encode_fused(self.student, pos, neg)
        with torch.no_grad():
            teacher_dense = encode_fused(self.teacher, pos, neg)
        student_dense = F.normalize(student_dense.float(), dim=1)
        teacher_dense = F.normalize(teacher_dense.float(), dim=1)
        return (1.0 - (student_dense * teacher_dense).sum(dim=1)).mean()
