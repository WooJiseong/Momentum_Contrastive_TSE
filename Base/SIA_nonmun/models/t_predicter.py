import torch
import torch.nn as nn
from .ecapa_tdnn import ECAPA_TDNN

class TPredicter(nn.Module):
    def __init__(self, C):
        super(TPredicter, self).__init__()
        self.ecapa_tdnn = ECAPA_TDNN(C=C)
        self.output_activ = nn.Sigmoid()
        self.output_layer = nn.Sequential(
            nn.Linear(192 * 2, 192),
            nn.SiLU(),
            nn.Linear(192, 1),
        )

    def forward(self, mixture, enrollment, aug=False):
        """
        Args:
            mixture (torch.Tensor): Noisy input tensor of shape (batch_size, time_steps)
            enrollment (torch.Tensor): Enrollment tensor of shape (batch_size, time_steps)
        
        Returns:
            torch.Tensor: Predicted tensor of shape (batch_size, C)
        """
        # Pass through ECAPA-TDNN
        enrollment_feat = self.ecapa_tdnn(enrollment, aug)
        mixture_feat = self.ecapa_tdnn(mixture, aug)
        sqrt_d = enrollment_feat.shape[1] ** 0.5
        enrollment_feat = enrollment_feat / sqrt_d
        mixture_feat = mixture_feat / sqrt_d

        # simularity = torch.einsum('bd,bd->b', enrollment_feat, mixture_feat)
        # simularity = self.cos_sim(enrollment_feat, mixture_feat)
        simularity = torch.cat([enrollment_feat, mixture_feat], dim=-1)
        simularity = self.output_layer(simularity).squeeze(-1)
        t = self.output_activ(simularity)

        return t


class TPredicterPN(nn.Module):
    """IDENTICAL to TPredicter EXCEPT the enrollment branch: instead of running ECAPA-TDNN on a clean
    positive utterance, the enrollment feature comes from OUR frozen PN encoder output (pos vs neg
    contrast), projected to the 192-d ECAPA embedding space. The mixture branch (ECAPA-TDNN), the
    L2-style normalization, the concat head, and the Sigmoid output are byte-identical to TPredicter.
    The frozen PN encoder lives OUTSIDE this module (in the LightningModule); forward() consumes its
    precomputed embedding `enroll_emb` of shape [B, 64, Tpn, 65] (== pn_encoder.encode(pos, neg))."""

    def __init__(self, C, pn_in_ch: int = 64, pn_in_freq: int = 65, emb_dim: int = 192,
                 pool_mode: str = "mean"):
        super(TPredicterPN, self).__init__()
        assert pool_mode in ("mean", "attn_stats", "cross_attn"), pool_mode
        self.pool_mode = pool_mode
        self.ecapa_tdnn = ECAPA_TDNN(C=C)                                  # mixture branch (UNCHANGED across modes)
        Cpn = pn_in_ch * pn_in_freq                                        # 4160
        if pool_mode == "mean":
            self.enroll_proj = nn.Linear(Cpn, emb_dim)                     # mean over Tpn -> Linear
        elif pool_mode == "attn_stats":
            # ECAPA-style attentive statistics pooling over Tpn (symmetric with the mixture ECAPA branch)
            self.asp_attn = nn.Sequential(nn.Conv1d(Cpn, 128, 1), nn.Tanh(), nn.Conv1d(128, Cpn, 1))
            self.enroll_proj = nn.Linear(Cpn * 2, emb_dim)                 # [mean|std] -> Linear
        else:  # cross_attn: a learned query attends over the Tpn token sequence
            self.tok_proj = nn.Linear(Cpn, emb_dim)
            self.query = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
            self.mha = nn.MultiheadAttention(emb_dim, num_heads=4, batch_first=True)
        self.output_activ = nn.Sigmoid()
        self.output_layer = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 1),
        )

    def _enroll_feat(self, enroll_emb):
        """enroll_emb [B,64,Tpn,65] -> [B,emb_dim], per pool_mode."""
        B = enroll_emb.shape[0]
        if self.pool_mode == "mean":
            e = enroll_emb.mean(dim=2).reshape(B, -1)                      # [B,4160]
            return self.enroll_proj(e)
        # reshape to a [B, Cpn, Tpn] sequence (Cpn = 64*65)
        x = enroll_emb.permute(0, 1, 3, 2).reshape(B, -1, enroll_emb.shape[2])  # [B,4160,Tpn]
        if self.pool_mode == "attn_stats":
            w = torch.softmax(self.asp_attn(x), dim=2)                    # [B,4160,Tpn]
            mu = torch.sum(w * x, dim=2)                                   # [B,4160]
            sg = torch.sqrt((torch.sum(w * x * x, dim=2) - mu * mu).clamp_min(1e-6))
            return self.enroll_proj(torch.cat([mu, sg], dim=1))           # [B,emb_dim]
        else:  # cross_attn
            tok = self.tok_proj(x.transpose(1, 2))                        # [B,Tpn,emb_dim]
            q = self.query.expand(B, 1, -1)
            return self.mha(q, tok, tok)[0].squeeze(1)                    # [B,emb_dim]

    def forward(self, mixture, enroll_emb, aug=False):
        """
        Args:
            mixture (torch.Tensor): mixture waveform, shape (B, T).
            enroll_emb (torch.Tensor): frozen PN-encoder output, shape (B, 64, Tpn, 65).
        Returns:
            torch.Tensor: predicted mixing ratio t in (0,1), shape (B,).
        """
        mixture_feat = self.ecapa_tdnn(mixture, aug)                       # [B,192]  (UNCHANGED branch)
        enrollment_feat = self._enroll_feat(enroll_emb)                    # [B,192]  (pool_mode-dependent)
        sqrt_d = enrollment_feat.shape[1] ** 0.5
        enrollment_feat = enrollment_feat / sqrt_d
        mixture_feat = mixture_feat / sqrt_d
        simularity = torch.cat([enrollment_feat, mixture_feat], dim=-1)    # head IDENTICAL to TPredicter
        simularity = self.output_layer(simularity).squeeze(-1)
        t = self.output_activ(simularity)
        return t