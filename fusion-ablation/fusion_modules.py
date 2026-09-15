import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


# ==============================================================================
# F0: Hierarchical Linear Concatenation Fusion (Base V6)
# ==============================================================================
class LinearFusion(nn.Module):
    """
    Standard Hierarchical 2-Stage Linear Concatenation Projection.
    z_RNA = Linear([z_sim || z_dist])
    Z = Linear([z_RNA || z_aux])
    """
    def __init__(self, out_dim: int = 64):
        super(LinearFusion, self).__init__()
        self.fusion1 = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.fusion2 = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim)
        )

    def forward(self, x_sim: torch.Tensor, x_dist: torch.Tensor, x_aux: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        fused_rna = self.fusion1(torch.cat([x_sim, x_dist], dim=1))
        fused_joint = self.fusion2(torch.cat([fused_rna, x_aux], dim=1))
        return fused_joint, fused_rna


# ==============================================================================
# F1: Local-Global Attention Smoothing Fusion (from V3 / spaFusion)
# ==============================================================================
class LocalGlobalAttentionFusion(nn.Module):
    """
    Local physical spatial 1-hop smoothing followed by global spot affinity self-attention.
    z_l = A_spatial * z_intra
    S = Softmax(z_l * z_l^T / sqrt(d)),  z_g = S * z_l
    z_smoothed = alpha * z_g + z_l
    """
    def __init__(self, out_dim: int = 64, alpha: float = 0.5):
        super(LocalGlobalAttentionFusion, self).__init__()
        self.alpha = alpha
        self.fusion_intra = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.fusion_inter = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim)
        )

    def forward(self, x_sim: torch.Tensor, x_dist: torch.Tensor, x_aux: torch.Tensor,
                spatial_adj: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Intra-RNA combination
        z_intra = self.fusion_intra(torch.cat([x_sim, x_dist], dim=1))
        
        # Local spatial smoothing
        if spatial_adj is not None:
            if spatial_adj.is_sparse:
                z_local = torch.sparse.mm(spatial_adj, z_intra)
            else:
                z_local = torch.mm(spatial_adj, z_intra)
        else:
            z_local = z_intra

        # Global self-attention smoothing
        d = z_local.shape[1]
        attn_scores = torch.mm(z_local, z_local.t()) / math.sqrt(d)
        attn_weights = F.softmax(attn_scores, dim=-1)
        z_global = torch.mm(attn_weights, z_local)

        fused_rna = self.alpha * z_global + (1.0 - self.alpha) * z_local
        fused_joint = self.fusion_inter(torch.cat([fused_rna, x_aux], dim=1))
        return fused_joint, fused_rna


# ==============================================================================
# F2: Dynamic Variance-Weighted Modal Fusion (from V4 / spaFusion)
# ==============================================================================
class DynamicVarianceWeightingFusion(nn.Module):
    """
    Calculates channel variance per modality embedding: Var(z_m) = (1/D) * sum(Var(z_m[:, d]))
    Weights modalities via Softmax(Var(z_m)) and computes variance-weighted convex combination.
    """
    def __init__(self, out_dim: int = 64, temp: float = 1.0):
        super(DynamicVarianceWeightingFusion, self).__init__()
        self.temp = temp
        self.fusion_rna = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.proj_out = nn.Linear(out_dim, out_dim)

    def forward(self, x_sim: torch.Tensor, x_dist: torch.Tensor, x_aux: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        fused_rna = self.fusion_rna(torch.cat([x_sim, x_dist], dim=1))

        var_rna = torch.var(fused_rna, dim=0).mean()
        var_aux = torch.var(x_aux, dim=0).mean()

        vars_stack = torch.stack([var_rna, var_aux]) / self.temp
        weights = F.softmax(vars_stack, dim=0)

        fused_joint = weights[0] * fused_rna + weights[1] * x_aux
        fused_joint = self.proj_out(fused_joint)
        return fused_joint, fused_rna


# ==============================================================================
# F3: Graph-Masked Spatial Cross-Attention Fusion (from V11)
# ==============================================================================
class GraphMaskedSpatialCrossAttentionFusion(nn.Module):
    """
    Topology-constrained spatial cross-attention between RNA and Aux modalities:
    Attn_ij = (Q_i * K_j^T) / sqrt(d) + Mask_ij, where Mask_ij = 0 for 1-hop physical neighbors, -inf elsewhere.
    Directly prevents global over-smoothing while enabling cross-modal feature exchange.
    """
    def __init__(self, out_dim: int = 64, heads: int = 2):
        super(GraphMaskedSpatialCrossAttentionFusion, self).__init__()
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.fusion_rna = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

        self.q_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)

        self.joint_linear = nn.Sequential(
            nn.Linear(3 * out_dim, out_dim)
        )

    def forward(self, x_sim: torch.Tensor, x_dist: torch.Tensor, x_aux: torch.Tensor,
                dist_edge_index: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        fused_rna = self.fusion_rna(torch.cat([x_sim, x_dist], dim=1))
        N, D = fused_rna.shape

        q = self.q_proj(fused_rna).view(N, self.heads, self.head_dim).transpose(0, 1) # [heads, N, head_dim]
        k = self.k_proj(x_aux).view(N, self.heads, self.head_dim).transpose(0, 1)     # [heads, N, head_dim]
        v = self.v_proj(x_aux).view(N, self.heads, self.head_dim).transpose(0, 1)     # [heads, N, head_dim]

        # Dot product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # [heads, N, N]

        # Apply 1-hop spatial physical graph mask
        if dist_edge_index is not None and N < 15000:
            mask = torch.full((N, N), float('-inf'), device=fused_rna.device)
            mask[dist_edge_index[0], dist_edge_index[1]] = 0.0
            mask.fill_diagonal_(0.0)
            attn_scores = attn_scores + mask.unsqueeze(0)

        attn_weights = F.softmax(attn_scores, dim=-1)
        z_cross = torch.matmul(attn_weights, v).transpose(0, 1).contiguous().view(N, D)
        z_cross = self.out_proj(z_cross)

        # Multi-modal joint projection
        fused_joint = self.joint_linear(torch.cat([fused_rna, z_cross, x_aux], dim=1))
        return fused_joint, fused_rna


# ==============================================================================
# F4: Bilinear Outer-Product Tensor Fusion
# ==============================================================================
class BilinearTensorFusion(nn.Module):
    """
    Computes outer product interaction tensor: (z_RNA [x] z_aux) via low-rank factorized bilinear pooling.
    Captures non-linear cross-modal co-expression correlations without prohibitive memory blowup.
    """
    def __init__(self, out_dim: int = 64, rank: int = 32):
        super(BilinearTensorFusion, self).__init__()
        self.fusion_rna = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.proj_r = nn.Linear(out_dim, rank, bias=False)
        self.proj_a = nn.Linear(out_dim, rank, bias=False)
        self.proj_out = nn.Sequential(
            nn.Linear(rank + 2 * out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x_sim: torch.Tensor, x_dist: torch.Tensor, x_aux: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        fused_rna = self.fusion_rna(torch.cat([x_sim, x_dist], dim=1))
        
        # Low-rank outer product approximation
        hr = self.proj_r(fused_rna)
        ha = self.proj_a(x_aux)
        tensor_interaction = hr * ha  # Hadamard product of projected rank representations

        fused_joint = self.proj_out(torch.cat([fused_rna, tensor_interaction, x_aux], dim=1))
        return fused_joint, fused_rna


# ==============================================================================
# F5: Learnable Gated Multi-Modal Fusion (from CAGE / SpatialGlue)
# ==============================================================================
class GatedMultiModalFusion(nn.Module):
    """
    Learnable elementwise Sigmoid Gate (from CAGE & SpatialGlue):
    gate = Sigmoid(W_RNA * z_RNA + W_aux * z_aux)
    Z = gate * z_RNA + (1 - gate) * z_aux
    """
    def __init__(self, out_dim: int = 64):
        super(GatedMultiModalFusion, self).__init__()
        self.fusion_rna = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.fc_rna = nn.Linear(out_dim, out_dim, bias=True)
        self.fc_aux = nn.Linear(out_dim, out_dim, bias=True)
        self.post_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

        nn.init.xavier_uniform_(self.fc_rna.weight)
        nn.init.xavier_uniform_(self.fc_aux.weight)
        nn.init.zeros_(self.fc_rna.bias)
        nn.init.zeros_(self.fc_aux.bias)

    def forward(self, x_sim: torch.Tensor, x_dist: torch.Tensor, x_aux: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        fused_rna = self.fusion_rna(torch.cat([x_sim, x_dist], dim=1))

        # Dynamic elementwise gating
        gate = torch.sigmoid(self.fc_rna(fused_rna) + self.fc_aux(x_aux))
        gated_repr = gate * fused_rna + (1.0 - gate) * x_aux

        fused_joint = self.post_proj(gated_repr)
        return fused_joint, fused_rna


# ==============================================================================
# F6: Bidirectional Symmetric QKV Cross-Fusion (from CAGE)
# ==============================================================================
class QKVCrossFusion(nn.Module):
    """
    Bidirectional Symmetric QKV Cross-Attention Fusion (from CAGE):
    Projects both modalities into Q, K, V spaces and performs bidirectional cross-modal interaction:
    - 'local' mode: within-spot cross-gating:
        score1 = (Q_aux * K_rna).sum(dim=-1) / sqrt(d)
        score2 = (Q_rna * K_aux).sum(dim=-1) / sqrt(d)
        w1 = sigmoid(score1), w2 = sigmoid(score2)
        z1 = w1 * V_rna, z2 = w2 * V_aux
    - 'global' mode: full N x N cross-attention:
        z1 = Softmax(Q_aux * K_rna^T / sqrt(d)) * V_rna
        z2 = Softmax(Q_rna * K_aux^T / sqrt(d)) * V_aux
    Z = Linear([z1 || z2])
    """
    def __init__(self, out_dim: int = 64, attention_type: str = 'local'):
        super(QKVCrossFusion, self).__init__()
        self.out_dim = out_dim
        self.attention_type = attention_type
        self.scale = out_dim ** -0.5

        self.fusion_rna = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

        self.q_proj1 = nn.Linear(out_dim, out_dim, bias=False)
        self.k_proj1 = nn.Linear(out_dim, out_dim, bias=False)
        self.v_proj1 = nn.Linear(out_dim, out_dim, bias=False)

        self.q_proj2 = nn.Linear(out_dim, out_dim, bias=False)
        self.k_proj2 = nn.Linear(out_dim, out_dim, bias=False)
        self.v_proj2 = nn.Linear(out_dim, out_dim, bias=False)

        self.fc_out = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj1.weight)
        nn.init.xavier_uniform_(self.k_proj1.weight)
        nn.init.xavier_uniform_(self.v_proj1.weight)
        nn.init.xavier_uniform_(self.q_proj2.weight)
        nn.init.xavier_uniform_(self.k_proj2.weight)
        nn.init.xavier_uniform_(self.v_proj2.weight)

    def forward(self, x_sim: torch.Tensor, x_dist: torch.Tensor, x_aux: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        fused_rna = self.fusion_rna(torch.cat([x_sim, x_dist], dim=1))

        q1 = self.q_proj1(fused_rna)
        k1 = self.k_proj1(fused_rna)
        v1 = self.v_proj1(fused_rna)

        q2 = self.q_proj2(x_aux)
        k2 = self.k_proj2(x_aux)
        v2 = self.v_proj2(x_aux)

        if self.attention_type == 'global':
            attn_scores1 = torch.matmul(q2, k1.t()) * self.scale
            attn_probs1 = F.softmax(attn_scores1, dim=-1)
            z1 = torch.matmul(attn_probs1, v1)

            attn_scores2 = torch.matmul(q1, k2.t()) * self.scale
            attn_probs2 = F.softmax(attn_scores2, dim=-1)
            z2 = torch.matmul(attn_probs2, v2)
        else:
            # Local cross-gating within spot
            score1 = (q2 * k1).sum(dim=-1, keepdim=True) * self.scale
            score2 = (q1 * k2).sum(dim=-1, keepdim=True) * self.scale

            w1 = torch.sigmoid(score1)
            w2 = torch.sigmoid(score2)

            z1 = w1 * v1
            z2 = w2 * v2

        fused_joint = self.fc_out(torch.cat([z1, z2], dim=-1))
        return fused_joint, fused_rna

