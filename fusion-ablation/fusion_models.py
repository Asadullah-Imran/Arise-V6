import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fusion_modules import (
    LinearFusion,
    LocalGlobalAttentionFusion,
    DynamicVarianceWeightingFusion,
    GraphMaskedSpatialCrossAttentionFusion,
    BilinearTensorFusion,
    GatedMultiModalFusion,
    QKVCrossFusion
)

# Shared Loss & DEC distribution helpers
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'loss-ablation'))
from loss_modules import (
    QDistribution,
    compute_target_distribution,
    consensus_distribution_loss,
    spatial_regularization_loss,
    parameter_regularization_loss
)


class V6FusionAblationModel(nn.Module):
    """
    Unified V6 Architecture with customizable fusion layer (F0 - F5).
    """
    def __init__(
        self,
        fusion_type: str,
        in_rna_dim: int,
        in_aux_dim: int,
        num_clusters: int,
        hidden_dim: int = 512,
        out_dim: int = 64,
        beta: float = 25.0,
        gamma: float = 10.0,
        delta: float = 1.0,
        lambda_kl: float = 0.1,
        **kwargs
    ):
        super(V6FusionAblationModel, self).__init__()
        self.fusion_type = fusion_type
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lambda_kl = lambda_kl

        # V6 Standard Encoders
        self.x_RNA1 = GCNConv(in_rna_dim, hidden_dim)
        self.x_RNA2 = GCNConv(in_rna_dim, hidden_dim)
        self.sim_conv = GCNConv(hidden_dim, out_dim)
        self.dist_conv = GCNConv(hidden_dim, out_dim)
        self.aux_conv = GCNConv(in_aux_dim, out_dim)

        # Pluggable Fusion Layer
        if fusion_type == 'F0_Linear':
            self.fusion = LinearFusion(out_dim=out_dim)
        elif fusion_type == 'F1_LocalGlobal':
            self.fusion = LocalGlobalAttentionFusion(out_dim=out_dim, alpha=0.5)
        elif fusion_type == 'F2_VarianceWeight':
            self.fusion = DynamicVarianceWeightingFusion(out_dim=out_dim)
        elif fusion_type == 'F3_SpatialCrossAttention':
            self.fusion = GraphMaskedSpatialCrossAttentionFusion(out_dim=out_dim, heads=2)
        elif fusion_type == 'F4_BilinearTensor':
            self.fusion = BilinearTensorFusion(out_dim=out_dim, rank=32)
        elif fusion_type == 'F5_GatedMultiModal':
            self.fusion = GatedMultiModalFusion(out_dim=out_dim)
        elif fusion_type == 'F6_QKVCrossFusion':
            self.fusion = QKVCrossFusion(out_dim=out_dim, attention_type='local')
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # V6 Consensus DEC Kernel
        self.q_dist = QDistribution(num_clusters=num_clusters, embed_dim=out_dim)

        # Decoders
        self.deconv1 = nn.Linear(out_dim, hidden_dim)
        self.deconv_rna = nn.Linear(hidden_dim, in_rna_dim)
        self.deconv_aux = nn.Linear(hidden_dim, in_aux_dim)
        self.deconv_joint = nn.Linear(hidden_dim, in_rna_dim + in_aux_dim)

    def set_cluster_centers(self, centers_np):
        dev = next(self.parameters()).device
        self.q_dist.cluster_centers.data = torch.tensor(centers_np, dtype=torch.float, device=dev)

    def forward(self, data, compute_q: bool = False) -> Dict[str, torch.Tensor]:
        # Encoders
        x_RNA1 = F.relu(self.x_RNA1(data.x_RNA, data.sim_edge_index, data.sim_edge_weight))
        x_sim = self.sim_conv(x_RNA1, data.sim_edge_index, data.sim_edge_weight)

        x_RNA2 = F.relu(self.x_RNA2(data.x_RNA, data.dist_edge_index, data.dist_edge_weight))
        x_dist = self.dist_conv(x_RNA2, data.dist_edge_index, data.dist_edge_weight)

        x_aux = self.aux_conv(data.x_ADT, data.common_edge_index, data.common_edge_weight)

        # Spatial matrix for modules that require it
        sp_adj = getattr(data, 'spatial_adj', None)
        dist_edge_idx = getattr(data, 'dist_edge_index', None)

        # Pluggable Fusion
        fused_joint, fused_rna = self.fusion(
            x_sim, x_dist, x_aux,
            spatial_adj=sp_adj,
            dist_edge_index=dist_edge_idx
        )

        q_list = self.q_dist([fused_joint, fused_rna, x_aux]) if compute_q else None

        return {
            'embedding': fused_joint,
            'fused_joint': fused_joint,
            'fused_rna': fused_rna,
            'x_sim': x_sim,
            'x_dist': x_dist,
            'x_aux': x_aux,
            'q_list': q_list
        }

    def compute_loss(self, data, outputs: Dict[str, torch.Tensor], stage: int = 1) -> Tuple[torch.Tensor, Dict[str, float]]:
        num_nodes = data.x_RNA.shape[0]
        l_rec = F.mse_loss(torch.cat([data.x_RNA, data.x_ADT], dim=1), self.deconv_joint(F.relu(self.deconv1(outputs['fused_joint']))))
        l_sim = F.mse_loss(data.x_RNA, self.deconv_rna(F.relu(self.deconv1(outputs['x_sim']))))
        l_dist = F.mse_loss(data.x_RNA, self.deconv_rna(F.relu(self.deconv1(outputs['x_dist']))))
        l_aux = F.mse_loss(data.x_ADT, self.deconv_aux(F.relu(self.deconv1(outputs['x_aux']))))
        total_recon = l_rec + l_sim + l_dist + l_aux

        l_spatial = spatial_regularization_loss(outputs['fused_rna'], data.dist_edge_index, data.dist_edge_weight, num_nodes)
        l_reg = parameter_regularization_loss(self)
        total_loss = self.beta * total_recon + self.gamma * l_spatial + self.delta * l_reg

        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_recon': total_recon.item(),
            'loss_spatial': l_spatial.item(),
            'loss_kl': 0.0
        }

        if stage == 2 and outputs['q_list'] is not None:
            Q_joint = outputs['q_list'][0]
            target_P = compute_target_distribution(Q_joint.detach())
            l_kl = consensus_distribution_loss(outputs['q_list'], target_P)
            total_loss = total_loss + self.lambda_kl * l_kl
            loss_dict['loss_kl'] = l_kl.item()
            loss_dict['loss_total'] = total_loss.item()

        return total_loss, loss_dict


# Model Factory
FUSION_MODEL_REGISTRY = {
    'F0_Linear': lambda in_r, in_a, nc, **kw: V6FusionAblationModel('F0_Linear', in_r, in_a, nc, **kw),
    'F1_LocalGlobal': lambda in_r, in_a, nc, **kw: V6FusionAblationModel('F1_LocalGlobal', in_r, in_a, nc, **kw),
    'F2_VarianceWeight': lambda in_r, in_a, nc, **kw: V6FusionAblationModel('F2_VarianceWeight', in_r, in_a, nc, **kw),
    'F3_SpatialCrossAttention': lambda in_r, in_a, nc, **kw: V6FusionAblationModel('F3_SpatialCrossAttention', in_r, in_a, nc, **kw),
    'F4_BilinearTensor': lambda in_r, in_a, nc, **kw: V6FusionAblationModel('F4_BilinearTensor', in_r, in_a, nc, **kw),
    'F5_GatedMultiModal': lambda in_r, in_a, nc, **kw: V6FusionAblationModel('F5_GatedMultiModal', in_r, in_a, nc, **kw),
    'F6_QKVCrossFusion': lambda in_r, in_a, nc, **kw: V6FusionAblationModel('F6_QKVCrossFusion', in_r, in_a, nc, **kw),
}
