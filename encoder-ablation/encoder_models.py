import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from encoder_modules import (
    StandardGCNEncoder,
    TransformerTokenEncoder,
    HeatDiffusionWaveletGraph,
    GATv2ResidualEncoder,
    compute_3_node_motifs,
    blend_3_node_motifs,
    compute_4_node_cycle_motifs,
    blend_higher_order_motifs
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


class V6EncoderAblationModel(nn.Module):
    """
    Unified V6 Architecture with customizable encoder backbones & graph topologies (E0 - E5).
    """
    def __init__(
        self,
        encoder_type: str,
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
        super(V6EncoderAblationModel, self).__init__()
        self.encoder_type = encoder_type
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lambda_kl = lambda_kl

        # Pluggable Encoder Backbones
        if encoder_type in ['E0_StandardGCN', 'E2_TriangularMotif', 'E3_HigherOrderMotif']:
            # Standard or Motif-Enriched GCNs
            self.x_RNA1 = GCNConv(in_rna_dim, hidden_dim)
            self.x_RNA2 = GCNConv(in_rna_dim, hidden_dim)
            self.sim_conv = GCNConv(hidden_dim, out_dim)
            self.dist_conv = GCNConv(hidden_dim, out_dim)
            self.aux_conv = GCNConv(in_aux_dim, out_dim)

        elif encoder_type == 'E1_GlobalTransformer':
            # Transformer Token Encoders + GCN Hybrid (from V1)
            self.x_RNA1 = GCNConv(in_rna_dim, hidden_dim)
            self.x_RNA2 = GCNConv(in_rna_dim, hidden_dim)
            self.sim_conv = GCNConv(hidden_dim, out_dim)
            self.dist_conv = GCNConv(hidden_dim, out_dim)
            self.aux_conv = GCNConv(in_aux_dim, out_dim)
            self.trans_rna = TransformerTokenEncoder(in_rna_dim, embed_dim=out_dim)
            self.trans_aux = TransformerTokenEncoder(in_aux_dim, embed_dim=out_dim)
            self.rna_trans_fuse = nn.Linear(3 * out_dim, out_dim)
            self.aux_trans_fuse = nn.Linear(2 * out_dim, out_dim)

        elif encoder_type == 'E4_HeatWavelet':
            # Continuous Multiscale Spectral Heat Diffusion Wavelet Encoders (from V10)
            self.wavelet_sim = HeatDiffusionWaveletGraph(in_rna_dim, hidden_dim, out_dim, order=3, time_scale=1.0)
            self.wavelet_dist = HeatDiffusionWaveletGraph(in_rna_dim, hidden_dim, out_dim, order=3, time_scale=1.0)
            self.aux_conv = GCNConv(in_aux_dim, out_dim)

        elif encoder_type == 'E5_GATAttention':
            # GATv2 Multi-Head Attention Encoders (from SpatialGlue-GATCL)
            self.gat_sim = GATv2ResidualEncoder(in_rna_dim, hidden_dim=256, out_dim=out_dim, heads=2)
            self.gat_dist = GATv2ResidualEncoder(in_rna_dim, hidden_dim=256, out_dim=out_dim, heads=2)
            self.gat_aux = GATv2ResidualEncoder(in_aux_dim, hidden_dim=256, out_dim=out_dim, heads=2)

        # Base V6 Hierarchical Linear Fusion
        self.fusion1 = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.fusion2 = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim)
        )

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
        if self.encoder_type in ['E0_StandardGCN', 'E2_TriangularMotif', 'E3_HigherOrderMotif']:
            xs = F.relu(self.x_RNA1(data.x_RNA, data.sim_edge_index, data.sim_edge_weight))
            x_sim = self.sim_conv(xs, data.sim_edge_index, data.sim_edge_weight)

            xd = F.relu(self.x_RNA2(data.x_RNA, data.dist_edge_index, data.dist_edge_weight))
            x_dist = self.dist_conv(xd, data.dist_edge_index, data.dist_edge_weight)

            x_aux = self.aux_conv(data.x_ADT, data.common_edge_index, data.common_edge_weight)
            fused_rna = self.fusion1(torch.cat([x_sim, x_dist], dim=1))

        elif self.encoder_type == 'E1_GlobalTransformer':
            xs = F.relu(self.x_RNA1(data.x_RNA, data.sim_edge_index, data.sim_edge_weight))
            x_sim = self.sim_conv(xs, data.sim_edge_index, data.sim_edge_weight)

            xd = F.relu(self.x_RNA2(data.x_RNA, data.dist_edge_index, data.dist_edge_weight))
            x_dist = self.dist_conv(xd, data.dist_edge_index, data.dist_edge_weight)

            x_aux_gcn = self.aux_conv(data.x_ADT, data.common_edge_index, data.common_edge_weight)
            
            z_t_rna = self.trans_rna(data.x_RNA)
            z_t_aux = self.trans_aux(data.x_ADT)

            fused_rna = self.rna_trans_fuse(torch.cat([x_sim, x_dist, z_t_rna], dim=1))
            x_aux = self.aux_trans_fuse(torch.cat([x_aux_gcn, z_t_aux], dim=1))

        elif self.encoder_type == 'E4_HeatWavelet':
            spatial_adj = getattr(data, 'spatial_adj', None)
            x_sim = self.wavelet_sim(data.x_RNA, data.sim_edge_index, data.sim_edge_weight, spatial_adj)
            x_dist = self.wavelet_dist(data.x_RNA, data.dist_edge_index, data.dist_edge_weight, spatial_adj)
            x_aux = self.aux_conv(data.x_ADT, data.common_edge_index, data.common_edge_weight)
            fused_rna = self.fusion1(torch.cat([x_sim, x_dist], dim=1))

        elif self.encoder_type == 'E5_GATAttention':
            x_sim = self.gat_sim(data.x_RNA, data.sim_edge_index)
            x_dist = self.gat_dist(data.x_RNA, data.dist_edge_index)
            x_aux = self.gat_aux(data.x_ADT, data.common_edge_index)
            fused_rna = self.fusion1(torch.cat([x_sim, x_dist], dim=1))

        fused_joint = self.fusion2(torch.cat([fused_rna, x_aux], dim=1))
        q_list = self.q_dist([fused_joint, fused_rna, x_aux]) if compute_q else None

        return {
            'x_sim': x_sim,
            'x_dist': x_dist,
            'fused_rna': fused_rna,
            'x_aux': x_aux,
            'fused_joint': fused_joint,
            'embedding': fused_joint,
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
ENCODER_MODEL_REGISTRY = {
    'E0_StandardGCN': lambda in_r, in_a, nc, **kw: V6EncoderAblationModel('E0_StandardGCN', in_r, in_a, nc, **kw),
    'E1_GlobalTransformer': lambda in_r, in_a, nc, **kw: V6EncoderAblationModel('E1_GlobalTransformer', in_r, in_a, nc, **kw),
    'E2_TriangularMotif': lambda in_r, in_a, nc, **kw: V6EncoderAblationModel('E2_TriangularMotif', in_r, in_a, nc, **kw),
    'E3_HigherOrderMotif': lambda in_r, in_a, nc, **kw: V6EncoderAblationModel('E3_HigherOrderMotif', in_r, in_a, nc, **kw),
    'E4_HeatWavelet': lambda in_r, in_a, nc, **kw: V6EncoderAblationModel('E4_HeatWavelet', in_r, in_a, nc, **kw),
    'E5_GATAttention': lambda in_r, in_a, nc, **kw: V6EncoderAblationModel('E5_GATAttention', in_r, in_a, nc, **kw),
}
