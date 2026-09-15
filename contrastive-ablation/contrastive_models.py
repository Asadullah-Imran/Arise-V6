import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from contrastive_modules import (
    CrossModalContrastiveLoss,
    ProustDGIContrastiveLoss,
    SpatialNeighborContrastiveLoss,
    ClusterAwareContrastiveLoss,
    GraphAugmentationConsistencyLoss,
    HybridMultiLevelContrastiveLoss
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


class V6ContrastiveAblationModel(nn.Module):
    """
    Unified V6 Architecture with customizable Contrastive Learning Objectives (C0 - C6).
    """
    def __init__(
        self,
        cl_type: str,
        in_rna_dim: int,
        in_aux_dim: int,
        num_clusters: int,
        hidden_dim: int = 512,
        out_dim: int = 64,
        beta: float = 25.0,
        gamma: float = 10.0,
        delta: float = 1.0,
        lambda_kl: float = 0.1,
        lambda_cl: float = 1.0,
        **kwargs
    ):
        super(V6ContrastiveAblationModel, self).__init__()
        self.cl_type = cl_type
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lambda_kl = lambda_kl
        self.lambda_cl = lambda_cl

        # Base V6 Standard Encoders
        self.x_RNA1 = GCNConv(in_rna_dim, hidden_dim)
        self.x_RNA2 = GCNConv(in_rna_dim, hidden_dim)
        self.sim_conv = GCNConv(hidden_dim, out_dim)
        self.dist_conv = GCNConv(hidden_dim, out_dim)
        self.aux_conv = GCNConv(in_aux_dim, out_dim)

        # Base V6 Hierarchical Linear Fusion
        self.fusion1 = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.fusion2 = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim)
        )

        # Pluggable Contrastive Learning Modules
        if cl_type == 'C1_CrossModalCL':
            self.cl_module = CrossModalContrastiveLoss(init_tau=0.1, learnable_tau=True)
        elif cl_type == 'C2_ProustDGI':
            self.cl_module = ProustDGIContrastiveLoss(embed_dim=out_dim)
        elif cl_type == 'C3_SpatialNeighborCL':
            self.cl_module = SpatialNeighborContrastiveLoss(temperature=0.1)
        elif cl_type == 'C4_ClusterAwareCL':
            self.cl_module = ClusterAwareContrastiveLoss(temperature=0.1)
        elif cl_type == 'C5_GraphAugConsistency':
            self.cl_module = GraphAugmentationConsistencyLoss(temperature=0.1)
        elif cl_type == 'C6_HybridMultiLevelCL':
            self.cl_module = HybridMultiLevelContrastiveLoss(lambda_cross=1.0, lambda_spatial=0.5, lambda_cluster=0.2)

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
        xs = F.relu(self.x_RNA1(data.x_RNA, data.sim_edge_index, data.sim_edge_weight))
        x_sim = self.sim_conv(xs, data.sim_edge_index, data.sim_edge_weight)

        xd = F.relu(self.x_RNA2(data.x_RNA, data.dist_edge_index, data.dist_edge_weight))
        x_dist = self.dist_conv(xd, data.dist_edge_index, data.dist_edge_weight)

        x_aux = self.aux_conv(data.x_ADT, data.common_edge_index, data.common_edge_weight)

        fused_rna = self.fusion1(torch.cat([x_sim, x_dist], dim=1))
        fused_joint = self.fusion2(torch.cat([fused_rna, x_aux], dim=1))

        # Auxiliary forward pass for dual-view or corruption if needed
        out_corrupt = None
        out_aug2 = None

        if self.cl_type == 'C2_ProustDGI':
            # Proust corrupted graph forward pass: shuffle features randomly
            perm = torch.randperm(data.x_RNA.shape[0])
            x_RNA_shuf = data.x_RNA[perm]
            x_ADT_shuf = data.x_ADT[perm]

            xs_c = F.relu(self.x_RNA1(x_RNA_shuf, data.sim_edge_index, data.sim_edge_weight))
            x_sim_c = self.sim_conv(xs_c, data.sim_edge_index, data.sim_edge_weight)

            xd_c = F.relu(self.x_RNA2(x_RNA_shuf, data.dist_edge_index, data.dist_edge_weight))
            x_dist_c = self.dist_conv(xd_c, data.dist_edge_index, data.dist_edge_weight)

            x_aux_c = self.aux_conv(x_ADT_shuf, data.common_edge_index, data.common_edge_weight)
            fused_rna_c = self.fusion1(torch.cat([x_sim_c, x_dist_c], dim=1))
            out_corrupt = self.fusion2(torch.cat([fused_rna_c, x_aux_c], dim=1))

        elif self.cl_type == 'C5_GraphAugConsistency':
            # Perturbed graph view 2: random feature dropout
            mask = (torch.rand_like(data.x_RNA) > 0.1).float()
            x_RNA_aug = data.x_RNA * mask
            xs_a2 = F.relu(self.x_RNA1(x_RNA_aug, data.sim_edge_index, data.sim_edge_weight))
            x_sim_a2 = self.sim_conv(xs_a2, data.sim_edge_index, data.sim_edge_weight)
            xd_a2 = F.relu(self.x_RNA2(x_RNA_aug, data.dist_edge_index, data.dist_edge_weight))
            x_dist_a2 = self.dist_conv(xd_a2, data.dist_edge_index, data.dist_edge_weight)
            x_aux_a2 = self.aux_conv(data.x_ADT, data.common_edge_index, data.common_edge_weight)
            fused_rna_a2 = self.fusion1(torch.cat([x_sim_a2, x_dist_a2], dim=1))
            out_aug2 = self.fusion2(torch.cat([fused_rna_a2, x_aux_a2], dim=1))

        q_list = self.q_dist([fused_joint, fused_rna, x_aux]) if compute_q else None

        return {
            'x_sim': x_sim,
            'x_dist': x_dist,
            'fused_rna': fused_rna,
            'x_aux': x_aux,
            'fused_joint': fused_joint,
            'embedding': fused_joint,
            'out_corrupt': out_corrupt,
            'out_aug2': out_aug2,
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
            'loss_recon': total_recon.item(),
            'loss_spatial': l_spatial.item(),
            'loss_cl': 0.0,
            'loss_kl': 0.0
        }

        # Pluggable Contrastive Objective
        if self.cl_type == 'C1_CrossModalCL':
            l_cl = self.cl_module(outputs['fused_rna'], outputs['x_aux'])
            total_loss = total_loss + self.lambda_cl * l_cl
            loss_dict['loss_cl'] = l_cl.item()

        elif self.cl_type == 'C2_ProustDGI':
            if outputs['out_corrupt'] is not None:
                l_cl = self.cl_module(outputs['fused_joint'], outputs['out_corrupt'], data.dist_edge_index)
                total_loss = total_loss + self.lambda_cl * l_cl
                loss_dict['loss_cl'] = l_cl.item()

        elif self.cl_type == 'C3_SpatialNeighborCL':
            l_cl = self.cl_module(outputs['fused_joint'], data.dist_edge_index)
            total_loss = total_loss + self.lambda_cl * l_cl
            loss_dict['loss_cl'] = l_cl.item()

        elif self.cl_type == 'C4_ClusterAwareCL':
            if stage == 2 and outputs['q_list'] is not None:
                l_cl = self.cl_module(outputs['fused_joint'], outputs['q_list'][0])
                total_loss = total_loss + self.lambda_cl * l_cl
                loss_dict['loss_cl'] = l_cl.item()

        elif self.cl_type == 'C5_GraphAugConsistency':
            if outputs['out_aug2'] is not None:
                l_cl = self.cl_module(outputs['fused_joint'], outputs['out_aug2'])
                total_loss = total_loss + self.lambda_cl * l_cl
                loss_dict['loss_cl'] = l_cl.item()

        elif self.cl_type == 'C6_HybridMultiLevelCL':
            q_soft = outputs['q_list'][0] if (stage == 2 and outputs['q_list'] is not None) else None
            l_cl, cl_dict = self.cl_module(
                outputs['fused_rna'],
                outputs['x_aux'],
                outputs['fused_joint'],
                data.dist_edge_index,
                q_soft
            )
            total_loss = total_loss + self.lambda_cl * l_cl
            loss_dict.update(cl_dict)
            loss_dict['loss_cl'] = l_cl.item()

        # Stage 2 Consensus DEC KL loss
        if stage == 2 and outputs['q_list'] is not None:
            Q_joint = outputs['q_list'][0]
            target_P = compute_target_distribution(Q_joint.detach())
            l_kl = consensus_distribution_loss(outputs['q_list'], target_P)
            total_loss = total_loss + self.lambda_kl * l_kl
            loss_dict['loss_kl'] = l_kl.item()

        loss_dict['loss_total'] = total_loss.item()
        return total_loss, loss_dict


# Model Factory
CONTRASTIVE_MODEL_REGISTRY = {
    'C0_BaseV6': lambda in_r, in_a, nc, **kw: V6ContrastiveAblationModel('C0_BaseV6', in_r, in_a, nc, **kw),
    'C1_CrossModalCL': lambda in_r, in_a, nc, **kw: V6ContrastiveAblationModel('C1_CrossModalCL', in_r, in_a, nc, **kw),
    'C2_ProustDGI': lambda in_r, in_a, nc, **kw: V6ContrastiveAblationModel('C2_ProustDGI', in_r, in_a, nc, **kw),
    'C3_SpatialNeighborCL': lambda in_r, in_a, nc, **kw: V6ContrastiveAblationModel('C3_SpatialNeighborCL', in_r, in_a, nc, **kw),
    'C4_ClusterAwareCL': lambda in_r, in_a, nc, **kw: V6ContrastiveAblationModel('C4_ClusterAwareCL', in_r, in_a, nc, **kw),
    'C5_GraphAugConsistency': lambda in_r, in_a, nc, **kw: V6ContrastiveAblationModel('C5_GraphAugConsistency', in_r, in_a, nc, **kw),
    'C6_HybridMultiLevelCL': lambda in_r, in_a, nc, **kw: V6ContrastiveAblationModel('C6_HybridMultiLevelCL', in_r, in_a, nc, **kw),
}
