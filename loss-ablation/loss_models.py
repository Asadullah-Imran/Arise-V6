import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loss_modules import (
    QDistribution,
    compute_target_distribution,
    consensus_distribution_loss,
    spatial_regularization_loss,
    parameter_regularization_loss,
    DenseRelationalAlignmentLoss,
    SinkhornOptimalTransportLoss,
    SpatialMultimodalInfoNCELoss,
    SpatialPottsRegularizer,
    UncertaintyWeightedMultiTaskLoss
)


class V6LossAblationModel(nn.Module):
    """
    Unified V6 Architecture with customizable loss objectives (L0 - L5).
    """
    def __init__(
        self,
        loss_type: str,
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
        super(V6LossAblationModel, self).__init__()
        self.loss_type = loss_type
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

        # Base V6 Hierarchical Linear Fusion
        self.fusion1 = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.fusion2 = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim)
        )

        # Pluggable Loss Modules
        if loss_type == 'L1_DenseRelational':
            self.dense_module = DenseRelationalAlignmentLoss(rel_weight=0.1)
        elif loss_type == 'L2_SinkhornOT':
            self.sinkhorn_module = SinkhornOptimalTransportLoss(eps=0.1, max_iter=40)
        elif loss_type == 'L3_SpatialInfoNCE':
            self.infonce_module = SpatialMultimodalInfoNCELoss(temperature=0.1)
        elif loss_type == 'L4_SpatialPotts':
            self.potts_module = SpatialPottsRegularizer()
        elif loss_type == 'L5_UncertaintyBalancing':
            self.uncertainty_module = UncertaintyWeightedMultiTaskLoss(num_tasks=4)

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

        loss_dict = {
            'loss_recon': total_recon.item(),
            'loss_spatial': l_spatial.item(),
            'loss_kl': 0.0,
            'loss_aux_obj': 0.0
        }

        # Pluggable Loss Formulation
        if self.loss_type == 'L0_Standard':
            total_loss = self.beta * total_recon + self.gamma * l_spatial + self.delta * l_reg

        elif self.loss_type == 'L1_DenseRelational':
            l_dense = self.dense_module(outputs['fused_joint'], outputs['fused_rna'], outputs['x_aux'])
            total_loss = self.beta * total_recon + self.gamma * l_spatial + 1.0 * l_dense + self.delta * l_reg
            loss_dict['loss_aux_obj'] = l_dense.item()

        elif self.loss_type == 'L2_SinkhornOT':
            l_ot = self.sinkhorn_module(outputs['fused_rna'], outputs['x_aux'])
            total_loss = self.beta * total_recon + self.gamma * l_spatial + 0.5 * l_ot + self.delta * l_reg
            loss_dict['loss_aux_obj'] = l_ot.item()

        elif self.loss_type == 'L3_SpatialInfoNCE':
            l_nce = self.infonce_module(outputs['fused_rna'], outputs['x_aux'], data.dist_edge_index)
            total_loss = self.beta * total_recon + self.gamma * l_spatial + 0.1 * l_nce + self.delta * l_reg
            loss_dict['loss_aux_obj'] = l_nce.item()

        elif self.loss_type == 'L4_SpatialPotts':
            total_loss = self.beta * total_recon + self.gamma * l_spatial + self.delta * l_reg
            if stage == 2 and outputs['q_list'] is not None:
                l_potts = self.potts_module(outputs['q_list'][0], data.dist_edge_index)
                total_loss = total_loss + 0.2 * l_potts
                loss_dict['loss_aux_obj'] = l_potts.item()

        elif self.loss_type == 'L5_UncertaintyBalancing':
            tasks = [total_recon, l_spatial, l_reg]
            if stage == 2 and outputs['q_list'] is not None:
                Q_joint = outputs['q_list'][0]
                target_P = compute_target_distribution(Q_joint.detach())
                l_kl = consensus_distribution_loss(outputs['q_list'], target_P)
                tasks.append(l_kl)
            else:
                tasks.append(torch.tensor(0.0, device=total_recon.device))
            total_loss, u_dict = self.uncertainty_module(tasks)
            loss_dict.update(u_dict)

        # Stage 2 DEC consensus KL loss
        if stage == 2 and outputs['q_list'] is not None and self.loss_type != 'L5_UncertaintyBalancing':
            Q_joint = outputs['q_list'][0]
            target_P = compute_target_distribution(Q_joint.detach())
            l_kl = consensus_distribution_loss(outputs['q_list'], target_P)
            total_loss = total_loss + self.lambda_kl * l_kl
            loss_dict['loss_kl'] = l_kl.item()

        loss_dict['loss_total'] = total_loss.item()
        return total_loss, loss_dict


# Model Factory
LOSS_MODEL_REGISTRY = {
    'L0_Standard': lambda in_r, in_a, nc, **kw: V6LossAblationModel('L0_Standard', in_r, in_a, nc, **kw),
    'L1_DenseRelational': lambda in_r, in_a, nc, **kw: V6LossAblationModel('L1_DenseRelational', in_r, in_a, nc, **kw),
    'L2_SinkhornOT': lambda in_r, in_a, nc, **kw: V6LossAblationModel('L2_SinkhornOT', in_r, in_a, nc, **kw),
    'L3_SpatialInfoNCE': lambda in_r, in_a, nc, **kw: V6LossAblationModel('L3_SpatialInfoNCE', in_r, in_a, nc, **kw),
    'L4_SpatialPotts': lambda in_r, in_a, nc, **kw: V6LossAblationModel('L4_SpatialPotts', in_r, in_a, nc, **kw),
    'L5_UncertaintyBalancing': lambda in_r, in_a, nc, **kw: V6LossAblationModel('L5_UncertaintyBalancing', in_r, in_a, nc, **kw),
}
