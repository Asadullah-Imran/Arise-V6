import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict


# ==============================================================================
# C1: Cross-Modality Contrastive Learning (from GATCL Paper Eq. 18 - 21)
# ==============================================================================
class CrossModalContrastiveLoss(nn.Module):
    """
    Cross-Modality Contrastive Learning (from GATCL):
    - Positive pair: Transcriptomic & Aux representations at the SAME spot (H_i^RNA, H_i^Aux).
      s_i^+ = < \tilde{H}_i^RNA, \tilde{H}_i^Aux >
    - Negative pairs: Any omics datasets from DIFFERENT locations j \neq i:
      s_{i,j}^- = <\tilde{H}_i^R, \tilde{H}_j^A> + <\tilde{H}_i^A, \tilde{H}_j^R> + <\tilde{H}_i^A, \tilde{H}_j^A> + <\tilde{H}_i^R, \tilde{H}_j^R>
    - Loss:
      L_cl = - (1/N) \sum_{i=1}^N \log \frac{\exp(s_i^+ / \tau)}{\exp(s_i^+ / \tau) + \sum_{j \in N_{neg}(i)} \exp(s_{i,j}^- / \tau)}
    """
    def __init__(self, init_tau: float = 0.1, learnable_tau: bool = True):
        super(CrossModalContrastiveLoss, self).__init__()
        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(init_tau))
        else:
            self.register_buffer('tau', torch.tensor(init_tau))

    def forward(self, z_rna: torch.Tensor, z_aux: torch.Tensor) -> torch.Tensor:
        N = z_rna.shape[0]
        # Eq. 18: L2-normalize representations
        h_r = F.normalize(z_rna, p=2, dim=-1)
        h_a = F.normalize(z_aux, p=2, dim=-1)

        tau = torch.clamp(self.tau, min=0.01, max=1.0)

        # Eq. 19: Positive similarity (same spot, across modalities)
        pos_sim = torch.sum(h_r * h_a, dim=-1) / tau  # [N]

        # Compute full similarity matrices
        sim_ra = torch.mm(h_r, h_a.t()) / tau  # [N, N]
        sim_ar = torch.mm(h_a, h_r.t()) / tau  # [N, N]
        sim_aa = torch.mm(h_a, h_a.t()) / tau  # [N, N]
        sim_rr = torch.mm(h_r, h_r.t()) / tau  # [N, N]

        # Mask out self-similarities on diagonal for negative pairs
        diag_mask = torch.eye(N, dtype=torch.bool, device=z_rna.device)

        sim_ra_neg = sim_ra.masked_fill(diag_mask, float('-inf'))
        sim_ar_neg = sim_ar.masked_fill(diag_mask, float('-inf'))
        sim_aa_neg = sim_aa.masked_fill(diag_mask, float('-inf'))
        sim_rr_neg = sim_rr.masked_fill(diag_mask, float('-inf'))

        # Eq. 20 & 21: Log-sum-exp over all negative cross & intra pairs
        all_negs = torch.cat([sim_ra_neg, sim_ar_neg, sim_aa_neg, sim_rr_neg], dim=1) # [N, 4N]
        denom = torch.log(torch.exp(pos_sim) + torch.sum(torch.exp(all_negs.clamp(max=20.0)), dim=1) + 1e-10)

        loss = torch.mean(denom - pos_sim)
        return loss


# ==============================================================================
# C2: Local Neighborhood Deep Graph Infomax (DGI) Contrastive Learning (from Proust 2025)
# ==============================================================================
class ProustDGIContrastiveLoss(nn.Module):
    """
    Proust (2025) Contrastive Self-Supervised Learning (CSL) adapted from Deep Graph Infomax (Veličković et al. 2019):
    - Feature corruption: Shuffle spot features across nodes to generate corrupted graph (X', A).
    - Local neighborhood context readout: S_i = R(Z_i) = \sigma( (1/k) \sum_{j=1}^k Z_j + Z_i )
    - Bilinear discriminator: D(Z_i, S_i) = \sigma( Z_i^T W S_i )
    - Dual contrastive loss:
      L_CSL = - (1/2N) [ \sum \log D(Z_i, S_i) + \sum \log(1 - D(Z'_j, S_j)) ]
      L_CSL_corrupt = - (1/2N) [ \sum \log D(Z'_i, S'_i) + \sum \log(1 - D(Z_j, S'_j)) ]
    """
    def __init__(self, embed_dim: int = 64):
        super(ProustDGIContrastiveLoss, self).__init__()
        self.W = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.W)

    def readout(self, Z: torch.Tensor, dist_edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        # Neighborhood average pooling + self embedding
        src, dst = dist_edge_index[0], dist_edge_index[1]
        neigh_sum = torch.zeros_like(Z)
        deg = torch.zeros((num_nodes, 1), device=Z.device)
        
        neigh_sum.index_add_(0, src, Z[dst])
        deg.index_add_(0, src, torch.ones((src.shape[0], 1), device=Z.device))
        deg = torch.clamp(deg, min=1.0)

        S = torch.sigmoid(neigh_sum / deg + Z)
        return S

    def discriminate(self, Z: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        # D(Z, S) = \sigma( Z^T W S )
        ZW = torch.mm(Z, self.W)
        scores = torch.sum(ZW * S, dim=-1)
        return torch.sigmoid(scores)

    def forward(self, Z_real: torch.Tensor, Z_corrupt: torch.Tensor, dist_edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = Z_real.shape[0]

        S_real = self.readout(Z_real, dist_edge_index, num_nodes)
        S_corrupt = self.readout(Z_corrupt, dist_edge_index, num_nodes)

        # Real pairs vs Corrupted pairs
        prob_real_pos = self.discriminate(Z_real, S_real)
        prob_corrupt_neg_on_real_S = self.discriminate(Z_corrupt, S_real)

        prob_corrupt_pos = self.discriminate(Z_corrupt, S_corrupt)
        prob_real_neg_on_corrupt_S = self.discriminate(Z_real, S_corrupt)

        # Binary Cross-Entropy DGI Loss
        l_csl = -0.5 * torch.mean(torch.log(prob_real_pos + 1e-10) + torch.log(1.0 - prob_corrupt_neg_on_real_S + 1e-10))
        l_csl_corrupt = -0.5 * torch.mean(torch.log(prob_corrupt_pos + 1e-10) + torch.log(1.0 - prob_real_neg_on_corrupt_S + 1e-10))

        return l_csl + l_csl_corrupt


# ==============================================================================
# C3: Spatial Neighbor-Aware Contrastive Loss (from CoMo / Proust)
# ==============================================================================
class SpatialNeighborContrastiveLoss(nn.Module):
    """
    Neighbor-Aware Contrastive Loss:
    - Positive pairs: 1-hop physical spatial neighbors (i, j) \in E_dist.
    - Negative pairs: Non-neighboring spatial spots (i, k) \notin E_dist.
    - InfoNCE objective with cosine similarity.
    """
    def __init__(self, temperature: float = 0.1):
        super(SpatialNeighborContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, Z: torch.Tensor, dist_edge_index: torch.Tensor) -> torch.Tensor:
        Z_norm = F.normalize(Z, p=2, dim=-1)
        sim_mat = torch.mm(Z_norm, Z_norm.t()) / self.temperature

        src, dst = dist_edge_index[0], dist_edge_index[1]
        pos_sim = torch.sum(Z_norm[src] * Z_norm[dst], dim=-1) / self.temperature

        logsumexp = torch.logsumexp(sim_mat[src], dim=-1)
        loss = torch.mean(logsumexp - pos_sim)
        return loss


# ==============================================================================
# C4: Cluster-Aware / Prototype Contrastive Loss (from SpaMOAL / SpaConTDS)
# ==============================================================================
class ClusterAwareContrastiveLoss(nn.Module):
    """
    Cluster-Aware Contrastive Learning:
    - Positive pairs: Spots assigned to the same pseudo-cluster / domain (q_i ~ q_j).
    - Negative pairs: Spots assigned to different clusters.
    - Minimizes intra-cluster variance while maximizing inter-cluster margin using soft assignment Q.
    """
    def __init__(self, temperature: float = 0.1):
        super(ClusterAwareContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, Z: torch.Tensor, Q_soft: torch.Tensor) -> torch.Tensor:
        # Q_soft: [N, K] cluster assignments from DEC
        Z_norm = F.normalize(Z, p=2, dim=-1)
        
        # Prototype representation per cluster: C_k = sum_i q_ik * z_i / sum_i q_ik
        weights = Q_soft.t() # [K, N]
        cluster_sums = weights.sum(dim=1, keepdim=True) + 1e-10
        prototypes = torch.mm(weights, Z_norm) / cluster_sums # [K, D]
        proto_norm = F.normalize(prototypes, p=2, dim=-1) # [K, D]

        # Similarity between spot embeddings and all cluster prototypes
        spot_proto_sim = torch.mm(Z_norm, proto_norm.t()) / self.temperature # [N, K]

        # Cross-entropy alignment with soft target Q_soft
        log_prob = F.log_softmax(spot_proto_sim, dim=-1)
        loss = -torch.mean(torch.sum(Q_soft * log_prob, dim=-1))
        return loss


# ==============================================================================
# C5: Dual-View Graph Augmentation Consistency Contrastive Loss (from GRAS4T / STAIG)
# ==============================================================================
class GraphAugmentationConsistencyLoss(nn.Module):
    """
    Dual-View Graph Augmentation Contrastive Learning:
    - Generates two augmented views of the graph via edge dropping (drop_p=0.1) and feature masking (mask_p=0.1).
    - Same spot across views forms positive pair (z_i^(1), z_i^(2)).
    - All other spots act as negatives.
    """
    def __init__(self, temperature: float = 0.1):
        super(GraphAugmentationConsistencyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1_norm = F.normalize(z1, p=2, dim=-1)
        z2_norm = F.normalize(z2, p=2, dim=-1)

        N = z1.shape[0]
        sim_12 = torch.mm(z1_norm, z2_norm.t()) / self.temperature
        sim_11 = torch.mm(z1_norm, z1_norm.t()) / self.temperature

        diag_mask = torch.eye(N, dtype=torch.bool, device=z1.device)
        pos_sim = sim_12.diag()

        sim_11_neg = sim_11.masked_fill(diag_mask, float('-inf'))
        all_sim = torch.cat([sim_12, sim_11_neg], dim=1)

        loss = torch.mean(torch.logsumexp(all_sim, dim=1) - pos_sim)
        return loss


# ==============================================================================
# C6: Hybrid Multi-Level Contrastive Learning (Cross-Modal + Spatial + Cluster)
# ==============================================================================
class HybridMultiLevelContrastiveLoss(nn.Module):
    """
    Synergistic Multi-Level Contrastive Learning (Cross-Modal + Spatial Neighbor + Cluster Prototype):
    L_hybrid = lambda_1 * L_cross + lambda_2 * L_spatial_neighbor + lambda_3 * L_cluster
    """
    def __init__(self, lambda_cross: float = 1.0, lambda_spatial: float = 0.5, lambda_cluster: float = 0.2):
        super(HybridMultiLevelContrastiveLoss, self).__init__()
        self.cross_loss = CrossModalContrastiveLoss()
        self.spatial_loss = SpatialNeighborContrastiveLoss()
        self.cluster_loss = ClusterAwareContrastiveLoss()
        self.lambda_cross = lambda_cross
        self.lambda_spatial = lambda_spatial
        self.lambda_cluster = lambda_cluster

    def forward(self, z_rna: torch.Tensor, z_aux: torch.Tensor, fused_joint: torch.Tensor,
                dist_edge_index: torch.Tensor, Q_soft: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_cross = self.cross_loss(z_rna, z_aux)
        l_spatial = self.spatial_loss(fused_joint, dist_edge_index)
        
        l_cluster = torch.tensor(0.0, device=fused_joint.device)
        if Q_soft is not None:
            l_cluster = self.cluster_loss(fused_joint, Q_soft)

        total_cl = self.lambda_cross * l_cross + self.lambda_spatial * l_spatial + self.lambda_cluster * l_cluster

        loss_dict = {
            'cl_total': total_cl.item(),
            'cl_cross': l_cross.item(),
            'cl_spatial': l_spatial.item(),
            'cl_cluster': l_cluster.item()
        }
        return total_cl, loss_dict
