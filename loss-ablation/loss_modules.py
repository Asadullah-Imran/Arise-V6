import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


# ==============================================================================
# Base DEC Clustering & Distribution Helpers (Identical to V6 Engine)
# ==============================================================================
class QDistribution(nn.Module):
    """
    Student-t distribution kernel for Deep Embedding Clustering (DEC).
    q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(- (alpha + 1)/2) / sum_k (...)
    """
    def __init__(self, num_clusters: int, embed_dim: int, alpha: float = 1.0):
        super(QDistribution, self).__init__()
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, embed_dim))
        nn.init.xavier_normal_(self.cluster_centers.data)

    def forward(self, embeddings_list: List[torch.Tensor]) -> List[torch.Tensor]:
        q_list = []
        for z in embeddings_list:
            dist_sq = torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_centers, 2), dim=2)
            q = 1.0 / (1.0 + (dist_sq / self.alpha))
            q = torch.pow(q, (self.alpha + 1.0) / 2.0)
            q = (q.t() / (torch.sum(q, dim=1) + 1e-10)).t()
            q_list.append(q)
        return q_list


def compute_target_distribution(Q: torch.Tensor) -> torch.Tensor:
    """
    Computes sharpened self-training target distribution P from soft assignment Q.
    p_ij = (q_ij^2 / f_j) / sum_k (q_ik^2 / f_k), where f_j = sum_i q_ij.
    """
    weight = Q ** 2 / (Q.sum(0) + 1e-10)
    P = (weight.t() / (weight.sum(1) + 1e-10)).t()
    return P


def consensus_distribution_loss(q_list: List[torch.Tensor], target_P: torch.Tensor) -> torch.Tensor:
    """
    Consensus multi-representation KL divergence loss:
    L_KL = KL( (log Q_joint + log Q_RNA + log Q_aux)/3 || P )
    """
    log_q_mean = sum([q.clamp(min=1e-10).log() for q in q_list]) / len(q_list)
    return F.kl_div(log_q_mean, target_P, reduction='batchmean')


def spatial_regularization_loss(emb: torch.Tensor, dist_edge_index: torch.Tensor,
                                dist_edge_weight: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Spatial contrastive regularization pulling 1-hop physical neighbors together while pushing non-neighbors apart.
    """
    graph_nei = torch.sparse_coo_tensor(
        dist_edge_index, torch.ones_like(dist_edge_weight), size=(num_nodes, num_nodes)
    ).to_dense()
    graph_neg = 1.0 - graph_nei

    emb_norm = F.normalize(emb, p=2, dim=1, eps=1e-8)
    sim_mat = torch.matmul(emb_norm, emb_norm.T)
    sim_mat = sim_mat - torch.diag_embed(torch.diag(sim_mat))
    sim_mat = torch.sigmoid(sim_mat)

    neigh_loss = torch.mul(graph_nei, torch.log(sim_mat + 1e-10)).mean()
    neg_loss = torch.mul(graph_neg, torch.log(1.0 - sim_mat + 1e-10)).mean()
    return -(neigh_loss + neg_loss) / 2.0


def parameter_regularization_loss(model: nn.Module, l1_lambda: float = 1e-4, l2_lambda: float = 1e-3) -> torch.Tensor:
    l1_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    l2_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        if p.requires_grad:
            l1_loss = l1_loss + torch.sum(torch.abs(p))
            l2_loss = l2_loss + torch.sum(p ** 2)
    return l1_lambda * l1_loss + l2_lambda * l2_loss


# ==============================================================================
# L1: Dense Cross-Modal Relational Alignment Loss (from V7)
# ==============================================================================
class DenseRelationalAlignmentLoss(nn.Module):
    """
    Computes spot embedding distance + cross-modal Gram matrix relational Frobenius norm:
    L_dense = (1/N) * sum(||Z_i - z_RNA,i||^2 + ||Z_i - z_aux,i||^2) + ||Z_RNA*Z_RNA^T - Z_aux*Z_aux^T||_F^2
    """
    def __init__(self, rel_weight: float = 0.1):
        super(DenseRelationalAlignmentLoss, self).__init__()
        self.rel_weight = rel_weight

    def forward(self, fused_joint: torch.Tensor, fused_rna: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        num_nodes = fused_joint.shape[0]
        l_emb = torch.mean((fused_joint - fused_rna) ** 2) + torch.mean((fused_joint - x_aux) ** 2)
        
        # Relational Gram alignment
        gram_r = torch.mm(fused_rna, fused_rna.t())
        gram_a = torch.mm(x_aux, x_aux.t())
        l_rel = torch.norm(gram_r - gram_a, p='fro') / (num_nodes * num_nodes)
        
        return l_emb + self.rel_weight * l_rel


# ==============================================================================
# L2: Sinkhorn Optimal Transport Alignment Loss (from V12)
# ==============================================================================
class SinkhornOptimalTransportLoss(nn.Module):
    """
    Computes entropic regularized Wasserstein-1 optimal transport distance between RNA and Aux modality representations:
    min_T <T, C> - epsilon * H(T), solved via matrix Sinkhorn-Knopp iterations in log-space.
    """
    def __init__(self, eps: float = 0.1, max_iter: int = 40):
        super(SinkhornOptimalTransportLoss, self).__init__()
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, z_rna: torch.Tensor, z_aux: torch.Tensor) -> torch.Tensor:
        # Cosine distance cost matrix C in [0, 2]
        z_r_norm = F.normalize(z_rna, p=2, dim=-1)
        z_a_norm = F.normalize(z_aux, p=2, dim=-1)
        C = 1.0 - torch.mm(z_r_norm, z_a_norm.t())

        N = z_rna.shape[0]
        mu = torch.full((N,), 1.0 / N, device=z_rna.device, dtype=torch.float)
        nu = torch.full((N,), 1.0 / N, device=z_rna.device, dtype=torch.float)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        # Log-space stabilized Sinkhorn iterations
        K = -C / self.eps
        for _ in range(self.max_iter):
            u = self.eps * (torch.log(mu + 1e-12) - torch.logsumexp(K + v.unsqueeze(0) / self.eps, dim=1))
            v = self.eps * (torch.log(nu + 1e-12) - torch.logsumexp(K + u.unsqueeze(1) / self.eps, dim=0))

        T = torch.exp((u.unsqueeze(1) + v.unsqueeze(0) - C) / self.eps)
        ot_loss = torch.sum(T * C)
        return ot_loss


# ==============================================================================
# L3: Spatial Multi-Modal InfoNCE Contrastive Loss (from V13)
# ==============================================================================
class SpatialMultimodalInfoNCELoss(nn.Module):
    """
    Contrastive InfoNCE loss treating 1-hop physical neighbors in RNA and Aux as positive pairs,
    and non-neighbor distant spots as hard negative pairs with temperature scaling.
    """
    def __init__(self, temperature: float = 0.1):
        super(SpatialMultimodalInfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_rna: torch.Tensor, z_aux: torch.Tensor, dist_edge_index: torch.Tensor) -> torch.Tensor:
        z_r_norm = F.normalize(z_rna, p=2, dim=-1)
        z_a_norm = F.normalize(z_aux, p=2, dim=-1)

        sim_matrix = torch.mm(z_r_norm, z_a_norm.t()) / self.temperature
        
        src, dst = dist_edge_index[0], dist_edge_index[1]
        pos_sim = torch.sum(z_r_norm[src] * z_a_norm[dst], dim=-1) / self.temperature

        # InfoNCE denominator: log-sum-exp over row
        logsumexp = torch.logsumexp(sim_matrix[src], dim=-1)
        infonce_loss = torch.mean(logsumexp - pos_sim)
        return infonce_loss


# ==============================================================================
# L4: Spatial Potts Markov Random Field Consensus Regularizer (from V14)
# ==============================================================================
class SpatialPottsRegularizer(nn.Module):
    """
    Differentiable Dirichlet Energy / Potts Model spatial smoothness penalty on soft cluster assignments Q:
    L_Potts = (1/2|E|) * sum_{(i,j) in E} ||q_i - q_j||_2^2
    Eliminates noisy isolated 'salt-and-pepper' misclassifications in heterogeneous tissues.
    """
    def __init__(self):
        super(SpatialPottsRegularizer, self).__init__()

    def forward(self, q_tensor: torch.Tensor, dist_edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = dist_edge_index[0], dist_edge_index[1]
        q_diff = q_tensor[src] - q_tensor[dst]
        potts_loss = torch.mean(torch.sum(q_diff ** 2, dim=-1))
        return potts_loss


# ==============================================================================
# L5: Homoscedastic Multi-Task Loss Balancing Engine (Kendall & Gal)
# ==============================================================================
class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """
    Learns task log-variance parameters s_m = log(sigma_m^2) to balance multi-task gradients automatically:
    L_total = sum_m exp(-s_m) * L_m + sum_m s_m
    """
    def __init__(self, num_tasks: int = 4):
        super(UncertaintyWeightedMultiTaskLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0.0
        weighted_dict = {}
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            task_loss = precision * loss + self.log_vars[i]
            total_loss = total_loss + task_loss
            weighted_dict[f'task_{i}_weight'] = precision.item()
            weighted_dict[f'task_{i}_loss'] = loss.item()
        return total_loss, weighted_dict
