import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from torch_geometric.nn import GCNConv, GATv2Conv


# ==============================================================================
# E0: Standard 2-Layer GCN Encoder (Base V6 Backbone)
# ==============================================================================
class StandardGCNEncoder(nn.Module):
    """
    Standard 2-layer Graph Convolutional Network:
    X -> GCNConv(in_dim, hidden_dim) -> ReLU -> GCNConv(hidden_dim, out_dim)
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 64):
        super(StandardGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        out = self.conv2(h, edge_index, edge_weight)
        return out


# ==============================================================================
# E1: Global Multi-Head Transformer Encoder (from V1)
# ==============================================================================
class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_size: int, heads: int = 2):
        super(SelfAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        v_len, k_len, q_len = values.shape[1], keys.shape[1], query.shape[1]
        v = values.reshape(N, v_len, self.heads, self.head_dim)
        k = keys.reshape(N, k_len, self.heads, self.head_dim)
        q = query.reshape(N, q_len, self.heads, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        attention = torch.softmax(energy / math.sqrt(self.head_dim), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(N, q_len, self.embed_size)
        return self.fc_out(out)


class TransformerTokenEncoder(nn.Module):
    """
    Global Multi-Head Self-Attention Transformer Encoder block.
    """
    def __init__(self, in_dim: int, embed_dim: int = 64, heads: int = 2, forward_expansion: int = 4, dropout: float = 0.1):
        super(TransformerTokenEncoder, self).__init__()
        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.attention = SelfAttentionLayer(embed_dim, heads=heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x).unsqueeze(1) # [N, 1, embed_dim]
        attn = self.attention(h, h, h)
        h = self.dropout(self.norm1(attn + h))
        forward = self.feed_forward(h)
        out = self.dropout(self.norm2(forward + h)).squeeze(1)
        return out


# ==============================================================================
# E2: 3-Node Triangular Clique Motif Graph (from V2)
# ==============================================================================
def compute_3_node_motifs(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Extracts 3-node triangular cliques: M_3 = (A . A) [x] A
    """
    A = (adj_matrix > 0).astype(np.float32)
    np.fill_diagonal(A, 0)
    A2 = np.matmul(A, A)
    M3 = A2 * A
    return M3.astype(np.float32)


def blend_3_node_motifs(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                        motif_matrix: np.ndarray, k1: float = 0.5, k2: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    m_w = motif_matrix[src, dst]
    if m_w.max() > 0:
        m_w = m_w / m_w.max()
    m_tensor = torch.tensor(m_w, dtype=torch.float, device=edge_index.device)
    new_weight = k1 * edge_weight + k2 * m_tensor
    return edge_index, new_weight


# ==============================================================================
# E3: Higher-Order 4-Node Cycle Motif Graph (from V9)
# ==============================================================================
def compute_4_node_cycle_motifs(adj_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 3-node triangular cliques M_3 = (A . A) [x] A and 4-node square cycle motifs M_4 = (A . A . A) [x] A
    """
    A = (adj_matrix > 0).astype(np.float32)
    np.fill_diagonal(A, 0)
    A2 = np.matmul(A, A)
    M3 = A2 * A
    A3 = np.matmul(A2, A)
    M4 = A3 * A
    return M3.astype(np.float32), M4.astype(np.float32)


def blend_higher_order_motifs(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                              M3: np.ndarray, M4: np.ndarray,
                              k1: float = 0.4, k2: float = 0.3, k3: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    w3 = M3[src, dst]
    w4 = M4[src, dst]
    if w3.max() > 0:
        w3 = w3 / w3.max()
    if w4.max() > 0:
        w4 = w4 / w4.max()
    t3 = torch.tensor(w3, dtype=torch.float, device=edge_index.device)
    t4 = torch.tensor(w4, dtype=torch.float, device=edge_index.device)
    new_weight = k1 * edge_weight + k2 * t3 + k3 * t4
    return edge_index, new_weight


# ==============================================================================
# E4: Spectral Heat Diffusion Wavelet Graph Filtering (from V10)
# ==============================================================================
class HeatDiffusionWaveletGraph(nn.Module):
    """
    Multiscale continuous Heat Diffusion Wavelet filter on normalized Laplacian:
    psi_t = exp(-t * L_tilde) approximated via recursive Chebyshev polynomials.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 64, order: int = 3, time_scale: float = 1.0):
        super(HeatDiffusionWaveletGraph, self).__init__()
        self.order = order
        self.time_scale = time_scale
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)
        self.cheb_weights = nn.Parameter(torch.Tensor(order + 1, out_dim, out_dim))
        nn.init.xavier_uniform_(self.cheb_weights)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None,
                spatial_adj_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = F.relu(self.gcn1(x, edge_index, edge_weight))
        z = self.gcn2(h, edge_index, edge_weight)

        if spatial_adj_norm is None:
            return z

        # Recursive Chebyshev polynomial diffusion approximation
        N = z.shape[0]
        T0 = z
        if spatial_adj_norm.is_sparse:
            T1 = torch.sparse.mm(spatial_adj_norm, z)
        else:
            T1 = torch.mm(spatial_adj_norm, z)

        cheb_out = torch.mm(T0, self.cheb_weights[0]) + math.exp(-self.time_scale) * torch.mm(T1, self.cheb_weights[1])
        T_prev2 = T0
        T_prev1 = T1

        for k in range(2, self.order + 1):
            if spatial_adj_norm.is_sparse:
                Tk = 2.0 * torch.sparse.mm(spatial_adj_norm, T_prev1) - T_prev2
            else:
                Tk = 2.0 * torch.mm(spatial_adj_norm, T_prev1) - T_prev2
            coeff = (self.time_scale ** k) / math.factorial(k) * math.exp(-self.time_scale)
            cheb_out = cheb_out + coeff * torch.mm(Tk, self.cheb_weights[k])
            T_prev2 = T_prev1
            T_prev1 = Tk

        return F.normalize(cheb_out, p=2, dim=-1)


# ==============================================================================
# E5: Graph Attention Network Backbone (GATv2 with Residual Connections)
# ==============================================================================
class GATv2ResidualEncoder(nn.Module):
    """
    Multi-Head Graph Attention Network (GATv2) with Residual LayerNorm projections (from SpatialGlue-GATCL).
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 64, heads: int = 2, dropout: float = 0.0):
        super(GATv2ResidualEncoder, self).__init__()
        self.gat1 = GATv2Conv(in_channels=in_dim, out_channels=hidden_dim, heads=heads, concat=True, dropout=dropout)
        dim1 = hidden_dim * heads
        self.norm1 = nn.LayerNorm(dim1)
        self.res1 = nn.Linear(in_dim, dim1) if in_dim != dim1 else nn.Identity()

        self.gat2 = GATv2Conv(in_channels=dim1, out_channels=out_dim, heads=1, concat=False, dropout=dropout)
        self.norm2 = nn.LayerNorm(out_dim)
        self.res2 = nn.Linear(dim1, out_dim) if dim1 != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.gat1(x, edge_index))
        h1 = self.norm1(h1) + self.res1(x)

        h2 = F.relu(self.gat2(h1, edge_index))
        out = self.norm2(h2) + self.res2(h1)
        return out
