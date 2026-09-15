#!/usr/bin/env python3
"""
Unit test to verify instantiation and forward pass for all 25 V6 ablation variants across 4 tracks.
"""
import torch
import scipy.sparse as sp
import numpy as np
from torch_geometric.data import Data

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fusion-ablation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'loss-ablation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'encoder-ablation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'contrastive-ablation'))

from fusion_models import FUSION_MODEL_REGISTRY
from loss_models import LOSS_MODEL_REGISTRY
from encoder_models import ENCODER_MODEL_REGISTRY
from contrastive_models import CONTRASTIVE_MODEL_REGISTRY


def create_dummy_graph_data(num_nodes=50, rna_dim=100, aux_dim=30):
    x_rna = torch.randn(num_nodes, rna_dim)
    x_aux = torch.randn(num_nodes, aux_dim)

    # 1-hop ring graph for testing
    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    edge_weight = torch.ones(edge_index.shape[1])

    data = Data(
        x_RNA=x_rna,
        x_ADT=x_aux,
        sim_edge_index=edge_index,
        sim_edge_weight=edge_weight,
        dist_edge_index=edge_index,
        dist_edge_weight=edge_weight,
        common_edge_index=edge_index,
        common_edge_weight=edge_weight
    )

    sp_adj = sp.eye(num_nodes, dtype=np.float32).tocoo()
    indices = torch.from_numpy(np.vstack((sp_adj.row, sp_adj.col)).astype(np.int64))
    values = torch.from_numpy(sp_adj.data)
    data.spatial_adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
    return data


def run_tests():
    num_nodes = 50
    rna_dim = 100
    aux_dim = 30
    num_clusters = 5
    data = create_dummy_graph_data(num_nodes, rna_dim, aux_dim)

    all_registries = {
        'Fusion Track (F0 - F5)': FUSION_MODEL_REGISTRY,
        'Loss Track (L0 - L5)': LOSS_MODEL_REGISTRY,
        'Encoder Track (E0 - E5)': ENCODER_MODEL_REGISTRY,
        'Contrastive Track (C0 - C6)': CONTRASTIVE_MODEL_REGISTRY
    }

    total_models = sum(len(r) for r in all_registries.values())
    print("=" * 70)
    print(f"Testing all {total_models} V6 Ablation Variants (Forward Pass & Loss Computation)")
    print("=" * 70)

    for track_name, registry in all_registries.items():
        print(f"\n--- Testing {track_name} ---")
        for v_name, model_fn in registry.items():
            model = model_fn(in_r=rna_dim, in_a=aux_dim, nc=num_clusters, hidden_dim=64, out_dim=32)
            model.set_cluster_centers(np.random.randn(num_clusters, 32))

            # Stage 1 forward & loss
            out_stage1 = model(data, compute_q=False)
            loss_s1, d1 = model.compute_loss(data, out_stage1, stage=1)

            # Stage 2 forward & loss (with DEC Student-t Q)
            out_stage2 = model(data, compute_q=True)
            loss_s2, d2 = model.compute_loss(data, out_stage2, stage=2)

            assert out_stage1['embedding'].shape == (num_nodes, 32), f"Embedding shape mismatch: {out_stage1['embedding'].shape}"
            assert not torch.isnan(loss_s1), f"Stage 1 Loss is NaN for {v_name}"
            assert not torch.isnan(loss_s2), f"Stage 2 Loss is NaN for {v_name}"
            assert out_stage2['q_list'] is not None, f"q_list is None for {v_name}"

            print(f"  [PASS] {v_name:<28} | S1 Loss: {loss_s1.item():.4f} | S2 Loss: {loss_s2.item():.4f}")

    print("\n" + "=" * 70)
    print(f"🎉 ALL {total_models} V6 ABLATION VARIANTS PASSED VALIDATION PERFECTLY!")
    print("=" * 70)


if __name__ == '__main__':
    run_tests()
