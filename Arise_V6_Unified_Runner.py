#!/usr/bin/env python3
"""
================================================================================
  🧬 ARISE-V6 Component Ablation Master Unified Runner
================================================================================

  Evaluates 26 Architectural Variants across 4 Systematic Ablation Tracks:
  ------------------------------------------------------------------------------
  1. Fusion Ablation:      F0_Linear, F1_LocalGlobal, F2_VarianceWeight, 
                           F3_SpatialCrossAttention, F4_BilinearTensor, 
                           F5_GatedMultiModal, F6_QKVCrossFusion
  2. Loss Ablation:        L0_Standard, L1_DenseRelational, L2_SinkhornOT, 
                           L3_SpatialInfoNCE, L4_SpatialPotts, L5_UncertaintyBalancing
  3. Encoder Ablation:     E0_StandardGCN, E1_GlobalTransformer, E2_TriangularMotif, 
                           E3_HigherOrderMotif, E4_HeatWavelet, E5_GATAttention
  4. Contrastive Ablation: C0_BaseV6, C1_CrossModalCL, C2_ProustDGI, 
                           C3_SpatialNeighborCL, C4_ClusterAwareCL, 
                           C5_GraphAugConsistency, C6_HybridMultiLevelCL
  ------------------------------------------------------------------------------
  Base Anchor: ARISE-V6 (ARISE + 2-Stage Consensus DEC)
  Benchmark Datasets: 6 Multi-Omics Datasets (10x Visium & Spatial-Epigenome)
================================================================================
"""

import os
import sys
import time
import math
import argparse
import random
import warnings
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    v_measure_score,
    fowlkes_mallows_score,
    silhouette_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATv2Conv

warnings.filterwarnings('ignore')

# Add sub-folders to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'fusion-ablation'))
sys.path.insert(0, os.path.join(current_dir, 'loss-ablation'))
sys.path.insert(0, os.path.join(current_dir, 'encoder-ablation'))
sys.path.insert(0, os.path.join(current_dir, 'contrastive-ablation'))

from fusion_models import FUSION_MODEL_REGISTRY
from loss_models import LOSS_MODEL_REGISTRY
from encoder_models import ENCODER_MODEL_REGISTRY, compute_3_node_motifs, blend_3_node_motifs, compute_4_node_cycle_motifs, blend_higher_order_motifs
from contrastive_models import CONTRASTIVE_MODEL_REGISTRY

# Consolidated Registry
ALL_ABLATION_MODELS = {
    # Fusion Track
    **FUSION_MODEL_REGISTRY,
    # Loss Track
    **LOSS_MODEL_REGISTRY,
    # Encoder Track
    **ENCODER_MODEL_REGISTRY,
    # Contrastive Learning Track
    **CONTRASTIVE_MODEL_REGISTRY
}


# ----------------------------------------------------------------------
# 1. REPRODUCIBILITY & SEEDING
# ----------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ----------------------------------------------------------------------
# 2. BENCHMARK DATASET REGISTRY & DOWNLOADER
# ----------------------------------------------------------------------
BENCHMARK_DATASETS = [
    ("10x_human_lymph_node_A1", "https://drive.google.com/drive/folders/10z1N4MwW8Y49o8GlkYGBKVx1N7fiMuyC"),
    ("10x_human_lymph_node_D1", "https://drive.google.com/drive/folders/1-g_Ca2XMaMXF-MisuVY-wobWDX86O6zz"),
    ("Mouse_Brain_E11_S1", "https://drive.google.com/drive/folders/1zRwDJrYnks0LRzlAVRqPU7jE_OcStgPo"),
    ("Mouse_Brain_E13_S1", "https://drive.google.com/drive/folders/1GOufwIRjjfcd9Bi2GKtebzKoPCg2jVud"),
    ("Mouse_Brain_E15_S1", "https://drive.google.com/drive/folders/1rHkTL5OF5qPsEERypRGMS51SjUQ69tdD"),
    ("Mouse_Brain_E18_S1", "https://drive.google.com/drive/folders/1Xj1LNIAY93biS6JIMKNRODn5GvtCKADB"),
]

DATASET_METADATA = {
    '10x_human_lymph_node_A1': {'name': 'Human Lymph Node A1', 'omics': 'RNA + ADT', 'tissue': 'Human Lymph Node', 'platform': '10x Visium', 'spots': 3484, 'clusters': 9},
    '10x_human_lymph_node_D1': {'name': 'Human Lymph Node D1', 'omics': 'RNA + ADT', 'tissue': 'Human Lymph Node', 'platform': '10x Visium', 'spots': 3487, 'clusters': 10},
    'Mouse_Brain_E11_S1': {'name': 'Mouse Brain E11 S1', 'omics': 'RNA + ATAC', 'tissue': 'Embryonic Mouse Brain', 'platform': 'Spatial-Epigenome', 'spots': 2473, 'clusters': 11},
    'Mouse_Brain_E13_S1': {'name': 'Mouse Brain E13 S1', 'omics': 'RNA + ATAC', 'tissue': 'Embryonic Mouse Brain', 'platform': 'Spatial-Epigenome', 'spots': 2676, 'clusters': 12},
    'Mouse_Brain_E15_S1': {'name': 'Mouse Brain E15 S1', 'omics': 'RNA + ATAC', 'tissue': 'Embryonic Mouse Brain', 'platform': 'Spatial-Epigenome', 'spots': 3241, 'clusters': 13},
    'Mouse_Brain_E18_S1': {'name': 'Mouse Brain E18 S1', 'omics': 'RNA + ATAC', 'tissue': 'Embryonic Mouse Brain', 'platform': 'Spatial-Epigenome', 'spots': 3192, 'clusters': 14},
}


def download_dataset_if_needed(dataset_name: str, folder_url: str, base_dir: str = "data") -> str:
    local_path = os.path.join(base_dir, dataset_name)
    os.makedirs(local_path, exist_ok=True)
    rna_path = os.path.join(local_path, "adata_RNA.h5ad")
    if not os.path.exists(rna_path):
        print(f"Downloading dataset files for {dataset_name}...")
        try:
            import gdown
            gdown.download_folder(folder_url, output=local_path, quiet=False, use_cookies=False)
        except Exception as e:
            print(f"Warning: gdown download encountered: {e}. Checking local data fallback...")
    return local_path


# ----------------------------------------------------------------------
# 3. PREPROCESSING & GRAPH BUILDER
# ----------------------------------------------------------------------
def clr_normalize(adata: sc.AnnData) -> sc.AnnData:
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)
    adata = adata.copy()
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def tfidf(X):
    idf = X.shape[0] / (X.sum(axis=0) + 1e-10)
    if sp.issparse(X):
        tf = X.multiply(1.0 / (X.sum(axis=1) + 1e-10))
        return tf.multiply(idf)
    else:
        tf = X / (X.sum(axis=1, keepdims=True) + 1e-10)
        return tf * idf


def lsi_preprocess(adata: sc.AnnData, n_components: int = 50) -> np.ndarray:
    X_tf = tfidf(adata.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X_tf)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components + 1, random_state=42)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= (X_lsi.std(axis=1, ddof=1, keepdims=True) + 1e-10)
    return X_lsi[:, 1:]


def build_multimodal_graphs(adata_rna: sc.AnnData, adata_aux: sc.AnnData, is_human: bool, device: str = 'cpu') -> Data:
    sc.pp.filter_genes(adata_rna, min_cells=10)
    if not is_human:
        sc.pp.filter_cells(adata_rna, min_genes=200)

    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)
    adata_rna_high = adata_rna[:, adata_rna.var['highly_variable']]

    if is_human:
        # RNA + ADT
        pca_engine = PCA(n_components=adata_aux.n_vars - 1, random_state=42)
        X_pca_rna = pca_engine.fit_transform(adata_rna_high.X if not sp.issparse(adata_rna_high.X) else adata_rna_high.X.toarray())
        
        adata_aux = clr_normalize(adata_aux)
        sc.pp.scale(adata_aux)
        pca_aux = PCA(n_components=adata_aux.n_vars - 1, random_state=42)
        X_pca_aux = pca_aux.fit_transform(adata_aux.X if not sp.issparse(adata_aux.X) else adata_aux.X.toarray())
    else:
        # RNA + ATAC
        pca_engine = PCA(n_components=50, random_state=42)
        X_pca_rna = pca_engine.fit_transform(adata_rna_high.X if not sp.issparse(adata_rna_high.X) else adata_rna_high.X.toarray())
        
        adata_aux = adata_aux[adata_rna.obs_names].copy()
        X_pca_aux = lsi_preprocess(adata_aux, n_components=50)

    num_nodes = X_pca_rna.shape[0]

    # 1. Similarity Graph (G_sim from transcriptomic cosine distance)
    k_sim = 20
    sim_matrix = cosine_similarity(X_pca_rna)
    sim_graph_coo = kneighbors_graph(X_pca_rna, n_neighbors=k_sim, mode='connectivity', metric='cosine', include_self=False).tocoo()
    sim_edge_index = torch.tensor(np.vstack((sim_graph_coo.row, sim_graph_coo.col)), dtype=torch.long).to(device)
    sim_edge_weight = torch.tensor(sim_matrix[sim_graph_coo.row, sim_graph_coo.col], dtype=torch.float).to(device)

    # 2. Distance Graph (G_dist from spatial coordinates)
    spatial_coords = adata_rna.obsm['spatial']
    k_dist = 6
    nbrs = NearestNeighbors(n_neighbors=k_dist + 1).fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)
    dist_src = np.repeat(np.arange(num_nodes), k_dist)
    dist_dst = indices[:, 1:].flatten()
    dist_edge_index = torch.tensor(np.vstack((dist_src, dist_dst)), dtype=torch.long).to(device)
    dist_edge_weight = torch.ones(dist_edge_index.shape[1], dtype=torch.float).to(device)

    # 3. Common Scaffold (G_common = G_sim ∩ G_dist)
    sim_set = set(zip(sim_graph_coo.row, sim_graph_coo.col))
    dist_set = set(zip(dist_src, dist_dst))
    common_set = sim_set.intersection(dist_set)
    if len(common_set) == 0:
        common_edge_index = dist_edge_index
        common_edge_weight = dist_edge_weight
    else:
        c_src, c_dst = zip(*common_set)
        common_edge_index = torch.tensor(np.vstack((c_src, c_dst)), dtype=torch.long).to(device)
        common_edge_weight = torch.ones(common_edge_index.shape[1], dtype=torch.float).to(device)

    # Normalized spatial adjacency for wavelets / attention
    sp_adj = sp.coo_matrix((np.ones(len(dist_src)), (dist_src, dist_dst)), shape=(num_nodes, num_nodes))
    sp_adj = sp_adj + sp.eye(num_nodes)
    d_inv_sqrt = np.power(np.array(sp_adj.sum(1)), -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = sp.diags(d_inv_sqrt)
    sp_norm = sp_adj.dot(d_mat).transpose().dot(d_mat).tocoo().astype(np.float32)
    
    indices = torch.from_numpy(np.vstack((sp_norm.row, sp_norm.col)).astype(np.int64))
    values = torch.from_numpy(sp_norm.data)
    spatial_adj_norm = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).to(device)

    data = Data(
        x_RNA=torch.tensor(X_pca_rna, dtype=torch.float).to(device),
        x_ADT=torch.tensor(X_pca_aux, dtype=torch.float).to(device),
        sim_edge_index=sim_edge_index,
        sim_edge_weight=sim_edge_weight,
        dist_edge_index=dist_edge_index,
        dist_edge_weight=dist_edge_weight,
        common_edge_index=common_edge_index,
        common_edge_weight=common_edge_weight
    )
    data.spatial_adj = spatial_adj_norm
    return data


# ----------------------------------------------------------------------
# 4. EVALUATION METRICS
# ----------------------------------------------------------------------
def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, embeddings: np.ndarray) -> Dict[str, float]:
    y_true_str = np.asarray(y_true).astype(str)
    y_pred_str = np.asarray(y_pred).astype(str)
    return {
        'ARI': float(adjusted_rand_score(y_true_str, y_pred_str)),
        'NMI': float(normalized_mutual_info_score(y_true_str, y_pred_str)),
        'AMI': float(adjusted_mutual_info_score(y_true_str, y_pred_str)),
        'Homogeneity': float(homogeneity_score(y_true_str, y_pred_str)),
        'V-measure': float(v_measure_score(y_true_str, y_pred_str)),
        'FMI': float(fowlkes_mallows_score(y_true_str, y_pred_str)),
        'Silhouette': float(silhouette_score(embeddings, y_pred_str))
    }


def run_kmeans_clustering(embeddings: np.ndarray, num_clusters: int, seed: int = 42) -> np.ndarray:
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed)
    return kmeans.fit_predict(embeddings)


# ----------------------------------------------------------------------
# 5. UNIFIED TWO-STAGE DEC ABLATION RUNNER
# ----------------------------------------------------------------------
def train_and_evaluate_variant(variant: str, model, graph_data, ground_truth, num_clusters: int, args, seed: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_sil = -1.0
    best_embeddings = None
    best_labels = None

    pretrain_epochs = args.pretrain_epochs if args.pretrain_epochs is not None else int(args.epochs * 0.625)
    finetune_epochs = args.finetune_epochs if args.finetune_epochs is not None else (args.epochs - pretrain_epochs)

    # STAGE 1: Pre-training Representation Learning
    model.train()
    for epoch in range(pretrain_epochs):
        optimizer.zero_grad()
        outputs = model(graph_data, compute_q=False)
        loss, loss_dict = model.compute_loss(graph_data, outputs, stage=1)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_out = model(graph_data, compute_q=False)
            emb = eval_out['embedding'].detach().cpu().numpy()
        model.train()

        pred_labels = run_kmeans_clustering(emb, num_clusters, seed=seed)
        metrics = compute_all_metrics(ground_truth, pred_labels, emb)
        sil = metrics['Silhouette']

        if sil > best_sil:
            best_sil = sil
            best_embeddings = emb.copy()
            best_labels = pred_labels.copy()

        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == pretrain_epochs - 1:
            print(f"[{variant} | Seed {seed}] Pre-train Epoch {epoch+1:3d}/{pretrain_epochs} | Loss: {loss.item():.4f} | ARI: {metrics['ARI']:.4f} | NMI: {metrics['NMI']:.4f} | Sil: {sil:.4f}")

    # STAGE 2: Consensus DEC Fine-Tuning
    if finetune_epochs > 0:
        print(f"[{variant} | Seed {seed}] Initializing Cluster Centers from best pre-train representation...")
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed)
        kmeans.fit(best_embeddings)
        model.set_cluster_centers(kmeans.cluster_centers_)

        model.train()
        for epoch in range(finetune_epochs):
            optimizer.zero_grad()
            outputs = model(graph_data, compute_q=True)
            loss, loss_dict = model.compute_loss(graph_data, outputs, stage=2)
            loss.backward()
            optimizer.step()

            q_joint = outputs['q_list'][0].detach()
            pred_labels = torch.argmax(q_joint, dim=1).cpu().numpy()
            emb = outputs['embedding'].detach().cpu().numpy()

            metrics = compute_all_metrics(ground_truth, pred_labels, emb)
            sil = metrics['Silhouette']

            if sil > best_sil:
                best_sil = sil
                best_embeddings = emb.copy()
                best_labels = pred_labels.copy()

            if (epoch + 1) % 50 == 0 or epoch == finetune_epochs - 1:
                print(f"[{variant} | Seed {seed}] DEC Fine-tune Epoch {epoch+1:3d}/{finetune_epochs} | Loss: {loss.item():.4f} | KL: {loss_dict.get('loss_kl', 0.0):.4f} | ARI: {metrics['ARI']:.4f} | NMI: {metrics['NMI']:.4f} | Sil: {sil:.4f}")

    final_metrics = compute_all_metrics(ground_truth, best_labels, best_embeddings)
    return final_metrics, best_embeddings, best_labels


# ----------------------------------------------------------------------
# 6. CLI ENTRYPOINT
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ARISE-V6 Systematic Component Ablation Runner")
    parser.add_argument('--track', type=str, default='all', choices=['all', 'fusion', 'loss', 'encoder', 'contrastive'],
                        help="Ablation track: 'fusion', 'loss', 'encoder', 'contrastive', or 'all'")
    parser.add_argument('--variant', type=str, default='all',
                        help="Specific variant code, e.g. 'F0_Linear,F1_LocalGlobal', 'L1_DenseRelational', 'E2_TriangularMotif', 'C2_ProustDGI', or 'all'")
    parser.add_argument('--dataset', type=str, default='all', help="Dataset index (0-5), comma-separated list '0,1', or 'all'")
    parser.add_argument('--data_dir', type=str, default='data', help="Local dataset folder")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 1234, 2024])
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--pretrain_epochs', type=int, default=250)
    parser.add_argument('--finetune_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out_dir', type=str, default='results_v6_ablation')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Determine variants to execute
    if args.variant.lower() != 'all':
        raw_vars = args.variant.replace(' ', ',').split(',')
        variants_to_run = [v.strip() for v in raw_vars if v.strip()]
    elif args.track == 'fusion':
        variants_to_run = list(FUSION_MODEL_REGISTRY.keys())
    elif args.track == 'loss':
        variants_to_run = list(LOSS_MODEL_REGISTRY.keys())
    elif args.track == 'encoder':
        variants_to_run = list(ENCODER_MODEL_REGISTRY.keys())
    elif args.track == 'contrastive':
        variants_to_run = list(CONTRASTIVE_MODEL_REGISTRY.keys())
    else:
        variants_to_run = list(ALL_ABLATION_MODELS.keys())

    dataset_indices = list(range(len(BENCHMARK_DATASETS))) if args.dataset.lower() == 'all' else [int(i.strip()) for i in args.dataset.split(',') if i.strip()]

    print("=" * 80)
    print(f"Starting ARISE-V6 Ablation Suite | Track={args.track} | Variants={variants_to_run} | Seeds={args.seeds} | Epochs={args.epochs}")
    print("=" * 80)

    all_results = []

    for ds_idx in dataset_indices:
        ds_name, ds_url = BENCHMARK_DATASETS[ds_idx]
        print("\n" + "=" * 80)
        print(f"================== DATASET: {ds_name} (Index {ds_idx}) ==================")
        print("=" * 80)

        local_dir = download_dataset_if_needed(ds_name, ds_url, args.data_dir)
        is_human = ds_name.startswith("10x")

        rna_file = os.path.join(local_dir, "adata_RNA.h5ad")
        aux_file = os.path.join(local_dir, "adata_ADT.h5ad" if is_human else "adata_ATAC.h5ad")
        anno_file = os.path.join(local_dir, "annotation.csv" if is_human else "anno.csv")

        if not os.path.exists(rna_file) or not os.path.exists(aux_file):
            print(f"Error: Missing data files in {local_dir}. Skipping.")
            continue

        adata_rna = sc.read_h5ad(rna_file)
        adata_aux = sc.read_h5ad(aux_file)
        adata_rna.var_names_make_unique()
        adata_aux.var_names_make_unique()

        anno_df = pd.read_csv(anno_file, index_col=0)
        gt_col = "manual-anno" if is_human else "cluster"
        ground_truth = anno_df[gt_col].values
        num_clusters = len(np.unique(ground_truth))

        print(f"Loaded {ds_name}: {adata_rna.n_obs} spots, {num_clusters} ground-truth clusters.")

        # Build base graphs
        graph_data = build_multimodal_graphs(adata_rna, adata_aux, is_human=is_human, device=args.device)

        # Precompute Motif Matrices if motif variants are requested
        motif_3_computed = False
        motif_4_computed = False
        M3, M4 = None, None

        for variant in variants_to_run:
            if variant not in ALL_ABLATION_MODELS:
                print(f"Warning: Variant {variant} not in registry. Skipping.")
                continue

            print("\n" + "-" * 60)
            print(f"---------- VARIANT: {variant} on {ds_name} ----------")
            print("-" * 60)

            # Apply motif graph augmentation if required
            var_graph_data = graph_data.clone()
            if variant == 'E2_TriangularMotif':
                if not motif_3_computed:
                    print("Computing 3-node triangular clique motifs...")
                    sim_coo = sp.coo_matrix((var_graph_data.sim_edge_weight.cpu().numpy(),
                                            (var_graph_data.sim_edge_index[0].cpu().numpy(), var_graph_data.sim_edge_index[1].cpu().numpy())),
                                            shape=(adata_rna.n_obs, adata_rna.n_obs))
                    M3 = compute_3_node_motifs(sim_coo.toarray())
                    motif_3_computed = True
                new_idx, new_wt = blend_3_node_motifs(var_graph_data.sim_edge_index, var_graph_data.sim_edge_weight, M3)
                var_graph_data.sim_edge_index = new_idx
                var_graph_data.sim_edge_weight = new_wt

            elif variant == 'E3_HigherOrderMotif':
                if not motif_4_computed:
                    print("Computing 4-node higher-order cycle motifs...")
                    sim_coo = sp.coo_matrix((var_graph_data.sim_edge_weight.cpu().numpy(),
                                            (var_graph_data.sim_edge_index[0].cpu().numpy(), var_graph_data.sim_edge_index[1].cpu().numpy())),
                                            shape=(adata_rna.n_obs, adata_rna.n_obs))
                    M3, M4 = compute_4_node_cycle_motifs(sim_coo.toarray())
                    motif_4_computed = True
                new_idx, new_wt = blend_higher_order_motifs(var_graph_data.sim_edge_index, var_graph_data.sim_edge_weight, M3, M4)
                var_graph_data.sim_edge_index = new_idx
                var_graph_data.sim_edge_weight = new_wt

            for seed in args.seeds:
                set_seed(seed)
                model_fn = ALL_ABLATION_MODELS[variant]
                model = model_fn(
                    in_r=var_graph_data.x_RNA.shape[1],
                    in_a=var_graph_data.x_ADT.shape[1],
                    nc=num_clusters,
                    hidden_dim=args.hidden_dim,
                    out_dim=args.out_dim
                ).to(args.device)

                metrics, final_emb, final_labels = train_and_evaluate_variant(
                    variant, model, var_graph_data, ground_truth, num_clusters, args, seed
                )

                record = {
                    'dataset': ds_name,
                    'variant': variant,
                    'track': 'Fusion' if variant.startswith('F') else ('Loss' if variant.startswith('L') else ('Encoder' if variant.startswith('E') else 'Contrastive')),
                    'seed': seed,
                    **metrics
                }
                all_results.append(record)
                print(f">> [RESULT] {ds_name} | {variant} | Seed {seed} => ARI: {metrics['ARI']:.4f} | NMI: {metrics['NMI']:.4f} | Sil: {metrics['Silhouette']:.4f}")

    # Export final results
    df = pd.DataFrame(all_results)
    out_csv = os.path.join(args.out_dir, "arise_v6_ablation_results.csv")
    df.to_csv(out_csv, index=False)
    print("\n" + "=" * 80)
    print(f"✅ Ablation Suite Completed! Full Results exported to: {out_csv}")
    print("=" * 80)

    # Print Summary Table
    if not df.empty:
        summary = df.groupby(['track', 'variant'])[['ARI', 'NMI', 'Silhouette']].agg(['mean', 'std'])
        print("\n🏆 Consolidated V6 Ablation Performance Summary:")
        print(summary)


if __name__ == '__main__':
    main()
