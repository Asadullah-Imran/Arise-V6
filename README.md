# 🧬 ARISE-V6 Component Ablation Framework

This repository establishes **ARISE-V6 (ARISE + 2-Stage Consensus DEC)** as the foundational base anchor model, systematically isolating and evaluating **26 architectural variants** across four core orthogonal tracks:
1. **Modality Fusion Mechanisms (`fusion-ablation/`)** (7 Variants)
2. **Loss Objectives & Optimization Regularizers (`loss-ablation/`)** (6 Variants)
3. **Encoder Backbones & Graph Topologies (`encoder-ablation/`)** (6 Variants)
4. **Contrastive Self-Supervised Learning Paradigms (`contrastive-ablation/`)** (7 Variants)

---

## 🏛️ Base Anchor Architecture (`ARISE-V6`)

- **Graph Scaffold:**
  - $G_{\text{sim}} = (V, E_{\text{sim}}, W_{\text{sim}})$: Transcriptomic kNN graph from PCA-projected RNA expressions with cosine similarity weights.
  - $G_{\text{dist}} = (V, E_{\text{dist}}, W_{\text{dist}})$: Physical coordinate spatial kNN graph.
  - $G_{\text{common}} = G_{\text{sim}} \cap G_{\text{dist}}$: Filtered shared-edge scaffold for the auxiliary modality (ADT / ATAC).
- **Backbone Encoders:** Dual 2-Layer GCNs for RNA ($X_{\text{RNA}}, G_{\text{sim}}$ & $X_{\text{RNA}}, G_{\text{dist}}$) + Single 2-Layer GCN for Aux modality ($X_{\text{aux}}, G_{\text{common}}$).
- **Fusion Mechanism:** Hierarchical Linear Projection:
  $$z_{\text{RNA}} = \text{Linear}([z_{\text{sim}} \parallel z_{\text{dist}}]), \quad Z = \text{Linear}([z_{\text{RNA}} \parallel z_{\text{aux}}])$$
- **Loss Objectives:** Multi-Head Feature Reconstruction $\mathcal{L}_{\text{rec}}$ + Spatial Neighborhood Contrastive Loss $\mathcal{L}_{\text{spatial}}$ + Parameter Regularization $\mathcal{L}_{\text{reg}}$.
- **Clustering Engine:** 2-Stage Consensus Deep Embedding Clustering (DEC) using Student-$t$ distribution $Q$ and target distribution $P$ with consensus KL divergence:
  $$q_{ij} = \frac{(1 + \|z_i - \mu_j\|^2 / \alpha)^{-\frac{\alpha+1}{2}}}{\sum_k (1 + \|z_i - \mu_k\|^2 / \alpha)^{-\frac{\alpha+1}{2}}}, \quad p_{ij} = \frac{q_{ij}^2 / \sum_i q_{ij}}{\sum_k (q_{ik}^2 / \sum_i q_{ik})}$$
  $$\mathcal{L}_{\text{KL}} = \text{KL}\left(\frac{\log Q_{\text{joint}} + \log Q_{\text{RNA}} + \log Q_{\text{aux}}}{3} \;\Big\|\; P\right)$$

---

## 📂 Repository Structure

```
Arise-V6/
├── README.md                           # Master architectural specifications
├── Arise_V6_Unified_Runner.py          # Unified CLI & Colab runner for all 25 variants
├── Arise_V6_Ablation_Colab.ipynb       # 1-Click Interactive Colab GPU Notebook
├── test_ablation_modules.py            # Automated unit test suite verifying all 25 models
├── generate_colab_nb.py                # Notebook generator utility
│
├── fusion-ablation/                    # 🔀 Track 1: Modality Fusion Mechanisms
│   ├── fusion_modules.py               # F0 - F5 modular layer implementations
│   ├── fusion_models.py                # Pluggable V6 fusion network models
│   └── run_fusion_ablation.py          # Dedicated fusion track CLI runner
│
├── loss-ablation/                      # 🎯 Track 2: Loss Objectives & Regularizers
│   ├── loss_modules.py                 # L0 - L5 modular objective implementations
│   ├── loss_models.py                  # Pluggable V6 loss network models
│   └── run_loss_ablation.py            # Dedicated loss track CLI runner
│
├── encoder-ablation/                   # ⚡ Track 3: Encoder Backbones & Topology
│   ├── encoder_modules.py              # E0 - E5 encoder backbones & graph filters
│   ├── encoder_models.py               # Pluggable V6 encoder network models
│   └── run_encoder_ablation.py         # Dedicated encoder track CLI runner
│
└── contrastive-ablation/               # 🔄 Track 4: Contrastive Learning Paradigms
    ├── contrastive_modules.py          # C0 - C6 modular contrastive objectives & discriminators
    ├── contrastive_models.py           # Pluggable V6 contrastive network models
    └── run_contrastive_ablation.py     # Dedicated contrastive track CLI runner
```

---

## 🔬 Detailed Theoretical Formulations of the 26 Variants

### Track 1: 🔀 `fusion-ablation/` (Modality Fusion Strategies)
*Encoders (Base V6 GCNs) and Loss (Base V6 DEC) are kept strictly constant.*

| Variant ID | Name | Mathematical Formulation & Strategy |
|:---|:---|:---|
| **`F0_Linear`** | **Linear Fusion (Base V6)** | $$z_{\text{RNA}} = \text{Linear}([z_{\text{sim}} \parallel z_{\text{dist}}]), \quad Z = \text{Linear}([z_{\text{RNA}} \parallel z_{\text{aux}}])$$ |
| **`F1_LocalGlobal`** | **Local-Global Attention Smoothing** | $$z_l = A_s \cdot z_{\text{intra}}, \quad S = \text{Softmax}(z_l z_l^T / \sqrt{d}), \quad z_g = S \cdot z_l, \quad z_{\text{RNA}} = \alpha z_g + (1-\alpha) z_l$$ |
| **`F2_VarianceWeight`** | **Dynamic Variance Weighting** | $$a_m = \frac{\exp(\text{Var}(z_m)/\tau)}{\sum_k \exp(\text{Var}(z_k)/\tau)}, \quad Z = a_{\text{RNA}} z_{\text{RNA}} + a_{\text{aux}} z_{\text{aux}}$$ |
| **`F3_SpatialCrossAttention`** | **Graph-Masked Cross-Attention** | $$\text{Attn}(Q_{\text{RNA}}, K_{\text{aux}}, V_{\text{aux}}) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M_{\text{spatial}}\right) V, \quad M_{ij} = \begin{cases} 0 & (i,j) \in E_{\text{dist}} \\ -\infty & \text{otherwise} \end{cases}$$ |
| **`F4_BilinearTensor`** | **Low-Rank Bilinear Tensor Fusion** | $$T = (W_r z_{\text{RNA}}) \odot (W_a z_{\text{aux}}), \quad Z = \text{Linear}([z_{\text{RNA}} \parallel T \parallel z_{\text{aux}}])$$ |
| **`F5_GatedMultiModal`** | **Learnable Sigmoid Gated Fusion** | $$g = \sigma(W_{\text{RNA}} z_{\text{RNA}} + W_{\text{aux}} z_{\text{aux}}), \quad Z = g \odot z_{\text{RNA}} + (1 - g) \odot z_{\text{aux}}$$ |
| **`F6_QKVCrossFusion`** | **Bidirectional QKV Cross-Fusion (CAGE)** | $$w_1 = \sigma\left(\frac{Q_{\text{aux}} \odot K_{\text{RNA}}}{\sqrt{d}}\right), \; w_2 = \sigma\left(\frac{Q_{\text{RNA}} \odot K_{\text{aux}}}{\sqrt{d}}\right), \quad Z = \text{Linear}([w_1 V_{\text{RNA}} \parallel w_2 V_{\text{aux}}])$$ |

---

### Track 2: 🎯 `loss-ablation/` (Optimization Objectives & Regularizers)
*Encoders (Base V6 GCNs) and Fusion (Base V6 Linear) are kept strictly constant.*

| Variant ID | Name | Mathematical Formulation & Loss Function |
|:---|:---|:---|
| **`L0_Standard`** | **Standard V6 Loss** | $$\mathcal{L}_{\text{total}} = 25\mathcal{L}_{\text{rec}} + 10\mathcal{L}_{\text{spatial}} + 0.1\mathcal{L}_{\text{KL}} + \mathcal{L}_{\text{reg}}$$ |
| **`L1_DenseRelational`** | **Dense Relational Gram Loss** | $$\mathcal{L}_{\text{dense}} = \frac{1}{N}\sum_{i=1}^N (\|Z_i - z_{\text{RNA},i}\|^2 + \|Z_i - z_{\text{aux},i}\|^2) + 0.1 \|Z_{\text{RNA}}Z_{\text{RNA}}^T - Z_{\text{aux}}Z_{\text{aux}}^T\|_F^2$$ |
| **`L2_SinkhornOT`** | **Sinkhorn Optimal Transport** | $$\mathcal{L}_{\text{OT}} = \min_{T \in \Pi} \langle T, C \rangle - \varepsilon H(T), \quad C_{ij} = 1 - \cos(z_{\text{RNA},i}, z_{\text{aux},j})$$ |
| **`L3_SpatialInfoNCE`** | **Spatial Multi-Modal InfoNCE** | $$\mathcal{L}_{\text{NCE}} = -\frac{1}{|E|} \sum_{(i,j) \in E_{\text{dist}}} \log \frac{\exp(\cos(z_{\text{RNA},i}, z_{\text{aux},j})/\tau)}{\sum_k \exp(\cos(z_{\text{RNA},i}, z_{\text{aux},k})/\tau)}$$ |
| **`L4_SpatialPotts`** | **Spatial Potts Consensus Regularizer** | $$\mathcal{L}_{\text{Potts}} = \frac{1}{2|E|} \sum_{(i,j) \in E_{\text{dist}}} \|q_i - q_j\|_2^2 \quad \text{(Penalizes spatial salt-and-pepper label noise)}$$ |
| **`L5_UncertaintyBalancing`** | **Homoscedastic Loss Balancing** | $$\mathcal{L}_{\text{total}} = \sum_{m} \exp(-s_m) \mathcal{L}_m + \sum_m s_m, \quad s_m = \log \sigma_m^2 \text{ (Learned Task Precision)}$$ |

---

### Track 3: ⚡ `encoder-ablation/` (Encoder Topologies & Backbones)
*Fusion (Base V6 Linear) and Loss (Base V6 DEC) are kept strictly constant.*

| Variant ID | Name | Encoder Backbone & Graph Topology Strategy |
|:---|:---|:---|
| **`E0_StandardGCN`** | **Standard Dual GCN (Base V6)** | 2-layer Graph Convolutional Networks ($in \to 512 \to 64$). |
| **`E1_GlobalTransformer`** | **Global Transformer Token Encoder** | Multi-head self-attention token projection: $\text{Trans}(X) + \text{GCN}(X, G)$. |
| **`E2_TriangularMotif`** | **3-Node Triangular Clique Motif ($M_3$)** | $$M_3 = (A \cdot A) \odot A, \quad W_{\text{sim}}' = k_1 W_{\text{sim}} + k_2 \frac{M_3}{\max(M_3)}$$ |
| **`E3_HigherOrderMotif`** | **3-Node + 4-Node Cycle Motifs ($M_3+M_4$)** | $$M_4 = (A \cdot A \cdot A) \odot A, \quad W_{\text{sim}}' = k_1 W + k_2 M_3 + k_3 M_4$$ |
| **`E4_HeatWavelet`** | **Spectral Heat Diffusion Wavelets** | Continuous multiscale Chebyshev diffusion: $\psi_t = \exp(-t \tilde{L}) = \sum_k c_k T_k(\hat{L})$. |
| **`E5_GATAttention`** | **Graph Attention Network (GATv2)** | Multi-head parameterized edge attention with residual LayerNorm connections. |

---

### Track 4: 🔄 `contrastive-ablation/` (Contrastive Learning Paradigms)
*Implements state-of-the-art biological spatial multi-omics self-supervised contrastive learning mechanisms adapted from Proust (2025), GATCL (2026), SpaMOAL, CoMo, and GRAS4T.*

```
                    Spatial Multi-Omics Contrastive Learning
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       │                              │                              │
  Cross-Modal                    Spatial DGI /                  Graph-View
   Alignment                      Local-Context                 Augmentation
  (GATCL 2026)                    (Proust 2025)                 Consistency
       │                              │                              │
   same spot                      spot embedding                 same graph
  RNA ↔ Protein/ATAC             vs shuffled summary            two augmentations
       │                              │                              │
       └──────────────┬───────────────┴──────────────┬───────────────┘
                      │                              │
                Cluster-Aware                  Hybrid Multi-Level
                  Prototypes                       Objective
                      │                              │
                same domain                   combines cross-modal,
                vs non-domain                 spatial DGI & clusters
```

| Variant ID | Name | Core Contrastive Objective & Mechanism | Key Reference |
|:---|:---|:---|:---|
| **`C0_BaseV6`** | **Base V6 Contrastive Anchor** | Standard ARISE-V6 pairwise distance-based contrastive loss on 1-hop spatial neighborhoods. | ARISE-V6 |
| **`C1_CrossModalCL`** | **Cross-Modality Contrastive Alignment** | **GATCL Formulation (Eq. 18–21)**:<br>$$\tilde{H}^{\text{RNA}}_i = \frac{H_i^{\text{RNA}}}{\|H_i^{\text{RNA}}\|_2}, \quad \tilde{H}^{\text{Aux}}_i = \frac{H_i^{\text{Aux}}}{\|H_i^{\text{Aux}}\|_2}$$<br>$$s_i^+ = \langle \tilde{H}_i^{\text{RNA}}, \tilde{H}_i^{\text{Aux}} \rangle$$<br>$$s_{i,j}^- = \langle \tilde{H}_i^R, \tilde{H}_j^A \rangle + \langle \tilde{H}_i^A, \tilde{H}_j^R \rangle + \langle \tilde{H}_i^A, \tilde{H}_j^A \rangle + \langle \tilde{H}_i^R, \tilde{H}_j^R \rangle$$<br>$$\mathcal{L}_{\text{CL}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(s_i^+ / \tau)}{\exp(s_i^+ / \tau) + \sum_{j \neq i}\exp(s_{i,j}^- / \tau)}$$ | GATCL (2026) |
| **`C2_ProustDGI`** | **Proust Deep Graph Infomax (CSL)** | **Proust Contrastive Self-Supervised Learning**:<br>Feature corruption $(\tilde{X}, A)$ with preserved spatial topology $A$.<br>Local neighborhood context readout: $S_i = \sigma\left(\frac{1}{k}\sum_{j \in \mathcal{N}(i)} Z_j + Z_i\right)$.<br>Bilinear scoring discriminator: $\mathcal{D}(Z, S) = \sigma(Z^T W S)$.<br>Dual cross-entropy maximization over real $(Z_i, S_i)$ and corrupted $(Z_i', S_i)$ pairs. | Proust (2025, *Genome Res.*) |
| **`C3_SpatialNeighborCL`** | **Spatial Topology Neighbor InfoNCE** | **Spatial Graph Preservation**:<br>Positive pairs: 1-hop physical spatial neighbors $(i, j) \in E_{\text{dist}}$.<br>Negative pairs: Non-neighboring spatial spots randomly sampled beyond 2-hop radius. | CoMo (2026), GRAS4T (2024) |
| **`C4_ClusterAwareCL`** | **Cluster-Aware Prototype InfoNCE** | **Semantic Domain Concordance**:<br>Constructs soft cluster prototypes $C_k = \frac{\sum_i q_{ik} z_i}{\sum_i q_{ik}}$ and optimizes spot-to-cluster InfoNCE loss using soft cluster assignment probability $q_{ik}$. | SpaMOAL (2026), CoMo (2026) |
| **`C5_GraphAugConsistency`** | **Dual-View Graph Augmentation CL** | **Topology & Feature Perturbation Invariance**:<br>Generates two stochastic views via edge drop ($p=0.15$) and feature masking ($p=0.15$), optimizing symmetric SimCLR InfoNCE loss between matched spot views. | STAIG (2025) |
| **`C6_HybridMultiLevelCL`** | **Hybrid Multi-Level Contrastive Suite** | **Comprehensive Joint Objective**:<br>$$\mathcal{L}_{\text{hybrid}} = \mathcal{L}_{\text{base}} + \lambda_1 \mathcal{L}_{\text{CrossModal}} + \lambda_2 \mathcal{L}_{\text{ProustDGI}} + \lambda_3 \mathcal{L}_{\text{ClusterAware}}$$ | Composite Frontier |

---

## 🚀 Execution & Usage Guide

### 1. Run Complete Ablation Suite via CLI:
```bash
# Run all 25 variants across all 6 datasets (3 random seeds each):
python Arise_V6_Unified_Runner.py --track all --dataset all --seeds 42 1234 2024 --epochs 400

# Run only the Contrastive track:
python Arise_V6_Unified_Runner.py --track contrastive --dataset all --seeds 42 1234 2024

# Run only the Fusion track:
python Arise_V6_Unified_Runner.py --track fusion --dataset all --seeds 42 1234 2024

# Run a specific variant (e.g. C2_ProustDGI on Dataset 0):
python Arise_V6_Unified_Runner.py --variant C2_ProustDGI --dataset 0 --seeds 42
```

### 2. Run Individual Track Sub-Runners:
```bash
# Contrastive track runner:
python contrastive-ablation/run_contrastive_ablation.py --dataset all --seeds 42 1234 2024

# Fusion track runner:
python fusion-ablation/run_fusion_ablation.py --dataset all --seeds 42 1234 2024

# Loss track runner:
python loss-ablation/run_loss_ablation.py --dataset all --seeds 42 1234 2024

# Encoder track runner:
python encoder-ablation/run_encoder_ablation.py --dataset all --seeds 42 1234 2024
```

### 3. Run on Google Colab (1-Click GPU Execution):
1. Open [`Arise_V6_Ablation_Colab.ipynb`](file:///Users/imran/Developer/FYDP/ARISE/Arise-V6/Arise_V6_Ablation_Colab.ipynb) in Google Colab.
2. Select your hardware accelerator (`GPU: T4 / A100`).
3. Select your track (`all`, `fusion`, `loss`, `encoder`, `contrastive`) or individual variant from the interactive form dropdown.
4. Click **Run All** to execute and benchmark.

---

## 📊 Automated Unit Test Verification

Run the test suite to verify forward-pass, Stage 1 representation learning, and Stage 2 consensus DEC loss computation for all 26 models:
```bash
python test_ablation_modules.py
```

```
======================================================================
Testing all 26 V6 Ablation Variants (Forward Pass & Loss Computation)
======================================================================

--- Testing Fusion Track (F0 - F6) ---
  [PASS] F0_Linear                    | S1 Loss: 104.0220 | S2 Loss: 104.0237
  [PASS] F1_LocalGlobal               | S1 Loss: 104.8657 | S2 Loss: 104.8671
  [PASS] F2_VarianceWeight            | S1 Loss: 104.1461 | S2 Loss: 104.1474
  [PASS] F3_SpatialCrossAttention     | S1 Loss: 103.9886 | S2 Loss: 103.9903
  [PASS] F4_BilinearTensor            | S1 Loss: 105.6043 | S2 Loss: 105.6077
  [PASS] F5_GatedMultiModal           | S1 Loss: 105.9319 | S2 Loss: 105.9359
  [PASS] F6_QKVCrossFusion            | S1 Loss: 105.4471 | S2 Loss: 105.4508

--- Testing Loss Track (L0 - L5) ---
  [PASS] L0_Standard                  | S1 Loss: 104.0439 | S2 Loss: 104.0460
  [PASS] L1_DenseRelational           | S1 Loss: 105.3727 | S2 Loss: 105.3748
  [PASS] L2_SinkhornOT                | S1 Loss: 105.1676 | S2 Loss: 105.1697
  [PASS] L3_SpatialInfoNCE            | S1 Loss: 104.9471 | S2 Loss: 104.9493
  [PASS] L4_SpatialPotts              | S1 Loss: 104.1605 | S2 Loss: 104.1623
  [PASS] L5_UncertaintyBalancing      | S1 Loss: 5.3138   | S2 Loss: 5.3374

--- Testing Encoder Track (E0 - E5) ---
  [PASS] E0_StandardGCN               | S1 Loss: 103.9676 | S2 Loss: 103.9695
  [PASS] E1_GlobalTransformer         | S1 Loss: 104.5788 | S2 Loss: 104.5835
  [PASS] E2_TriangularMotif           | S1 Loss: 104.7596 | S2 Loss: 104.7617
  [PASS] E3_HigherOrderMotif          | S1 Loss: 104.8150 | S2 Loss: 104.8167
  [PASS] E4_HeatWavelet               | S1 Loss: 103.8642 | S2 Loss: 103.8660
  [PASS] E5_GATAttention              | S1 Loss: 115.7790 | S2 Loss: 115.7814

--- Testing Contrastive Track (C0 - C6) ---
  [PASS] C0_BaseV6                    | S1 Loss: 103.9589 | S2 Loss: 103.9612
  [PASS] C1_CrossModalCL              | S1 Loss: 114.5308 | S2 Loss: 114.5324
  [PASS] C2_ProustDGI                 | S1 Loss: 105.5888 | S2 Loss: 105.5903
  [PASS] C3_SpatialNeighborCL         | S1 Loss: 107.4687 | S2 Loss: 107.4707
  [PASS] C4_ClusterAwareCL            | S1 Loss: 104.2810 | S2 Loss: 105.8924
  [PASS] C5_GraphAugConsistency       | S1 Loss: 105.0450 | S2 Loss: 105.0812
  [PASS] C6_HybridMultiLevelCL        | S1 Loss: 114.7988 | S2 Loss: 115.1225

======================================================================
🎉 ALL 26 V6 ABLATION VARIANTS PASSED VALIDATION PERFECTLY!
======================================================================
```

---

## 📈 Benchmark Datasets Supported

| Dataset ID | Name | Omics | Tissue | Platform | Spots | Clusters |
|:---:|:---|:---|:---|:---|:---:|:---:|
| `0` | **10x Human Lymph Node A1** | RNA + ADT | Human Lymph Node | 10x Visium | 3,484 | 9 |
| `1` | **10x Human Lymph Node D1** | RNA + ADT | Human Lymph Node | 10x Visium | 3,487 | 10 |
| `2` | **Mouse Brain E11 S1** | RNA + ATAC | Mouse Embryo Brain | Spatial-Epigenome | 2,473 | 11 |
| `3` | **Mouse Brain E13 S1** | RNA + ATAC | Mouse Embryo Brain | Spatial-Epigenome | 2,676 | 12 |
| `4` | **Mouse Brain E15 S1** | RNA + ATAC | Mouse Embryo Brain | Spatial-Epigenome | 3,241 | 13 |
| `5` | **Mouse Brain E18 S1** | RNA + ATAC | Mouse Embryo Brain | Spatial-Epigenome | 3,192 | 14 |
