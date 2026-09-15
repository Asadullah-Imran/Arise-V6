#!/usr/bin/env python3
"""
================================================================================
  🔬 ARISE-V6 Contrastive Learning Ablation Runner (C0 - C6)
================================================================================
  C0_BaseV6:               Base V6 Anchor (Recon + Spatial + DEC KL)
  C1_CrossModalCL:         Cross-Modality Contrastive Learning (from GATCL Paper)
  C2_ProustDGI:            Local Neighborhood Deep Graph Infomax (from Proust 2025)
  C3_SpatialNeighborCL:    Spatial Neighbor-Aware Contrastive Loss (from CoMo)
  C4_ClusterAwareCL:       Cluster-Aware / Prototype Contrastive Loss (from SpaMOAL)
  C5_GraphAugConsistency:  Dual-View Graph Augmentation Consistency (from GRAS4T)
  C6_HybridMultiLevelCL:   Hybrid Multi-Level CL (Cross-Modal + Spatial + Cluster)
================================================================================
"""
import os
import sys
import subprocess

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    unified_runner = os.path.join(root_dir, "Arise_V6_Unified_Runner.py")
    
    cmd = [sys.executable, unified_runner, "--track", "contrastive"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
