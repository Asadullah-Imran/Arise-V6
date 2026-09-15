#!/usr/bin/env python3
"""
================================================================================
  🔀 ARISE-V6 Fusion Ablation Runner (F0 - F6)
================================================================================
  F0_Linear:                Base V6 Hierarchical 2-stage Linear Projection
  F1_LocalGlobal:           Local-Global Attention Smoothing Fusion (from V3)
  F2_VarianceWeight:        Dynamic Latent Variance Modal Weighting (from V4)
  F3_SpatialCrossAttention:  Graph-Masked Spatial 1-Hop Cross-Attention (from V11)
  F4_BilinearTensor:        Outer-Product Bilinear Tensor Fusion
  F5_GatedMultiModal:       Learnable Elementwise Sigmoid Gated Fusion (from SpatialGlue)
  F6_QKVCrossFusion:        Bidirectional Symmetric QKV Cross-Attention Fusion (from CAGE)
================================================================================
"""
import os
import sys
import subprocess

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    unified_runner = os.path.join(root_dir, "Arise_V6_Unified_Runner.py")
    
    cmd = [sys.executable, unified_runner, "--track", "fusion"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
