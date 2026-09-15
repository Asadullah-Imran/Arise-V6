#!/usr/bin/env python3
"""
================================================================================
  🎯 ARISE-V6 Loss Objective Ablation Runner (L0 - L5)
================================================================================
  L0_Standard:             Base V6 Loss (Recon + Spatial + Consensus DEC KL)
  L1_DenseRelational:      + Dense Relational Gram Matrix Frobenius Loss (from V7)
  L2_SinkhornOT:           + Entropic Optimal Transport Distribution Matching (from V12)
  L3_SpatialInfoNCE:       + Spatial Multi-Modal Contrastive InfoNCE Loss (from V13)
  L4_SpatialPotts:         + Soft Assignment Dirichlet MRF Smoothness Regularizer (from V14)
  L5_UncertaintyBalancing: + Homoscedastic Multi-Task Loss Weighting (Kendall & Gal)
================================================================================
"""
import os
import sys
import subprocess

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    unified_runner = os.path.join(root_dir, "Arise_V6_Unified_Runner.py")
    
    cmd = [sys.executable, unified_runner, "--track", "loss"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
