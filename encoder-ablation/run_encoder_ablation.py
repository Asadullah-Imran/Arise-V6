#!/usr/bin/env python3
"""
================================================================================
  ⚡ ARISE-V6 Encoder & Topology Ablation Runner (E0 - E5)
================================================================================
  E0_StandardGCN:          Base V6 Dual 2-Layer GCNs
  E1_GlobalTransformer:    Multi-Head Self-Attention Transformer Encoders (from V1)
  E2_TriangularMotif:      3-Node Triangular Clique Motif Topology M3 (from V2)
  E3_HigherOrderMotif:     3-Node + 4-Node Cycle Motifs M3 + M4 (from V9)
  E4_HeatWavelet:          Spectral Chebyshev Heat Kernel Wavelet Diffusion (from V10)
  E5_GATAttention:         Graph Attention Network (GATv2) Backbone (from SpatialGlue)
================================================================================
"""
import os
import sys
import subprocess

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    unified_runner = os.path.join(root_dir, "Arise_V6_Unified_Runner.py")
    
    cmd = [sys.executable, unified_runner, "--track", "encoder"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
