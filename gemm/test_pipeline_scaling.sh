#!/usr/bin/env bash

# Test script to demonstrate pipelining advantage
# Run both sequential and pipelined versions with different parameters

set -e

echo "======================================================================"
echo "Testing Sequential vs Pipelined Multicasting"
echo "Goal: Show when pipelining provides advantage"
echo "======================================================================"
echo ""

# Test 1: Current baseline (P=4, tiles=14×14)
echo "Test 1: Baseline - P=4, Mt=Kt=Nt=14"
echo "Expected: Sequential wins (small broadcast time)"
echo ""

# Test 2: Large grid (P=8, tiles=14×14)
echo "Test 2: Large grid - P=8, Mt=Kt=Nt=14"
echo "Expected: Pipelined wins (broadcast time increases with P)"
echo "Broadcast grows linearly, GEMM stays constant"
echo ""

# Test 3: Very large grid (P=16, tiles=14×14) 
echo "Test 3: Very large grid - P=16, Mt=Kt=Nt=14"
echo "Expected: Pipelined wins significantly (long broadcast chains)"
echo ""

# Test 4: Small tiles (P=4, tiles=7×7)
echo "Test 4: Small tiles - P=4, Mt=Kt=Nt=7"
echo "Expected: Pipelined wins (less compute, fixed broadcast latency)"
echo ""

# Test 5: Very small tiles (P=4, tiles=4×4)
echo "Test 5: Very small tiles - P=4, Mt=Kt=Nt=4"
echo "Expected: Pipelined wins more (minimal compute, broadcast latency dominates)"
echo ""

echo "======================================================================"
echo "Prediction Summary:"
echo "======================================================================"
echo "| Test | P  | Tiles | Broadcast/GEMM Ratio | Winner      | Speedup |"
echo "|------|----| ------|----------------------|-------------|---------|"
echo "| 1    | 4  | 14×14 | 1:1 (balanced)       | Sequential  | -0.9%   |"
echo "| 2    | 8  | 14×14 | 2:1 (bcast 2× GEMM)  | Pipelined   | ~12%    |"
echo "| 3    | 16 | 14×14 | 4:1 (bcast 4× GEMM)  | Pipelined   | ~25%    |"
echo "| 4    | 4  | 7×7   | 2:1                  | Pipelined   | ~8%     |"
echo "| 5    | 4  | 4×4   | 3:1                  | Pipelined   | ~15%    |"
echo "======================================================================"
echo ""
echo "To run experiments:"
echo "  1. cd summa_manual_multicasting"
echo "  2. Edit commands_wse2.sh: --params=P:8,Mt:14,Kt:14,Nt:14"
echo "  3. ./commands_wse2.sh"
echo "  4. Check sim_stats.json"
echo "  5. Repeat for summa_manual_multicasting_pipelined"
echo "  6. Compare cycle_count between versions"
