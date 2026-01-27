#!/usr/bin/env bash

set -e

echo "=========================================================================="
echo "P=16 Test: Maximum Grid, Minimum Tiles (7×7×7)"
echo "=========================================================================="
echo ""

# Test Sequential (will take ~3-5 minutes for 16×16 grid)
echo "1. Running Sequential Multicasting (P=16, Mt=Kt=Nt=7)..."
cd /home/william_wang_yale/R_2.6.0/csl-examples/experiments/gemm/summa_manual_multicasting
./commands_wse2.sh
echo "   Sequential complete!"
echo ""

# Test Pipelined
echo "2. Running Pipelined Multicasting (P=16, Mt=Kt=Nt=7)..."
cd /home/william_wang_yale/R_2.6.0/csl-examples/experiments/gemm/summa_manual_multicasting_pipelined
./commands_wse2.sh
echo "   Pipelined complete!"
echo ""

# Compare Results
echo "3. Analyzing results..."
cd /home/william_wang_yale/R_2.6.0/csl-examples/experiments/gemm
python3 << 'EOF'
import json

with open('summa_manual_multicasting/sim_stats.json') as f:
    seq = json.load(f)['cycle_count']
    
with open('summa_manual_multicasting_pipelined/sim_stats.json') as f:
    pipe = json.load(f)['cycle_count']

print("=" * 60)
print("FINAL TEST: P=16, Mt=Kt=Nt=7 (Maximum Grid, Minimum Tiles)")
print("=" * 60)
print(f"Sequential:  {seq:,} cycles")
print(f"Pipelined:   {pipe:,} cycles")
print(f"Difference:  {seq-pipe:,} cycles")
print(f"Speedup:     {(seq-pipe)/seq*100:.2f}%")
print(f"Winner:      {'🎉🎉🎉 PIPELINED WINS!' if pipe < seq else 'Sequential (still)'}")
print("=" * 60)

# Expected results
print("\nExpected:")
print("  Broadcast per step: ~15,000-20,000 cycles (15 hops)")
print("  GEMM per step:      ~1,500 cycles")
print("  Broadcast/GEMM ratio: 10-13×")
print("  Pipeline should save: (16-1) × 17,500 = ~262,500 cycles")
print("  Minus overhead: ~2,000 cycles")
print("  Net speedup: ~30-40%")
EOF

echo ""
echo "=========================================================================="
echo "Test complete!"
echo "=========================================================================="
