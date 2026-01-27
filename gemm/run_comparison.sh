#!/usr/bin/env bash

set -e

echo "=========================================================================="
echo "Running P=8, Mt=Kt=Nt=14 Comparison: Sequential vs Pipelined"
echo "=========================================================================="
echo ""

echo "1. Running Sequential Multicasting..."
cd /home/william_wang_yale/R_2.6.0/csl-examples/experiments/gemm/summa_manual_multicasting
./commands_wse2.sh > /tmp/seq_output.log 2>&1
SEQ_CYCLES=$(grep '"cycle_count"' sim_stats.json | awk '{print $2}' | tr -d ',')
echo "   Sequential cycles: $SEQ_CYCLES"
echo ""

echo "2. Running Pipelined Multicasting..."
cd /home/william_wang_yale/R_2.6.0/csl-examples/experiments/gemm/summa_manual_multicasting_pipelined
./commands_wse2.sh > /tmp/pipe_output.log 2>&1
PIPE_CYCLES=$(grep '"cycle_count"' sim_stats.json | awk '{print $2}' | tr -d ',')
echo "   Pipelined cycles: $PIPE_CYCLES"
echo ""

echo "=========================================================================="
echo "Results Summary:"
echo "=========================================================================="
echo "Sequential: $SEQ_CYCLES cycles"
echo "Pipelined:  $PIPE_CYCLES cycles"
echo ""

# Calculate speedup
DIFF=$((SEQ_CYCLES - PIPE_CYCLES))
if [ $DIFF -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; $DIFF * 100.0 / $SEQ_CYCLES" | bc)
    echo "Pipelined WINS by $DIFF cycles ($SPEEDUP% speedup)"
else
    DIFF=$((-DIFF))
    SLOWDOWN=$(echo "scale=2; $DIFF * 100.0 / $SEQ_CYCLES" | bc)
    echo "Sequential WINS by $DIFF cycles ($SLOWDOWN% faster)"
fi
echo "=========================================================================="
