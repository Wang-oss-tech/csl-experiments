#!/usr/bin/env bash

set -e

echo "==================================="
echo "Testing 4-Color Pipeline Compilation"
echo "==================================="
echo ""

cd /home/william_wang_yale/R_2.6.0/csl-examples/experiments/gemm/summa_4color_pipelined

# Clean previous build
rm -rf out/

echo "Running compilation..."
./commands_wse2.sh

if [ -f out/sim_stats.json ]; then
    echo ""
    echo "✅ SUCCESS! Compilation and simulation completed."
    echo ""
    CYCLES=$(cat out/sim_stats.json | grep '"cycle_count"' | head -1 | awk '{print $2}' | tr -d ',')
    echo "Cycle count: $CYCLES"
    echo ""
    
    # Compare with sequential
    echo "Comparing with sequential version..."
    cd ../summa_manual_multicasting
    if [ -f out/sim_stats.json ]; then
        SEQ_CYCLES=$(cat out/sim_stats.json | grep '"cycle_count"' | head -1 | awk '{print $2}' | tr -d ',')
        echo "Sequential:      $SEQ_CYCLES cycles"
        echo "4-Color Pipeline: $CYCLES cycles"
        echo ""
        
        DIFF=$((SEQ_CYCLES - CYCLES))
        if [ $DIFF -gt 0 ]; then
            PERCENT=$(echo "scale=2; ($DIFF * 100.0) / $SEQ_CYCLES" | bc)
            echo "🎉 WINNER: 4-Color Pipeline!"
            echo "Speedup: $DIFF cycles ($PERCENT% faster)"
        else
            DIFF=$((-DIFF))
            PERCENT=$(echo "scale=2; ($DIFF * 100.0) / $CYCLES" | bc)
            echo "Sequential still wins"
            echo "Overhead: $DIFF cycles ($PERCENT% slower)"
        fi
    else
        echo "Sequential version not compiled. Run it first to compare."
    fi
else
    echo "❌ FAILED - no output generated"
    echo "Check compilation errors above"
fi

echo "==================================="
