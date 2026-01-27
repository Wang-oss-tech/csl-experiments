#!/usr/bin/env python3
"""
Analyze when pipelined multicasting outperforms sequential multicasting.

The key insight:
- Sequential: Total = P × (Broadcast + GEMM)
- Pipelined:  Total = Broadcast_0 + P × GEMM + overhead (broadcasts overlap with GEMM)
- Savings = (P-1) × Broadcast - overhead

Pipelined wins when: (P-1) × Broadcast > overhead + small_margin
"""

import math

def estimate_timing(P, Mt, Kt, Nt):
    """
    Estimate broadcast and GEMM times based on empirical data.
    
    From P=4, Mt=Kt=Nt=14 baseline:
    - Sequential total: 95,405 cycles
    - GEMM per step: ~11,897 cycles (from FMACS analysis)
    - Broadcast per step: ~11,954 cycles
    """
    
    # GEMM time scales with tile dimensions (Mt × Kt × Nt operations)
    # Base: 14×14×14 = 2,744 operations → 11,897 cycles
    # ~4.3 cycles per scalar multiply-add (including loop overhead)
    base_ops = 14 * 14 * 14
    ops = Mt * Kt * Nt
    gemm_time_per_step = 11897 * (ops / base_ops)
    
    # Broadcast time has two components:
    # 1. Data transfer: proportional to data size (Mt×Kt for A, Kt×Nt for B)
    # 2. Fabric latency: proportional to P (number of hops)
    
    # Base broadcast: A=14×14=196, B=14×14=196, P=4 → 11,954 cycles
    # Transfer time: ~30 cycles per element (2 cycles/element × 15 for both A and B)
    # Fabric latency: ~50 cycles per hop × P
    base_data = 2 * (14 * 14)  # A + B
    data_size = Mt * Kt + Kt * Nt
    transfer_time = 30 * (data_size / base_data)
    fabric_latency = 50 * P
    
    # Empirical fit: broadcast = transfer + latency + setup
    broadcast_time_per_step = transfer_time + fabric_latency + 11000
    
    return gemm_time_per_step, broadcast_time_per_step

def compare_versions(P, Mt, Kt, Nt):
    """Compare sequential vs pipelined for given parameters."""
    
    gemm_time, bcast_time = estimate_timing(P, Mt, Kt, Nt)
    
    # Sequential: all steps are broadcast + compute
    sequential_total = P * (bcast_time + gemm_time)
    
    # Pipelined: first broadcast, then overlapped steps, plus overhead
    pipeline_overhead = 2000  # Task management, state tracking
    pipelined_total = bcast_time + P * gemm_time + pipeline_overhead
    
    # Overlap savings
    savings = (P - 1) * bcast_time - pipeline_overhead
    speedup = (sequential_total - pipelined_total) / sequential_total * 100
    
    return {
        'P': P,
        'Mt': Mt, 'Kt': Kt, 'Nt': Nt,
        'gemm_per_step': gemm_time,
        'bcast_per_step': bcast_time,
        'ratio': bcast_time / gemm_time,
        'sequential_total': sequential_total,
        'pipelined_total': pipelined_total,
        'speedup_pct': speedup,
        'winner': 'Pipelined' if speedup > 0 else 'Sequential'
    }

if __name__ == '__main__':
    print("=" * 100)
    print("Analysis: When Does Pipelining Win?")
    print("=" * 100)
    print()
    
    test_configs = [
        # Current baseline
        (4, 14, 14, 14, "Baseline (current)"),
        
        # Increase P (longer broadcast chains)
        (6, 14, 14, 14, "Larger grid"),
        (8, 14, 14, 14, "Large grid"),
        (12, 14, 14, 14, "Very large grid"),
        (16, 14, 14, 14, "Huge grid"),
        
        # Decrease tiles (less compute)
        (4, 10, 10, 10, "Smaller tiles"),
        (4, 7, 7, 7, "Small tiles"),
        (4, 4, 4, 4, "Very small tiles"),
        
        # Increase tiles (more compute) - should favor sequential
        (4, 20, 20, 20, "Larger tiles"),
        (4, 28, 28, 28, "Large tiles"),
    ]
    
    results = []
    for P, Mt, Kt, Nt, desc in test_configs:
        result = compare_versions(P, Mt, Kt, Nt)
        result['description'] = desc
        results.append(result)
    
    # Print table
    print(f"{'Description':<20} | {'P':>3} | {'Tiles':>8} | {'B/G Ratio':>10} | {'Winner':>10} | {'Speedup':>8}")
    print("-" * 100)
    
    for r in results:
        tiles = f"{r['Mt']}×{r['Kt']}×{r['Nt']}"
        print(f"{r['description']:<20} | {r['P']:3d} | {tiles:>8} | "
              f"{r['ratio']:10.2f} | {r['winner']:>10} | {r['speedup_pct']:7.1f}%")
    
    print()
    print("=" * 100)
    print("Key Findings:")
    print("=" * 100)
    print()
    
    # Find best case for pipelining
    best_pipeline = max([r for r in results if r['winner'] == 'Pipelined'], 
                       key=lambda x: x['speedup_pct'], default=None)
    
    if best_pipeline:
        print(f"Best pipelined config: P={best_pipeline['P']}, "
              f"Tiles={best_pipeline['Mt']}×{best_pipeline['Kt']}×{best_pipeline['Nt']}")
        print(f"  Speedup: {best_pipeline['speedup_pct']:.1f}%")
        print(f"  Broadcast/GEMM ratio: {best_pipeline['ratio']:.2f}")
        print()
    
    print("Recommendations:")
    print("  1. To show pipelining advantage: Use P=8 or P=16 with Mt=Kt=Nt=14")
    print("  2. Alternative: Use P=4 with Mt=Kt=Nt=7 (small tiles)")
    print("  3. The larger the P, the more pipelining helps (linear broadcast chain)")
    print()
