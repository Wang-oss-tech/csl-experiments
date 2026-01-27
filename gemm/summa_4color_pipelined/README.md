# SUMMA with 4-Color Alternating Pipeline

## Strategy

This implementation uses **4 colors** (2 pairs) to achieve pipelining **without runtime reconfiguration overhead**.

### Key Innovations

1. **No `@block()` calls during execution**
   - All routing is configured statically at compile time
   - Eliminates the primary overhead from the previous pipelined version

2. **Color Alternation**
   - Even steps (0,2,4,...): Use `a_color_0`, `b_color_0`
   - Odd steps (1,3,5,...): Use `a_color_1`, `b_color_1`
   - Colors alternate naturally without reconfiguration

3. **True Task-Based GEMM**
   - `gemm_task()` is activated (not called synchronously)
   - Allows task scheduler to manage execution efficiently
   - Enables true instruction-level overlap

4. **Pipeline Flow**
   ```
   Broadcast(0) → Complete
   Broadcast(1) starts (async) + GEMM(0) starts (task)
   Broadcast(2) starts (async) + GEMM(1) starts (task)
   ...
   ```

## Expected Performance

### Overhead Elimination
- **Previous pipelined**: ~890 cycles overhead from reconfiguration per step
- **This version**: 0 cycles reconfiguration overhead

### Predicted Cycle Counts (P=4, 14×14)

| Version | Cycle Count | Notes |
|---------|-------------|-------|
| Sequential | 95,405 | Baseline |
| Previous pipelined | 96,295 | +890 (overhead > benefit) |
| **4-color (this)** | **~88,000** | **Should win!** |

**Why it should work:**
- No reconfiguration overhead (saves ~3,560 cycles total)
- Proper task-based GEMM (better scheduling)
- True async overlap of broadcast and compute

## Files

- `layout.csl`: Defines 4 colors with static routing
- `pe.csl`: Alternating color logic + task-based GEMM
- `run.py`: Standard host program
- `commands_wse2.sh`: Compile and run script

## Usage

```bash
ssh cer-usn-01
cd /home/william_wang_yale/R_2.6.0/csl-examples/experiments/gemm/summa_4color_pipelined
./commands_wse2.sh
```

## Comparison Test

To compare with sequential:
```bash
# Sequential
cd ../summa_manual_multicasting
./commands_wse2.sh
cat out/sim_stats.json | grep cycles

# 4-color pipelined
cd ../summa_4color_pipelined
./commands_wse2.sh
cat out/sim_stats.json | grep cycles
```

## Scalability

This approach scales well:
- **P ≤ 8**: Only 4 colors needed (plenty of headroom)
- **P = 16**: Can extend to 8 colors or cycle through 4
- **P = 32+**: Use wavefront pipelining instead

## Theory

The key insight: **Static routing is fast, dynamic routing is slow.**

By using separate colors for each broadcast pattern, we eliminate the need to:
1. Block colors (`@block()`)
2. Write color config registers
3. Wait for colors to become idle

The fabric hardware handles multiple colors in parallel naturally, so using 4 colors instead of 2 doesn't add overhead—it removes it!
