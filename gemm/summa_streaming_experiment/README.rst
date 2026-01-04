SUMMA GEMM with Streaming Data Transfer
========================================

This experiment modifies the ``gemm-collectives_2d`` benchmark to use **streaming mode** for data transfer instead of copy mode, demonstrating concurrent data streaming for matrix multiplication.

Motivation
----------

The original ``gemm-collectives_2d`` benchmark uses a sequential approach:

1. **Copy mode**: Copy entire A matrix tiles to all PEs (blocking)
2. **Copy mode**: Copy entire B matrix tiles to all PEs (blocking)
3. **RPC launch**: Trigger computation via ``main()``
4. **Compute**: Execute P-step SUMMA algorithm
5. **Copy mode**: Copy C result tiles back

This sequential approach means:

- Host waits for all A tiles to transfer before starting B transfer
- Device sits idle while data transfers
- No overlap between data transfer and potential preprocessing

**This experiment explores streaming mode** to:

- Stream A and B tiles concurrently/sequentially with less blocking
- Trigger computation only after data is ready
- Potentially enable better pipelining and bandwidth utilization

Changes from Original
---------------------

**layout.csl:**

- Added ``MEMCPYH2D_DATA_1_ID`` and ``MEMCPYH2D_DATA_2_ID`` parameters
- Added ``MEMCPYH2D_DATA_1`` and ``MEMCPYH2D_DATA_2`` color definitions
- Updated memcpy module import to include ``MEMCPYH2D_1`` and ``MEMCPYH2D_2``
- Updated color map documentation

**pe.csl:**

- Added ``h2d_a_iq`` and ``h2d_b_iq`` input queues for streaming
- Added ``memcpy_recv_a_task_id`` and ``memcpy_recv_b_task_id`` data task IDs
- Added ``stream_A_done_id`` and ``stream_B_done_id`` local task IDs
- Added counters ``num_recv_a``, ``num_recv_b`` and flags ``a_ready``, ``b_ready``
- Added ``memcpy_recv_a`` task to receive A tile elements via streaming
- Added ``memcpy_recv_b`` task to receive B tile elements via streaming
- Added ``stream_A_done`` and ``stream_B_done`` completion callbacks
- Bound streaming tasks and initialized queues (WSE-3)
- Kept original SUMMA algorithm and collectives unchanged

**run.py:**

- Extracted ``MEMCPYH2D_DATA_1`` and ``MEMCPYH2D_DATA_2`` from compile metadata
- Removed ``sym_A`` and ``sym_B`` symbol retrieval (no longer needed)
- Changed A transfer to **streaming mode** via ``MEMCPYH2D_DATA_1``
- Changed B transfer to **streaming mode** via ``MEMCPYH2D_DATA_2``
- Kept ``runner.launch("main")`` for triggering SUMMA computation
- Kept C retrieval in copy mode (streaming C back is a future optimization)

**commands_wse2.sh:**

- Added ``MEMCPYH2D_DATA_1_ID:2`` compile parameter
- Added ``MEMCPYH2D_DATA_2_ID:3`` compile parameter

Algorithm: SUMMA (Unchanged)
-----------------------------

The core SUMMA algorithm remains identical to the original:

**Grid:** P×P processing elements

**Per-PE data:**
- ``A_tile``: Mt×Kt portion of matrix A (column-major)
- ``B_tile``: Kt×Nt portion of matrix B (column-major)
- ``C_tile``: Mt×Nt result accumulator

**Execution (P iterations):**

For each step i = 0 to P-1:

1. PEs in column i broadcast their ``A_tile`` across their row (via ``mpi_x``)
2. PEs in row i broadcast their ``B_tile`` down their column (via ``mpi_y``)
3. Each PE performs local GEMM: ``C_tile += Ap × Bp``
4. Increment step and repeat

After P iterations, each PE holds its portion of the final result C.

Execution Flow
--------------

**1. Host prepares and streams data:**

.. code-block:: python

   # Transform A to column-major tiles
   A1 = A.reshape(h, Mt, w, Kt)
   A2 = A1.transpose(0, 2, 3, 1)
   A3 = A2.reshape(h, w, Mt*Kt)

   # Stream A tiles to all PEs via MEMCPYH2D_DATA_1
   runner.memcpy_h2d(MEMCPYH2D_DATA_1, A3.ravel(),
                     0, 0, w, h, Mt*Kt, streaming=True, ...)

   # Stream B tiles to all PEs via MEMCPYH2D_DATA_2
   runner.memcpy_h2d(MEMCPYH2D_DATA_2, B3.ravel(),
                     0, 0, w, h, Kt*Nt, streaming=True, ...)

**2. Device receives tiles (pe.csl):**

Each PE receives Mt×Kt elements for A and Kt×Nt elements for B:

.. code-block:: csl

   // Triggered for each A element
   task memcpy_recv_a(data: f32) void {
     A_tile[num_recv_a] = data;
     num_recv_a += 1;

     if (num_recv_a == Mt * Kt) {
       a_ready = true;
       @activate(stream_A_done_id);
     }
   }

   // Similar for B
   task memcpy_recv_b(data: f32) void { ... }

**3. Host launches SUMMA computation:**

After streaming completes, host triggers the algorithm:

.. code-block:: python

   runner.launch("main", nonblock=False)

**4. Device executes SUMMA:**

The ``main()`` function executes exactly as in the original benchmark:

- Initialize collectives (first iteration only)
- For each of P steps: broadcast A and B tiles, compute local GEMM
- Activate EXIT when all P iterations complete

**5. Host retrieves results:**

.. code-block:: python

   runner.memcpy_d2h(C3_1d_u32, sym_C, ..., streaming=False)
   # Transform back to row-major full matrix

Key Design Decisions
--------------------

**Streaming A and B separately:**

- Sequential streaming ensures proper ordering
- Each PE receives exactly Mt×Kt + Kt×Nt elements
- ``a_ready`` and ``b_ready`` flags ensure data integrity before computation

**Keeping RPC launch:**

- SUMMA requires coordinated start across all PEs
- RPC provides clear synchronization point
- Allows host to control when computation begins

**Not streaming C back (yet):**

- C is produced iteratively over P steps
- Streaming C would require coordination with SUMMA iterations
- Current approach: compute fully, then copy C back
- **Future optimization**: Stream C as it's computed

**Color allocation:**

- Color 2: ``MEMCPYH2D_DATA_1`` (A tiles)
- Color 3: ``MEMCPYH2D_DATA_2`` (B tiles)
- Avoids conflicts with collectives (0-1, 4-5) and memcpy reserved colors

Running the Example
-------------------

From this directory::

  ./commands_wse2.sh

Or manually::

  cslc --arch=wse2 ./layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \\
    --params=P:4,Mt:14,Kt:14,Nt:14 \\
    --params=MEMCPYH2D_DATA_1_ID:2,MEMCPYH2D_DATA_2_ID:3 \\
    --memcpy --channels=1 -o out
  cs_python run.py --name out

Expected output::

  SUCCESS

Performance Implications
------------------------

**Potential benefits:**

1. **Reduced host-device latency**: Streaming can overlap data transfer with setup
2. **Better bandwidth utilization**: Less idle time between A and B transfers
3. **Scalability**: Streaming scales better for larger tile sizes
4. **Pipelining**: Future work could pipeline multiple GEMM operations

**Considerations:**

- Streaming adds per-element task overhead vs. bulk copy
- For small tiles (14×14), overhead may dominate
- For large tiles, streaming benefits should be measurable
- Optimal choice depends on Mt, Kt, Nt sizes

Comparison with Original
-------------------------

+------------------------+-------------------------+---------------------------+
| Aspect                 | Original (Copy Mode)    | Streaming Experiment      |
+========================+=========================+===========================+
| A transfer             | Copy (blocking)         | Stream (per-element)      |
+------------------------+-------------------------+---------------------------+
| B transfer             | Copy (blocking)         | Stream (per-element)      |
+------------------------+-------------------------+---------------------------+
| Computation trigger    | RPC launch              | RPC launch (same)         |
+------------------------+-------------------------+---------------------------+
| SUMMA algorithm        | P-step collectives      | P-step collectives (same) |
+------------------------+-------------------------+---------------------------+
| C transfer             | Copy (blocking)         | Copy (blocking, same)     |
+------------------------+-------------------------+---------------------------+
| Concurrency            | Sequential phases       | Can overlap A/B streaming |
+------------------------+-------------------------+---------------------------+

Future Extensions
-----------------

**1. Stream C back during computation:**

Currently C is copied back after all P iterations. Could stream C incrementally as SUMMA progresses.

**2. Concurrent A and B streaming:**

Use ``nonblock=True`` to overlap A and B streaming operations.

**3. Pipelined multi-GEMM:**

Stream next A/B while computing current GEMM, creating a software pipeline.

**4. Dynamic tile sizes:**

Experiment with different Mt, Kt, Nt to find streaming breakeven point.

**5. Performance profiling:**

Compare wall-clock time, bandwidth utilization, and energy vs. copy mode.

**6. Hybrid approach:**

Use copy mode for small tiles, streaming for large tiles, decided at compile time.

Conclusion
----------

This experiment demonstrates that SUMMA GEMM can be adapted to use streaming data transfer while preserving the correctness and structure of the algorithm. The streaming approach opens opportunities for better pipelining and bandwidth utilization, especially for larger matrix tiles or multi-operation workflows.
