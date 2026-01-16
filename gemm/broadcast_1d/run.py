#!/usr/bin/env cs_python

# 1D Broadcast Test
# After P broadcast steps, all PEs should have PE (P-1)'s data

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()

# Get params from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
    compile_data = json.load(json_file)

P = int(compile_data['params']['P'])
N = int(compile_data['params']['N'])

print(f"Running 1D broadcast test: P={P} PEs, N={N} elements per PE")

# Initialize each PE with unique data
# PE i gets data = [i*N, i*N+1, i*N+2, ..., i*N+N-1] scaled
np.random.seed(42)
data = np.zeros((P, N), dtype=np.float32)
for i in range(P):
    data[i] = np.arange(i * N, (i + 1) * N, dtype=np.float32)

print(f"Initial data per PE:")
for i in range(P):
    print(f"  PE {i}: {data[i][:min(4,N)]}{'...' if N > 4 else ''}")

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

sym_data = runner.get_id("data")
sym_result = runner.get_id("result")

runner.load()
runner.run()

# Copy data to each PE
# data is shape (P, N), we copy to a P x 1 grid
runner.memcpy_h2d(sym_data, data.ravel(), 0, 0, P, 1, N,
    streaming=False,
    data_type=MemcpyDataType.MEMCPY_32BIT,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False)

print("Data copied to device, launching broadcast...")

# Launch the broadcast
runner.launch("main", nonblock=False)

print("Broadcast complete, reading results...")

# Read results from all PEs
result_raw = np.zeros(P * N, dtype=np.uint32)
runner.memcpy_d2h(result_raw, sym_result, 0, 0, P, 1, N,
    streaming=False,
    data_type=MemcpyDataType.MEMCPY_32BIT,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False)

result = result_raw.view(np.float32).reshape(P, N)

runner.stop()

# After P steps, the last broadcast was from PE (P-1)
# So all PEs should have PE (P-1)'s data
expected = data[P - 1]

print(f"\nExpected (PE {P-1}'s data): {expected[:min(4,N)]}{'...' if N > 4 else ''}")
print(f"\nResults per PE:")
for i in range(P):
    match = np.allclose(result[i], expected)
    status = "✓" if match else "✗"
    print(f"  PE {i}: {result[i][:min(4,N)]}{'...' if N > 4 else ''} {status}")

# Verify all PEs have correct result
all_correct = True
for i in range(P):
    if not np.allclose(result[i], expected):
        print(f"FAIL: PE {i} has incorrect data")
        all_correct = False

if all_correct:
    print("\nSUCCESS: All PEs received correct broadcast data")
else:
    print("\nFAILED: Some PEs have incorrect data")
    exit(1)
