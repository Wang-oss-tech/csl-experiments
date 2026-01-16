#!/usr/bin/env bash

set -e

# Enable instruction trace for debugging
export SINGULARITYENV_SIMFABRIC_DEBUG=inst_trace

# P = 4 (4x4 PE grid)
# Mt, Kt, Nt = 4 (4x4 tiles per PE)
# Full matrices: A is 16x16, B is 16x16, C is 16x16

# Testing with P=2 (2x2 grid)
cslc --arch=wse2 ./layout.csl \
  --fabric-dims=9,4 \
  --fabric-offsets=4,1 \
  -o out \
  --params=P:2,Mt:4,Kt:4,Nt:4 \
  --memcpy --channels=1

cs_python run.py --name out
