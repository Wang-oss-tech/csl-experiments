#!/bin/bash

set -e

# 1D Broadcast Example
# P PEs in a row, each PE has N elements
# At step i, PE i broadcasts its data to all PEs

cslc --arch=wse2 ./layout.csl \
  --fabric-dims=11,3 \
  --fabric-offsets=4,1 \
  --params=P:4,N:8 \
  -o out \
  --memcpy --channels=1

cs_python run.py --name out
