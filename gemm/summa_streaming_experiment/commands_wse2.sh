#!/usr/bin/env bash

set -e

# export SINGULARITYENV_SIMFABRIC_DEBUG=landing
export SINGULARITYENV_SIMFABRIC_DEBUG=inst_trace


cslc --arch=wse2 ./layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=P:4,Mt:14,Kt:14,Nt:14 \
--params=MEMCPYH2D_DATA_1_ID:2,MEMCPYH2D_DATA_2_ID:3 \
--memcpy --channels=1 -o out
cs_python run.py --name out
