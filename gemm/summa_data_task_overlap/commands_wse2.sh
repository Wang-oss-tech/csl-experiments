#!/usr/bin/env bash

set -e
export SINGULARITYENV_SIMFABRIC_DEBUG=landing


cslc --arch=wse2 ./layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=P:4,Mt:28,Kt:28,Nt:28 \
--memcpy --channels=1 -o out

cs_python run.py --name out
