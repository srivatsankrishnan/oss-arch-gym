#!/bin/bash

# run with astrasim
SCRIPT_DIR=$(dirname "$(realpath $0)")
echo "SCRIPT DIR: ${SCRIPT_DIR}"

ASTRASIM_BIN=$1
SYSTEM=$2
NETWORK=$3
WORKLOAD="${SCRIPT_DIR}/workload-et/generated"
GENERATE=$4
MEMORY="${SCRIPT_DIR}/memory.json"

echo "SH BINARY: ${ASTRASIM_BIN}"
echo "SH NETWORK: ${NETWORK}"
echo "SH SYSTEM: ${SYSTEM}"
echo "SH WORKLOAD: ${WORKLOAD}"
echo "SH GENERATE: ${GENERATE}"
echo "SH MEMORY: ${MEMORY}"

${ASTRASIM_BIN} \
--workload-configuration=${WORKLOAD} \
--system-configuration=${SYSTEM} \
--network-configuration=${NETWORK} \
--remote-memory-configuration=${MEMORY} \

echo "done script"

'''
astrasim_archgym_public/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
--workload-configuration=/home/archgym/workspace/aditi_jared/oss-arch-gym/sims/AstraSim/astrasim_220_example/workload-et/generated \
--system-configuration=/home/archgym/workspace/aditi_jared/oss-arch-gym/sims/AstraSim/astrasim_220_example/system.json \
--network-configuration=/home/archgym/workspace/aditi_jared/oss-arch-gym/sims/AstraSim/astrasim_220_example/network.yml \
--remote-memory-configuration=/home/archgym/workspace/aditi_jared/oss-arch-gym/sims/AstraSim/astrasim_220_example/memory.json
'''