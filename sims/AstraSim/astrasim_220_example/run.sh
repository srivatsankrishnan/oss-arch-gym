#!/bin/bash

# run with astrasim
SCRIPT_DIR=$(dirname "$(realpath $0)")
echo "SCRIPT DIR: ${SCRIPT_DIR}"

ASTRASIM_BIN=$1
SYSTEM=$2
NETWORK=$3
WORKLOAD=$4
GENERATE=$5

echo "SH WORKLOAD 1: ${WORKLOAD}"

if [ "$GENERATE" == "TRUE" ]; then
    echo "REACHED TRUE"
    python3 astrasim_220_example/workload_cfg_to_et.py --workload_cfg=${WORKLOAD} --workload_et=workload-et/generated.%d.eg
    WORKLOAD="${SCRIPT_DIR}/workload-et/generated"
else
    WORKLOAD="${SCRIPT_DIR}/workload-et/generated"
fi

MEMORY="${SCRIPT_DIR}/memory.json"

echo "SH BINARY: ${ASTRASIM_BIN}"
echo "SH NETWORK: ${NETWORK}"
echo "SH SYSTEM: ${SYSTEM}"
echo "SH WORKLOAD 2: ${WORKLOAD}"
echo "SH GENERATE: ${GENERATE}"

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