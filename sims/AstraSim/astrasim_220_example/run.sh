#!/bin/bash

# run with astrasim
SCRIPT_DIR=$(dirname "$(realpath $0)")
echo "SCRIPT DIR: ${SCRIPT_DIR}"

ASTRASIM_BIN=$1
SYSTEM=$2
NETWORK=$3
WORKLOAD=$4
GENERATE=$5

if [ "$GENERATE" == "TRUE" ]; then
    python3 workload_cfg_to_et.py --workload_cfg=${WORKLOAD} --workload_et=workload-et/generated.%d.eg
    WORKLOAD="${SCRIPT_DIR}/workload-et/generated"
else
    WORKLOAD="${SCRIPT_DIR}/workload-et/generated"
fi

MEMORY="${SCRIPT_DIR}/memory.json"

echo "SH BINARY: ${ASTRASIM_BIN}"
echo "SH NETWORK: ${NETWORK}"
echo "SH SYSTEM: ${SYSTEM}"
echo "SH WORKLOAD: ${WORKLOAD}"
echo "SH GENERATE: ${GENERATE}"

${ASTRASIM_BIN} \
--workload-configuration=${WORKLOAD} \
--system-configuration=${SYSTEM} \
--network-configuration=${NETWORK} \
--remote-memory-configuration=${MEMORY} \

echo "done script"
