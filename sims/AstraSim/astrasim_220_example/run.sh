#!/bin/bash

# run with astrasim
SCRIPT_DIR=$(dirname "$(realpath $0)")
echo "SCRIPT DIR: ${SCRIPT_DIR}"

ASTRASIM_BIN=$1
SYSTEM=$2
NETWORK=$3
WORKLOAD="${SCRIPT_DIR}/workload-et/generated"
COMM_GROUP="${SCRIPT_DIR}/workload-et/generated.json"
GENERATE=$4
MEMORY="${SCRIPT_DIR}/memory.json"
LOG="${SCRIPT_DIR}/log.log"

echo "SH BINARY: ${ASTRASIM_BIN}"
echo "SH NETWORK: ${NETWORK}"
echo "SH SYSTEM: ${SYSTEM}"
echo "SH WORKLOAD: ${WORKLOAD}"
echo "SH GENERATE: ${GENERATE}"
echo "SH MEMORY: ${MEMORY}"

echo "RUN COMMAND" "${ASTRASIM_BIN} --workload-configuration=${WORKLOAD} --system-configuration=${SYSTEM} --network-configuration=${NETWORK} --remote-memory-configuration=${MEMORY} --comm-group-configuration=${COMM_GROUP} --log-path=${LOG}"

${ASTRASIM_BIN} \
--workload-configuration=${WORKLOAD} \
--system-configuration=${SYSTEM} \
--network-configuration=${NETWORK} \
--remote-memory-configuration=${MEMORY} \
--comm-group-configuration=${COMM_GROUP} \
--log-path=${LOG}

echo "done script"
