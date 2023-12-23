#! /bin/bash -v

# Absolue path to this script
SCRIPT_DIR=$(dirname "$(realpath $0)")

# Absolute paths to useful directories
BINARY=$1
SYSTEM=$2
NETWORK=$3
WORKLOAD=$4

echo "SH BINARY: ${BINARY}"
echo "SH NETWORK: ${NETWORK}"
echo "SH SYSTEM: ${SYSTEM}"
echo "SH WORKLOAD: ${WORKLOAD}"

STATS="${SCRIPT_DIR:?}"/results/run_general

rm -rf "${STATS}"
mkdir "${STATS}"

"${BINARY}" \
--network-configuration="${NETWORK}" \
--system-configuration="${SYSTEM}" \
--workload-configuration="${WORKLOAD}" \
--path="${STATS}/" \
--run-name="sample_all_reduce" \

