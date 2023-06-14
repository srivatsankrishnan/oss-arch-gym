#! /bin/bash -v

# Absolue path to this script
SCRIPT_DIR=$(dirname "$(realpath $0)")

# Absolute paths to useful directories
BINARY="${SCRIPT_DIR:?}"/astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra
NETWORK="${SCRIPT_DIR:?}"/general_network.json
SYSTEM="${SCRIPT_DIR:?}"/general_system.txt
WORKLOAD="${SCRIPT_DIR:?}"/astra-sim/inputs/workload/Transformer_HybridParallel.txt
STATS="${SCRIPT_DIR:?}"/results/run_general

rm -rf "${STATS}"
mkdir "${STATS}"

"${BINARY}" \
--network-configuration="${NETWORK}" \
--system-configuration="${SYSTEM}" \
--workload-configuration="${WORKLOAD}" \
--path="${STATS}/" \
--run-name="sample_all_reduce" \
--num-passes=5 \
--comm-scale=50 \
--total-stat-rows=1 \
--stat-row=0 

