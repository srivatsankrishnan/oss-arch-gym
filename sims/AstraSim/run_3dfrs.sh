#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath $0)")
BINARY="${SCRIPT_DIR:?}"/astrasim-archgym/astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra
WORKLOAD="${SCRIPT_DIR:?}"/3d_frs_baseline_allreduce065/allreduce_065.txt
SYSTEM="${SCRIPT_DIR:?}"/3d_frs_baseline_allreduce065/general_system.txt
NETWORK="${SCRIPT_DIR:?}"/3d_frs_baseline_allreduce065/3d_fc_ring_switch.json
RUN_DIR="${SCRIPT_DIR:?}"/3d_frs_baseline_allreduce065/

"${BINARY}" \
          --workload-configuration="${WORKLOAD}" \
          --system-configuration="${SYSTEM}" \
          --network-configuration="${NETWORK}" \
          --path="${RUN_DIR}" \
          --run-name="3d_frs_baseline_allreduce065" > ${RUN_DIR}/stdout