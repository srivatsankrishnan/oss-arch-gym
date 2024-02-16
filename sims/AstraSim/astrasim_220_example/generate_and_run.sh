#!/bin/bash

# generate workload egs from workload cfg
python3 workload_cfg_to_et.py --workload_cfg=workload_cfg.json --workload_et=workload-et/generated.%d.et

cd ..
# run with astrasim
SCRIPT_DIR=$(dirname "$(realpath $0)")
ASTRASIM_BIN="${SCRIPT_DIR}/astrasim_archgym_public/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
WORKLOAD="${SCRIPT_DIR}/astrasim_220_example/workload-et/generated"
SYSTEM="${SCRIPT_DIR}/astrasim_220_example/system.json"
NETWORK="${SCRIPT_DIR}/astrasim_220_example/network.yml"
MEMORY="${SCRIPT_DIR}/astrasim_220_example/memory.json"
COMM_GROUP="${SCRIPT_DIR}/astrasim_220_example/workload-et/generated.json"

${ASTRASIM_BIN} --workload-configuration=${WORKLOAD} --system-configuration=${SYSTEM} --network-configuration=${NETWORK} --remote-memory-configuration=${MEMORY} --comm-group-configuration=${COMM_GROUP}

