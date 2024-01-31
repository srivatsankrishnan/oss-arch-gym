import argparse
import json
import sys
import os
current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_script_directory, ".."))
from astrasim_archgym_public.dse.conf_file_tools import workload_cfg_to_workload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload_cfg", type=str, help="path to workload_cfg")
    parser.add_argument("--workload_et", type=str, help="path to generated ets")
    args = parser.parse_args()
    print("REACHED workload_cfg_to_et.py")
    print("workload_cfg: ", args.workload_cfg)
    print("workload_et: ", args.workload_et)
    workload_cfg_dict = {}
    with open(args.workload_cfg, "r") as workload_cfg_f:
        workload_cfg_dict = json.load(workload_cfg_f)
    workload_cfg_to_workload(workload_cfg_dict, args.workload_et)

if __name__ == "__main__":
    main()
