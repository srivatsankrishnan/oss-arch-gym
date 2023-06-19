#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.append(os.path.abspath('./../'))
import home_settings
from top.main_FARSI import run_FARSI_only_simulation
from settings import config
import os
import itertools
# main function
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from visualization_utils.vis_hardware import *
import numpy as np
from specs.LW_cl import *
from specs.database_input import  *
import math
import matplotlib.colors as colors
#import pandas
import matplotlib.colors as mcolors
import pandas as pd
import argparse, sys
import data_collection.collection_utils.what_ifs.FARSI_what_ifs as wf


#  selecting the database based on the simulation method (power or performance)
if config.simulation_method == "power_knobs":
    from specs import database_input_powerKnobs as database_input
elif config.simulation_method == "performance":
    from specs import database_input
else:
    raise NameError("Simulation method unavailable")

if __name__ == "__main__":
    case_study = "simple_sim_run"
    file_prefix = config.FARSI_simple_sim_run_study_prefix
    current_process_id = 0
    total_process_cnt = 1
    #starting_exploration_mode = config.exploration_mode
    print('case study:' + case_study)

    # -------------------------------------------
    # set result folder
    # -------------------------------------------
    result_home_dir_default = os.path.join(os.getcwd(), "data_collection/data/" + case_study)
    result_home_dir = os.path.join(config.home_dir, "data_collection/data/" + case_study)
    date_time = datetime.now().strftime('%m-%d_%H-%M_%S')
    result_folder = os.path.join(result_home_dir,
                                 date_time)

    # -------------------------------------------
    # set parameters
    # -------------------------------------------
    experiment_repetition_cnt = 1
    reduction = "most_likely"
    #workloads = {"audio_decoder", "edge_detection"}
    #workloads = {"audio_decoder"}
    #workloads = {"edge_detection"}
    #workloads = {"hpvm_cava"}
    workloads = {"partial_SOC_example_hard"}
    workloads = {"SOC_example_8p"}
    tech_node_SF = {"perf":1, "energy":{"non_gpp":.064, "gpp":1}, "area":{"non_mem":.0374 , "mem":.079, "gpp":1}}   # technology node scaling factor
    db_population_misc_knobs = {"ip_freq_correction_ratio": 1, "gpp_freq_correction_ratio": 1,
                                "tech_node_SF":tech_node_SF,
                                "base_budget_scaling":{"latency":.5, "power":1, "area":1}}
    sw_hw_database_population = {"db_mode": "parse", "hw_graph_mode": "parse",
                                 "workloads": workloads, "misc_knobs":db_population_misc_knobs}

    # -------------------------------------------
    #  distribute the work
    # -------------------------------------------
    work_per_process = math.ceil(experiment_repetition_cnt / total_process_cnt)
    run_ctr = 0
    # -------------------------------------------
    # run the combination and collect the data
    # -------------------------------------------
    # -------------------------------------------
    # collect the exact hw sampling
    # -------------------------------------------
    accuracy_percentage = {}
    accuracy_percentage["sram"] = accuracy_percentage["dram"] = accuracy_percentage["ic"] = accuracy_percentage["gpp"] = accuracy_percentage[
        "ip"] = \
        {"latency": 1,
         "energy": 1,
         "area": 1,
         "one_over_area": 1}
    hw_sampling = {"mode": "exact", "population_size": 1, "reduction": reduction,
                   "accuracy_percentage": accuracy_percentage}



    burst_sizes = [64, 128, 256]
    queue_sizes = [1, 2, 4, 8]
    for burst_size in burst_sizes:
        for queue_size in queue_sizes:

            db_input = database_input_class(sw_hw_database_population)
            print("hw_sampling:" + str(hw_sampling))
            print("budget set to:" + str(db_input.get_budget_dict("glass")))
            unique_suffix = str(total_process_cnt) + "_" + str(current_process_id) + "_" + str(run_ctr)
            config.default_cmd_queue_size = queue_size
            config.default_data_queue_size = queue_size
            config.default_burst_size = burst_size
            dse_hndlr = run_FARSI_only_simulation(result_folder, unique_suffix, db_input, hw_sampling, sw_hw_database_population["hw_graph_mode"])
            run_ctr += 1

            # write the results in the general folder
            result_dir_specific = os.path.join(result_folder, "result_summary")
            reason_to_terminate = "simple_sim_run"
    #        wf.write_one_results(dse_hndlr.dse.so_far_best_sim_dp, dse_hndlr.dse, reason_to_terminate, case_study, result_dir_specific,
    #                  unique_suffix,
    #                  file_prefix + "_" + str(current_process_id) + "_" + str(total_process_cnt))

            # write the results in the specific folder
            result_folder_modified = result_folder + "/runs/" + str(run_ctr) + "/"
            os.system("mkdir -p " + result_folder_modified)

            for key, val in dse_hndlr.dse.so_far_best_sim_dp.dp_stats.SOC_metric_dict["latency"]["glass"][0].items():
                print("lat is {} for {}".format(val, key))
                lat = val
                burst_size = config.default_burst_size
                queue_size = config.default_data_queue_size
                print("burst size is {}".format(burst_size))
                print("queue size is {}".format(queue_size))

            with open(result_folder+"/latency_in_us.txt", "a") as f:
               f.write("burst_size={}, queue_size={}, farsi_simtime={}\n".format(burst_size,
                                                                            queue_size,
                                                                            lat*1000000))
            wf.copy_DSE_data(result_folder_modified)
    #        wf.write_one_results(dse_hndlr.dse.so_far_best_sim_dp, dse_hndlr.dse, reason_to_terminate, case_study,
    #                  result_folder_modified, unique_suffix,
    #                  file_prefix + "_" + str(current_process_id) + "_" + str(total_process_cnt))
