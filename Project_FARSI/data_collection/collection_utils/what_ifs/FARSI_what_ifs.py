#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.append(os.path.abspath('./../'))
#import home_settings
from top.main_FARSI import run_FARSI
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

#  selecting the database based on the simulation method (power or performance)
if config.simulation_method == "power_knobs":
    from specs import database_input_powerKnobs as database_input
elif config.simulation_method == "performance":
    from specs import database_input
else:
    raise NameError("Simulation method unavailable")


# ------------------------------
# Functionality:
#    show the the result of power/performance/area sweep 
# Variables:
#      full_dir_addr: name of the directory to get data from
#      full_file_addr: name of the file to get data from
# ------------------------------
def plot_3d_dist(full_dir_addr, full_file_addr, workloads):
    # getting data
    df = pd.read_csv(full_file_addr)

    # get avarages
    grouped_multiple = df.groupby(['latency_budget', 'power_budget', "area_budget"]).agg(
        {'latency': ['mean'], "power": ["mean"], "area": ["mean"], "cost":["mean"]})
    # the follow two lines is really usefull when we have mulitple aggregrations for each
    # key above, e.g., latency:['mean, 'max']
    grouped_multiple.columns = ['latency_avg', 'power_avg', 'area_avg', 'cost_avg']
    grouped_multiple = grouped_multiple.reset_index()

    # calculate the distance to goal and insert into the df
    dist_list = []
    for idx, row in grouped_multiple.iterrows():
        latency_dist = max(0, row['latency_avg'] - row['latency_budget'])/row['latency_budget']
        power_dist = max(0, row['power_avg'] - row['power_budget'])/row['power_budget']
        area_dist = max(0, row['area_avg'] - row['area_budget'])/row['area_budget']
        dist_list.append(latency_dist+ power_dist+ area_dist)
    grouped_multiple.insert(2, "norm_dist", dist_list, True)

    # get the data
    latency_budget = grouped_multiple["latency_budget"]
    power_budget = grouped_multiple["power_budget"]
    area_budget = grouped_multiple["area_budget"]

    color_values = []
    for el in grouped_multiple["norm_dist"]:
        if el == 0:
            color_values.append(0)
        else:
            color_values.append(el + max(grouped_multiple["norm_dist"]))

    #color_values = grouped_multiple["norm_dist"]
    print("maximum distance" + str(max(color_values)))
    X = latency_budget 
    Y = power_budget
    Z = area_budget

    """
    X = [el/min(latency_budget) for el in latency_budget]
    Y = [el/min(power_budget) for el in power_budget]
    Z = [el/min(area_budget) for el in area_budget]
    """

    # 3D plot, with color
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    bounds = np.array(list(np.arange(0, 1+.005, .01)))
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=230)

    # 3D plot, with color
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    color_values_norm = [col/max(max(color_values), .0000000000001) for col in color_values]
    p = ax.scatter(X, Y, Z, norm=norm, c=color_values_norm, cmap=plt.get_cmap("jet"))
    plt.colorbar(p)

    ax.set_xlabel("latency" + "(ms)")
    ax.set_ylabel("power"+ "(mw)")
    ax.set_zlabel("area" + "(mm2)")
    ax.set_title('Budget Sweep for ' + list(workloads)[0] + '.\n Hotter col= higher dist to budget, max:' +
                 str(max(color_values)) + " min:" + str(min(color_values)))
    fig.savefig(os.path.join(full_dir_addr, config.FARSI_cost_correlation_study_prefix +"_3d.png"))
    plt.show()


# copy the DSE results to the result dir
def copy_DSE_data(result_dir):
    #result_dir_specific = os.path.join(result_dirresult_summary")
    os.system("cp " + config.latest_visualization+"/*" + " " + result_dir)


# ------------------------------
# Functionality:
#     write the results into a file
# Variables:
#      sim_dp: design point simulation
#      result_dir: result directory
#      unique_number: a number to differentiate between designs
#      file_name: output file name
# ------------------------------
def write_one_results(sim_dp, dse, reason_to_terminate, case_study, result_dir_specific, unique_number, file_name):
    """
    def convert_dict_to_parsable_csv(dict_):
        list = []
        for k,v in dict_.items():
            list.append(str(k)+"="+str(v))
        return list
    """
    def convert_tuple_list_to_parsable_csv(list_):
        result = ""
        for k, v in list_:
            result +=str(k) + "=" + str(v) + "___"
        return result

    def convert_dictionary_to_parsable_csv_with_semi_column(dict_):
        result = ""
        for k, v in dict_.items():
            result +=str(k) + "=" + str(v) + ";"
        return result



    if not os.path.isdir(result_dir_specific):
        os.makedirs(result_dir_specific)


    compute_system_attrs = sim_dp.dp_stats.get_compute_system_attr()
    bus_system_attrs = sim_dp.dp_stats.get_bus_system_attr()
    memory_system_attrs = sim_dp.dp_stats.get_memory_system_attr()
    speedup_dict, speedup_attrs = sim_dp.dp_stats.get_speedup_analysis(dse)



    output_file_minimal = os.path.join(result_dir_specific, file_name+ ".csv")

    base_budget_scaling = sim_dp.database.db_input.sw_hw_database_population["misc_knobs"]["base_budget_scaling"]

    # minimal output
    if os.path.exists(output_file_minimal):
        output_fh_minimal = open(output_file_minimal, "a")
    else:
        output_fh_minimal = open(output_file_minimal, "w")
        for metric in config.all_metrics:
            output_fh_minimal.write(metric + ",")
            if metric in sim_dp.database.db_input.get_budget_dict("glass").keys():
                output_fh_minimal.write(metric+"_budget" + ",")
        output_fh_minimal.write("sampling_mode,")
        output_fh_minimal.write("sampling_reduction" +",")
        for metric, accuracy_percentage in sim_dp.database.hw_sampling["accuracy_percentage"]["ip"].items():
            output_fh_minimal.write(metric+"_accuracy" + ",")  # for now only write the latency accuracy as the other
        for block_type, porting_effort in sim_dp.database.db_input.porting_effort.items():
            output_fh_minimal.write(block_type+"_effort" + ",")  # for now only write the latency accuracy as the other

        output_fh_minimal.write("output_design_status"+ ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("case_study"+ ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("unique_number" + ",")  # for now only write the latency accuracy as the other

        output_fh_minimal.write("SA_total_depth,")
        output_fh_minimal.write("reason_to_terminate" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("population generation cnt" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("iteration cnt" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("workload_set" + ",")  # for now only write the latency accuracy as the other
        #output_fh_minimal.write("iterationxdepth number" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("simulation time" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("move generation time" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("kernel selection time" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("block selection time" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("transformation selection time" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("transformation_selection_mode" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("dist_to_goal_all" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("dist_to_goal_non_cost" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("system block count" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("system PE count" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("system bus count" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("system memory count" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("routing complexity" + ",")  # for now only write the latency accuracy as the other
        #output_fh_minimal.write("area_breakdown_subtype" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("block_impact_sorted" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("kernel_impact_sorted" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("metric_impact_sorted" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("move_metric" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("move_transformation_name" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("move_kernel" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("move_block_name" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("move_block_type" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("move_dir" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("comm_comp" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("high_level_optimization" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("architectural_principle" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("area_dram" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("area_non_dram" + ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("channel_cnt" + ",")  # for now only write the latency accuracy as the other
        for key, val in compute_system_attrs.items():
            output_fh_minimal.write(str(key) + ",")
        for key, val in bus_system_attrs.items():
            output_fh_minimal.write(str(key) + ",")
        for key, val in memory_system_attrs.items():
            output_fh_minimal.write(str(key) + ",")

        for key, val in speedup_attrs.items():
            output_fh_minimal.write(str(key) + ",")

        for key, val in speedup_dict.items():
            output_fh_minimal.write(str(key)+"_speedup_analysis" + ",")

        for key,val in base_budget_scaling.items():
            output_fh_minimal.write("budget_scaling_"+str(key) + ",")



    output_fh_minimal.write("\n")
    for metric in config.all_metrics:
        data_ = sim_dp.dp_stats.get_system_complex_metric(metric)
        if isinstance(data_, dict):
            data__ = convert_dictionary_to_parsable_csv_with_semi_column(data_)
        else:
            data__ = data_

        output_fh_minimal.write(str(data__) + ",")

        if metric in sim_dp.database.db_input.get_budget_dict("glass").keys():
            data_ = sim_dp.database.db_input.get_budget_dict("glass")[metric]
            if isinstance(data_, dict):
                data__ = convert_dictionary_to_parsable_csv_with_semi_column(data_)
            else:
                data__ = data_
            output_fh_minimal.write(str(data__) + ",")

    output_fh_minimal.write(sim_dp.database.hw_sampling["mode"] + ",")
    output_fh_minimal.write(sim_dp.database.hw_sampling["reduction"] + ",")
    for metric, accuracy_percentage in sim_dp.database.hw_sampling["accuracy_percentage"]["ip"].items():
        output_fh_minimal.write(str(accuracy_percentage) + ",")  # for now only write the latency accuracy as the other
    for block_type, porting_effort in sim_dp.database.db_input.porting_effort.items():
        output_fh_minimal.write(str(porting_effort)+ ",")  # for now only write the latency accuracy as the other

    if sim_dp.dp_stats.fits_budget(1):
        output_fh_minimal.write("budget_met"+ ",")  # for now only write the latency accuracy as the other
    else:
        output_fh_minimal.write("budget_not_met" + ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(case_study + ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(str(unique_number)+ ",")  # for now only write the latency accuracy as the other

    output_fh_minimal.write(str(config.SA_depth)+ ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(str(reason_to_terminate)+ ",")  # for now only write the latency accuracy as the other

    ma = sim_dp.get_move_applied() # move applied
    if not ma == None:
        sorted_metrics = convert_tuple_list_to_parsable_csv([(el,val)  for el,val in ma.sorted_metrics.items()])
        metric = ma.get_metric()
        transformation_name = ma.get_transformation_name()
        task_name = ma.get_kernel_ref().get_task_name()
        block_type = ma.get_block_ref().type
        dir = ma.get_dir()
        generation_time = ma.get_generation_time()
        sorted_blocks = convert_tuple_list_to_parsable_csv([(el.get_generic_instance_name(), val) for el,val in ma.sorted_blocks])
        sorted_kernels = convert_tuple_list_to_parsable_csv([(el.get_task_name(), val) for el,val in ma.sorted_kernels.items()])
        blk_instance_name = ma.get_block_ref().get_generic_instance_name()
        blk_type = ma.get_block_ref().type

        comm_comp = (ma.get_system_improvement_log())["comm_comp"]
        high_level_optimization = (ma.get_system_improvement_log())["high_level_optimization"]
        exact_optimization = (ma.get_system_improvement_log())["exact_optimization"]
        architectural_variable_to_improve = (ma.get_system_improvement_log())["architectural_principle"]
        block_selection_time = ma.get_logs("block_selection_time")
        kernel_selection_time = ma.get_logs("kernel_selection_time")
        transformation_selection_time = ma.get_logs("transformation_selection_time")
    else: # happens at the very fist iteration
        sorted_metrics = ""
        metric = ""
        transformation_name = ""
        task_name = ""
        block_type = ""
        dir = ""
        generation_time = ''
        sorted_blocks = ''
        sorted_kernels = {}
        blk_instance_name = ''
        blk_type = ''
        comm_comp = ""
        high_level_optimization = ""
        architectural_variable_to_improve = ""
        block_selection_time = ""
        kernel_selection_time = ""
        transformation_selection_time = ""

    routing_complexity = sim_dp.dp_rep.get_hardware_graph().get_routing_complexity()
    simple_topology = sim_dp.dp_rep.get_hardware_graph().get_simplified_topology_code()
    blk_cnt = sum([int(el) for el in simple_topology.split("_")])
    bus_cnt = [int(el) for el in simple_topology.split("_")][0]
    mem_cnt = [int(el) for el in simple_topology.split("_")][1]
    pe_cnt = [int(el) for el in simple_topology.split("_")][2]
    #itr_depth_multiplied = sim_dp.dp_rep.get_iteration_number()*config.SA_depth + sim_dp.dp_rep.get_depth_number()

    output_fh_minimal.write(str(sim_dp.dp_rep.get_population_generation_cnt())+ ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(str(dse.get_total_iteration_cnt())+ ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write('_'.join(sim_dp.database.db_input.workload_tasks.keys()) +",")
    #output_fh_minimal.write(str(itr_depth_multiplied)+ ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(str(sim_dp.dp_rep.get_simulation_time())+ ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(str(generation_time)+ ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(str(kernel_selection_time)+ ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(str(block_selection_time)+ ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(str(transformation_selection_time)+ ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(str(config.transformation_selection_mode)+ ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(str(sim_dp.dp_stats.dist_to_goal(metrics_to_look_into = ["area", "latency", "power", "cost"], mode = "eliminate")) + ",")
    output_fh_minimal.write(str(sim_dp.dp_stats.dist_to_goal(metrics_to_look_into = ["area", "latency", "power"], mode = "eliminate")) + ",")
    output_fh_minimal.write(str(blk_cnt) + ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(str(pe_cnt) + ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(str(bus_cnt) + ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(str(mem_cnt) + ",") # for now only write the latency accuracy as the other
    output_fh_minimal.write(str(routing_complexity) + ",")  # for now only write the latency accuracy as the other
    #output_fh_minimal.write(convert_dictionary_to_parsable_csv_with_semi_column(sim_dp.dp_stats.SOC_area_subtype_dict.keys()) + ",")
    output_fh_minimal.write(str(sorted_blocks) + ",")
    output_fh_minimal.write(str(sorted_kernels) + ",")
    output_fh_minimal.write(str(sorted_metrics)+  ",")
    output_fh_minimal.write(str(metric)+",")
    output_fh_minimal.write(transformation_name+",")
    output_fh_minimal.write(task_name+",")
    output_fh_minimal.write(blk_instance_name+",")
    output_fh_minimal.write(blk_type+",")
    output_fh_minimal.write(str(dir)+",")
    output_fh_minimal.write(str(comm_comp)+",")
    output_fh_minimal.write(str(high_level_optimization)+",")
    output_fh_minimal.write(str(architectural_variable_to_improve)+",")
    output_fh_minimal.write(str(sim_dp.dp_stats.get_system_complex_area_stacked_dram()["dram"]) +",")
    output_fh_minimal.write(str(sim_dp.dp_stats.get_system_complex_area_stacked_dram()["non_dram"]) +",")
    output_fh_minimal.write(str(sim_dp.dp_rep.get_hardware_graph().get_number_of_channels()) +",")
    for key, val in compute_system_attrs.items():
        output_fh_minimal.write(str(val) + ",")
    for key, val in bus_system_attrs.items():
        output_fh_minimal.write(str(val) + ",")
    for key, val in memory_system_attrs.items():
        output_fh_minimal.write(str(val) + ",")

    for key, val in speedup_attrs.items():
        output_fh_minimal.write(str(val) + ",")

    for key, val in speedup_dict.items():
        output_fh_minimal.write(convert_dictionary_to_parsable_csv_with_semi_column(val)+",")

    for key,val in base_budget_scaling.items():
        output_fh_minimal.write(str(val) + ",")

    output_fh_minimal.close()



def simple_run_iterative(result_folder, sw_hw_database_population, system_workers=(1, 1)):
    case_study = "simple_run_iterative"
    current_process_id = system_workers[0]
    total_process_cnt = system_workers[1]
    starting_exploration_mode = "from_scratch"
    print('cast study:' + case_study)
    # -------------------------------------------
    # set parameters
    # -------------------------------------------
    experiment_repetition_cnt = 1
    reduction = "most_likely"

    # -------------------------------------------
    #  distribute the work
    # -------------------------------------------
    work_per_process = math.ceil(experiment_repetition_cnt / total_process_cnt)
    run_ctr = 0

    # -------------------------------------------
    # run the combination and collect the data
    # -------------------------------------------
    # get the budget, set them and run FARSI
    for i in range(0, work_per_process):
        # -------------------------------------------
        # collect the exact hw sampling
        # -------------------------------------------
        accuracy_percentage = {}
        accuracy_percentage["sram"] = accuracy_percentage["dram"] = accuracy_percentage["ic"] = accuracy_percentage[
            "gpp"] = accuracy_percentage["ip"] = \
            {"latency": 1,
             "energy": 1,
             "area": 1,
             "one_over_area": 1}
        hw_sampling = {"mode": "exact", "population_size": 1, "reduction": reduction,
                       "accuracy_percentage": accuracy_percentage}

        unique_suffix = str(total_process_cnt) + "_" + str(current_process_id) + "_" + str(run_ctr)

        #study = ["boundedness", "serial_parallel"]
        study = "boundedness"
        #study = "serial_parallel"
        #study = "hop_NoC_studies"

        if study == "serial_parallel":
            # for serial/parallel studies
            #serial_sweep = list(range(0,20,8))
            #parallel_sweep = list(range(0,20, 8))
            #parallel_sweep = [0, 1,2, 4,8]
            #serial_sweep = [0, 1, 2, 4,8]


            parallel_sweep = [0,4,8] #, 4,8]
            serial_sweep = [0, 4,8]
            #parallel_sweep =[5]

            all_sweep = [serial_sweep, parallel_sweep]
            parallel_task_type_list= ["audio_style", "edge_detection_style"] # "audio_style, edge_detection_style
            parallel_task_type_list= ["audio_style"] #, "edge_detection_style"] # "audio_style, edge_detection_style
            #parallel_task_type_list= ["edge_detection"] #, "edge_detection_style"] # "audio_style, edge_detection_style
            for parallel_task_type in parallel_task_type_list:
                combos = itertools.product(*all_sweep)
                for serial_task_cnt, parallel_task_cnt in combos:
                    datamovement_scaling_ratio = 1
                    sw_hw_database_population["misc_knobs"]['task_spawn']['serial_task_cnt'] = serial_task_cnt
                    sw_hw_database_population["misc_knobs"]['task_spawn']['parallel_task_cnt'] = parallel_task_cnt
                    sw_hw_database_population["misc_knobs"]['task_spawn']['parallel_task_type'] = parallel_task_type  # can be audio or edge detection
                    sw_hw_database_population["misc_knobs"]['task_spawn']['boundedness'] = ["memory_intensive", datamovement_scaling_ratio, 1]
                    sw_hw_database_population["misc_knobs"]['num_of_hops'] = 4
                    sw_hw_database_population["misc_knobs"]['num_of_NoCs'] = 4
                    #sw_hw_database_population["hw_graph_mode"] = "star_mode"
                    sw_hw_database_population["hw_graph_mode"] = "hop_mode"
                    db_input = database_input_class(sw_hw_database_population)
                    print("hw_sampling:" + str(hw_sampling))
                    print("budget set to:" + str(db_input.get_budget_dict("glass")))
                    dse_hndlr = run_FARSI_only_simulation(result_folder, unique_suffix, db_input, hw_sampling,
                                          sw_hw_database_population["hw_graph_mode"])
        elif study == "boundedness":
            # for boundedness studies
            boundedness_ratio_range = list(np.arange(0,1.1,.1))
            #boundedness_ratio_range = [0, .3,.6,1]
            boundedness_ratio_range = [.3]
            #datamovement_scaling_ratio_range = list(np.arange(.1,1,.2))
            datamovement_scaling_ratio_range = [1]
            for boundedness_ratio in boundedness_ratio_range:
                for datamovement_scaling_ratio in datamovement_scaling_ratio_range:
                    sw_hw_database_population["misc_knobs"]['task_spawn']['serial_task_cnt'] = 4 #6
                    sw_hw_database_population["misc_knobs"]['task_spawn']['parallel_task_cnt'] = 0 #1
                    sw_hw_database_population["misc_knobs"]['task_spawn']['parallel_task_type'] = "audio_style" # can be audio or edge detection
                    sw_hw_database_population["misc_knobs"]['task_spawn']['boundedness'] = ["memory_intensive", datamovement_scaling_ratio, boundedness_ratio]
                    sw_hw_database_population["misc_knobs"]['num_of_hops'] = 1
                    sw_hw_database_population["misc_knobs"]['num_of_NoCs'] = 1
                    db_input = database_input_class(sw_hw_database_population)
                    print("hw_sampling:" + str(hw_sampling))
                    print("budget set to:" + str(db_input.get_budget_dict("glass")))
                    dse_hndlr = run_FARSI_only_simulation(result_folder, unique_suffix, db_input, hw_sampling,
                                          sw_hw_database_population["hw_graph_mode"])
        elif study == "hop_studies":
            # for boundedness studies
            boundedness_ratio_range = [1]
            datamovement_scaling_ratio_range = [1]
            datamovement_scaling_ratio = 1
            #num_of_hops_range = [1]
            num_of_hops_range = [4]
            boundedness_ratio = 1

            #serial_sweep = [0]
            #parallel_sweep = [0,2, 4, 8] #, 16]

            serial_sweep = [0]
            parallel_sweep = [5,7, 9] #, 16]

            all_sweep = [serial_sweep, parallel_sweep]
            parallel_task_type_list= ["audio_style"] #, "edge_detection_style"] # "audio_style, edge_detection_style

            for num_of_hops in num_of_hops_range:
                combos = itertools.product(*all_sweep)
                for serial_task_cnt, parallel_task_cnt in combos:
                    sw_hw_database_population["misc_knobs"]['task_spawn']['serial_task_cnt'] = serial_task_cnt
                    sw_hw_database_population["misc_knobs"]['task_spawn']['parallel_task_cnt'] = parallel_task_cnt
                    sw_hw_database_population["misc_knobs"]['task_spawn']['parallel_task_type'] = "audio_style" # can be audio or edge detection
                    sw_hw_database_population["misc_knobs"]['task_spawn']['boundedness'] = ["memory_intensive", datamovement_scaling_ratio, boundedness_ratio]
                    sw_hw_database_population["misc_knobs"]['num_of_hops'] = num_of_hops
                    sw_hw_database_population["misc_knobs"]['num_of_NoCs'] = num_of_hops  # same as num_of_hops
                    sw_hw_database_population["hw_graph_mode"] = "hop_mode"
                    db_input = database_input_class(sw_hw_database_population)
                    print("hw_sampling:" + str(hw_sampling))
                    print("budget set to:" + str(db_input.get_budget_dict("glass")))
                    dse_hndlr = run_FARSI_only_simulation(result_folder, unique_suffix, db_input, hw_sampling,
                                          sw_hw_database_population["hw_graph_mode"])
        elif study == "hop_NoC_studies":
            # for boundedness studies
            boundedness_ratio_range = [1]
            datamovement_scaling_ratio_range = [1]
            datamovement_scaling_ratio = 1
            #num_of_hops_range = [1]
            #num_of_hop_NoCs_range =[[1,1], [2,2],[3,3], [2,3], [2,4],[3,4]]
            #num_of_hop_NoCs_range =[[3,4]]
            #num_of_hop_NoCs_range =[[1,1], [2,2], [3,3]]
            num_of_hop_NoCs_range =[[1,1],[2,2],[2,3], [2,4], [3,3], [3,4], [4,4]]
            boundedness_ratio = 1

            #serial_sweep = [0]
            #parallel_sweep = [0,2, 4, 8] #, 16]

            serial_sweep = [4]
            parallel_sweep = [0] #, 16]

            all_sweep = [serial_sweep, parallel_sweep]
            parallel_task_type_list= ["audio_style"] #, "edge_detection_style"] # "audio_style, edge_detection_style

            for num_of_hops, num_of_NoCs in num_of_hop_NoCs_range:
                combos = itertools.product(*all_sweep)
                for serial_task_cnt, parallel_task_cnt in combos:
                    sw_hw_database_population["misc_knobs"]['task_spawn']['serial_task_cnt'] = serial_task_cnt
                    sw_hw_database_population["misc_knobs"]['task_spawn']['parallel_task_cnt'] = parallel_task_cnt
                    sw_hw_database_population["misc_knobs"]['task_spawn']['parallel_task_type'] = "audio_style" # can be audio or edge detection
                    sw_hw_database_population["misc_knobs"]['task_spawn']['boundedness'] = ["memory_intensive", datamovement_scaling_ratio, boundedness_ratio]
                    sw_hw_database_population["misc_knobs"]['num_of_hops'] = num_of_hops
                    sw_hw_database_population["misc_knobs"]['num_of_NoCs'] = num_of_NoCs
                    sw_hw_database_population["hw_graph_mode"] = "hop_mode"
                    db_input = database_input_class(sw_hw_database_population)
                    print("hw_sampling:" + str(hw_sampling))
                    print("budget set to:" + str(db_input.get_budget_dict("glass")))
                    dse_hndlr = run_FARSI_only_simulation(result_folder, unique_suffix, db_input, hw_sampling,
                                          sw_hw_database_population["hw_graph_mode"])


        run_ctr += 1
        # write the results in the general folder
        result_dir_specific = os.path.join(result_folder, "result_summary")
        write_one_results(dse_hndlr.dse.so_far_best_sim_dp,  dse_hndlr.dse, dse_hndlr.dse.reason_to_terminate, case_study,
                          result_dir_specific, unique_suffix,
                          config.FARSI_simple_run_prefix + "_" + str(current_process_id) + "_" + str(total_process_cnt))
        dse_hndlr.dse.write_data_log(list(dse_hndlr.dse.get_log_data()), dse_hndlr.dse.reason_to_terminate, case_study, result_dir_specific, unique_suffix,
                      config.FARSI_simple_run_prefix + "_" + str(current_process_id) + "_" + str(total_process_cnt))

        # write the results in the specific folder
        result_folder_modified = result_folder+ "/runs/" + str(ctr) + "/"
        os.system("mkdir -p " + result_folder_modified)
        copy_DSE_data(result_folder_modified)
        write_one_results(dse_hndlr.dse.so_far_best_sim_dp, dse_hndlr.dse, dse_hndlr.dse.reason_to_terminate, case_study, result_folder_modified, unique_suffix,
                      config.FARSI_simple_run_prefix + "_" + str(current_process_id) + "_" + str(total_process_cnt))

        os.system("cp " + config.home_dir+"/settings/config.py"+ " "+ result_folder)

# ------------------------------
# Functionality:
#  a simple run, where FARSI is run to meet certain budget
# Variables:
#      system_workers:  used for parallelizing the data collection: (current process id, total number workers)
# ------------------------------
def simple_run(result_folder, sw_hw_database_population, system_workers=(1, 1)):
    case_study = "simple_run"
    current_process_id = system_workers[0]
    total_process_cnt = system_workers[1]
    starting_exploration_mode = "from_scratch"
    print('cast study:' + case_study)
    # -------------------------------------------
    # set parameters
    # -------------------------------------------
    experiment_repetition_cnt = 1
    reduction = "most_likely"

    # -------------------------------------------
    #  distribute the work
    # -------------------------------------------
    work_per_process = math.ceil(experiment_repetition_cnt / total_process_cnt)
    run_ctr = 0

    # -------------------------------------------
    # run the combination and collect the data
    # -------------------------------------------
    # get the budget, set them and run FARSI
    for i in range(0, work_per_process):
        # -------------------------------------------
        # collect the exact hw sampling
        # -------------------------------------------
        accuracy_percentage = {}
        accuracy_percentage["sram"] = accuracy_percentage["dram"] = accuracy_percentage["ic"] = accuracy_percentage[
            "gpp"] = accuracy_percentage["ip"] = \
            {"latency": 1,
             "energy": 1,
             "area": 1,
             "one_over_area": 1}
        hw_sampling = {"mode": "exact", "population_size": 1, "reduction": reduction,
                       "accuracy_percentage": accuracy_percentage}
        db_input = database_input_class(sw_hw_database_population)
        print("hw_sampling:" + str(hw_sampling))
        print("budget set to:" + str(db_input.get_budget_dict("glass")))
        unique_suffix = str(total_process_cnt) + "_" + str(current_process_id) + "_" + str(run_ctr)


        # run FARSI
        dse_hndlr = run_FARSI(result_folder, unique_suffix, case_study, db_input, hw_sampling,
                              sw_hw_database_population["hw_graph_mode"])
        #dse_hndlr = run_FARSI_only_simulation(result_folder, unique_suffix, db_input, hw_sampling,
        #                      sw_hw_database_population["hw_graph_mode"])


        run_ctr += 1

        return dse_hndlr



# ------------------------------
# Functionality:
#     generate a range of values between a lower and upper bound
# Variables:
#      cnt: number of points within the range to generate
# ------------------------------
def gen_range(lower_bound, upper_bound, cnt):
    if cnt == 1:
        upper_bound = lower_bound
        step_size = lower_bound
    else:
        step_size = (upper_bound  - lower_bound) / cnt

    range_= list(np.arange(lower_bound, min(.9*lower_bound, (10**-9)*upper_bound) + upper_bound, step_size))
    range_formatted = [float("{:.9f}".format(el)) for el in range_]
    if len(range_formatted) == 1:
        return range_formatted
    else:
        return range_formatted[:-1]


# ------------------------------
# Functionality:
#     generate all the combinations of inputs
# Variables:
#      args: consist list of arg values. Each arg specfies a range of values using (lower bound, upper bound, cnt)
# ------------------------------
def gen_combinations(args):
    list_of_ranges = []
    for arg in args:
        list_of_ranges.append(gen_range(arg[0], arg[1], arg[2]))
    all_combinations = [*list_of_ranges]
    all_budget_combinations = itertools.product(*list_of_ranges)
    all_budget_combinations_listified = [el for el in all_budget_combinations]
    return all_budget_combinations_listified


# ------------------------------
# Functionality:
#  conduct a host of studies depending on whether input_error or input_cost are set. Here are the combinations:
#  input_error = False, input_cost = False: conduct a cost-PPA study
#  input_error = True , input_cost = False: conduct a study to figure out the impact of input error on cost
#  input_error = True , input_cost = True : TBD
# Variables:
#      system_workers:  used for parallelizing the data collection: (current process id, total number workers)
# ------------------------------
def input_error_output_cost_sensitivity_study(result_folder, sw_hw_database_population, system_workers=(1,1), input_error=False, input_cost=False):
    current_process_id = system_workers[0]
    total_process_cnt = system_workers[1]

    # -----------------------
    # set up the case study
    # -----------------------
    case_study = ""
    if not input_error and not input_cost:
        case_study = "cost_PPA"
        file_prefix = config.FARSI_cost_correlation_study_prefix
    elif input_error and not input_cost:
        case_study = "input_error_output_cost"
        print("input error cost study")
        file_prefix = config.FARSI_input_error_output_cost_sensitivity_study_prefix
    elif input_error and input_cost:
        case_study = "input_error_input_cost"
        file_prefix = config.FARSI_input_error_input_cost_sensitivity_study_prefix
    else:
        print("this study is not supported")
        exit(0)

    print("conducting " + case_study)

    # -----------------------
    # first extract the current budget
    # -----------------------
    accuracy_percentage = {}
    accuracy_percentage["sram"] = accuracy_percentage["dram"] = accuracy_percentage["ic"] = accuracy_percentage["gpp"] = \
    accuracy_percentage["ip"] = \
        {"latency": 1,
         "energy": 1,
         "area": 1,
         "one_over_area": 1}
    hw_sampling = {"mode": "exact", "population_size": 1, "reduction": "most_likey",
                   "accuracy_percentage": accuracy_percentage}
    db_input = database_input_class(sw_hw_database_population)
    budgets_dict = {}  # set the reference budget
    budgets_dict['latency'] = db_input.budgets_dict['glass']['latency'][list(workloads)[0]]
    budgets_dict['power'] = db_input.budgets_dict['glass']['power']
    budgets_dict['area'] = db_input.budgets_dict['glass']['area']

    #-------------------------------------------
    # set sweeping parameters
    #-------------------------------------------
    experiment_repetition_cnt = 1
    budget_cnt = 3
    budget_upper_bound_factor = {}
    budget_upper_bound_factor["perf"] = 10
    budget_upper_bound_factor["power"] = 10
    budget_upper_bound_factor["area"] = 100

    if not input_error:
        accuracy_lower_bound = accuracy_upper_bound = 1
        accuracy_cnt = 1  # number of accuracy values to use
    else:
        accuracy_lower_bound = .5
        accuracy_cnt = 3  # number of accuracy values to use
        accuracy_upper_bound = 1

    if not input_cost:
        effort_lower_bound = effort_upper_bound = 100
        effort_cnt = 1  # number of accuracy values to use
    else:
        effort_lower_bound = 20
        effort_cnt = 3  # number of accuracy values to use
        effort_upper_bound = 100

    # -------------------------------------------
    # generate all the combinations of the budgets
    # -------------------------------------------
    combination_input =[]
    combination_input.append((budgets_dict["latency"], budget_upper_bound_factor["perf"]*budgets_dict["latency"], budget_cnt))
    combination_input.append((budgets_dict["power"], budget_upper_bound_factor["power"]*budgets_dict["power"], budget_cnt))
    combination_input.append((budgets_dict["area"], budget_upper_bound_factor["area"]*budgets_dict["area"], budget_cnt))

    combination_input.append((accuracy_lower_bound, accuracy_upper_bound, accuracy_cnt))
    combination_input.append((effort_lower_bound, effort_upper_bound, effort_cnt))

    all_combinations = gen_combinations(combination_input)

    #-------------------------------------------
    #  distribute the work
    #-------------------------------------------
    combo_cnt = len(list(all_combinations))
    work_per_process = math.ceil(combo_cnt/total_process_cnt)
    run_ctr = 0

    #-------------------------------------------
    # run the combination and collect the data
    #-------------------------------------------
    # get the budget, set them and run FARSI
    reduction = "most_likely_with_accuracy_percentage"
    for i in range(0, experiment_repetition_cnt):
        for latency, power, area, accuracy, effort in list(all_combinations)[current_process_id* work_per_process: min((current_process_id+ 1) * work_per_process, combo_cnt)]:
            # iterate though metrics and set the budget

            accuracy_percentage = {}
            accuracy_percentage["sram"] = accuracy_percentage["dram"] = accuracy_percentage["ic"] = accuracy_percentage["gpp"] = {"latency": 1, "energy": 1,
                                                                                                   "area": 1, "one_over_area": 1}
            accuracy_percentage["ip"] = {"latency": accuracy, "energy": 1 / pow(accuracy, 2), "area": 1,
                                                   "one_over_area": 1}
            hw_sampling = {"mode": "exact", "population_size": 1, "reduction": reduction,
                           "accuracy_percentage": accuracy_percentage}
            db_input = database_input_class(sw_hw_database_population)

            # set the budget
            budgets_dict = {}
            budgets_dict["glass"] = {}
            budgets_dict["glass"]["latency"] = {list(workloads)[0]:latency}
            budgets_dict["glass"]["power"] = power
            budgets_dict["glass"]["area"] = area
            db_input.set_budgets_dict_directly(budgets_dict)
            db_input.set_porting_effort_for_block("ip", effort) # only playing with ip now
            unique_suffix = str(total_process_cnt) + "_" + str(current_process_id) + "_" + str(run_ctr)

            print("hw_sampling:" + str(hw_sampling))
            print("budget set to:" + str(db_input.budgets_dict))
            dse_hndlr = run_FARSI(result_folder, unique_suffix, db_input, hw_sampling, sw_hw_database_population["hw_graph_mode"])
            run_ctr += 1
            # write the results in the general folder
            result_dir_specific = os.path.join(result_folder, "result_summary")
            write_one_results(dse_hndlr.dse.so_far_best_sim_dp, dse_hndlr.dse, dse_hndlr.dse.reason_to_terminate, case_study, result_dir_specific, unique_suffix,
                          file_prefix + "_" + str(current_process_id) + "_" + str(total_process_cnt))

            # write the results in the specific folder
            result_folder_modified = result_folder + "/runs/" + str(run_ctr) + "/"
            os.system("mkdir -p " + result_folder_modified)
            copy_DSE_data(result_folder_modified)
            write_one_results(dse_hndlr.dse.so_far_best_sim_dp, dse_hndlr.dse, dse_hndlr.dse.reason_to_terminate, case_study, result_folder_modified, unique_suffix,
                          file_prefix + "_" + str(current_process_id) + "_" + str(total_process_cnt))


if __name__ == "__main__":
    # set the number of workers to be used (parallelism applied)
    current_process_id = 0
    total_process_cnt = 1
    system_workers = (current_process_id, total_process_cnt)


    # set the study type
    #study_type = "cost_PPA"
    study_type = "simple_run"
    #study_subtype = "plot_3d_distance"
    study_subtype = "run"
    assert study_type in ["cost_PPA", "simple_run", "input_error_output_cost_sensitivity", "input_error_input_cost_sensitivity"]
    assert study_subtype in ["run", "plot_3d_distance"]


    # set the study parameters
    # set the workload

    #workloads = {"edge_detection"}
    #workloads = {"hpvm_cava"}
    #workloads = {"audio_decoder"}
    #workloads ={"edge_detection","hpvm_cava", "audio_decoder"}
    workloads = {"edge_detection", "audio_decoder"}
    workloads = {"SLAM"}
    workloads = {"simple_multiple_hops"}
    workloads = {"simple_serial_tg"}
    #workloads = {"partial_SOC_example_hard"}
    #workloads = {"simple_all_parallel"}

    workloads_first_letter  = '_'.join(sorted([el[0] for el in workloads]))
    # set result folder
    result_home_dir_default = os.path.join(os.getcwd(), "data_collection/data/" + study_type)
    result_home_dir = os.path.join(config.home_dir, "data_collection/data/" + study_type)
    date_time = datetime.now().strftime('%m-%d_%H-%M_%S')
    #config_obj = config
    #config_obj_budg_dict = config_obj.budget_dict
    budget_values = "pow_"+str(config.budget_dict["glass"]["power"]) + "__area_"+str(config.budget_dict["glass"]["area"])
    result_folder = os.path.join(result_home_dir,
                                 date_time + "____"+ budget_values+"___workloads_"+workloads_first_letter)


    # set the IP spawning params
    ip_loop_unrolling = {"incr": 2, "max_spawn_ip": 17, "spawn_mode": "geometric"}
    #ip_freq_range = {"incr":3, "upper_bound":8}
    #mem_freq_range = {"incr":3, "upper_bound":6}
    #ic_freq_range = {"incr":4, "upper_bound":6}
    ip_freq_range = [1,4,6,8]
    mem_freq_range = [1,4,6]
    ic_freq_range = [1,4,6]
    #tech_node_SF = {"perf":1, "energy":.064, "area":.079}   # technology node scaling factor
    tech_node_SF = {"perf":1, "energy":{"non_gpp":.064, "gpp":1}, "area":{"non_mem":.0374 , "mem":.07, "gpp":1}}   # technology node scaling factor
    db_population_misc_knobs = {"ip_freq_correction_ratio": 1, "gpp_freq_correction_ratio": 1,
                                "ip_spawn": {"ip_loop_unrolling": ip_loop_unrolling, "ip_freq_range": ip_freq_range},
                                "mem_spawn": {"mem_freq_range":mem_freq_range},
                                "ic_spawn": {"ic_freq_range":ic_freq_range},
                                "tech_node_SF":tech_node_SF,
                                "base_budget_scaling":{"latency":.5, "power":1, "area":1}}

    # set software hardware database population
    # for SLAM, and serial experiment
    sw_hw_database_population = {"db_mode": "hardcoded", "hw_graph_mode": "generated_from_scratch",
                                 "workloads": workloads, "misc_knobs": db_population_misc_knobs}

    #for multiple hops experiment
    #sw_hw_database_population = {"db_mode": "hardcoded", "hw_graph_mode": "hardcoded",
    #                             "workloads": workloads, "misc_knobs": db_population_misc_knobs}
    # for paper workloads
    #sw_hw_database_population = {"db_mode": "parse", "hw_graph_mode": "generated_from_scratch",
    #                             "workloads": workloads, "misc_knobs": db_population_misc_knobs}
    # for check pointed
    #sw_hw_database_population = {"db_mode": "parse", "hw_graph_mode": "generated_from_check_point",
    #                             "workloads": workloads, "misc_knobs": db_population_misc_knobs}


    # depending on the study/substudy type, invoke the appropriate function
    if study_type == "simple_run":
        simple_run(result_folder, sw_hw_database_population, system_workers)
    elif study_type == "cost_PPA" and study_subtype == "run":
        input_error_output_cost_sensitivity_study(result_folder, sw_hw_database_population, system_workers, False, False)
    elif study_type == "input_error_output_cost_sensitivity" and study_subtype == "run":
        input_error_output_cost_sensitivity_study(result_folder, sw_hw_database_population, system_workers,  True, False)
    elif study_type == "input_error_input_cost_sensitivity" and study_subtype == "run":
        input_error_output_cost_sensitivity_study(result_folder, sw_hw_database_population, system_workers,True, True)
    elif study_type == "cost_PPA" and study_subtype == "plot_3d_distance":
        result_folder = "05-28_18-46_40"  # edge detection 
        result_folder = "05-28_18-47_33" # hpvm cava
        result_folder = "05-28_18-47_03"
        result_folder = "05-31_16-24_49" # hpvm cava (2, tighter constraints)
        result_dir_addr= os.path.join(config.home_dir, 'data_collection/data/', study_type, result_folder,
                                                   "result_summary")
        full_file_addr = os.path.join(result_dir_addr,
                                      config.FARSI_cost_correlation_study_prefix + "_0_1.csv")
        plot_3d_dist(result_dir_addr, full_file_addr, workloads)