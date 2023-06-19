#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import csv
import os
from collections import defaultdict
from design_utils.components.hardware import *
from SIM_utils.SIM import *
from specs.LW_cl import *
import pandas as pd
import itertools

# ------------------------------
# Functionality:
#   parse the csv file and return a dictionary containing the hardware graph
# ------------------------------
def parse_hardware_graph(hardware_graph_file):
    if not os.path.exists(hardware_graph_file):
        return ""

    reader = csv.DictReader(open(hardware_graph_file, 'r'))
    dict_list = []
    for line in reader:
        dict_list.append(line)

    # generate the task graph dictionary, a 2d dictionary where
    # the first coordinate is the parent and the second coordinate is the child
    hardware_graph_dict = defaultdict(dict)
    table_name = "Block Name"
    for dict_ in dict_list:
        block_name = [dict_[key] for key in dict_.keys() if key == table_name][0]
        hardware_graph_dict[block_name] = {}
        for child_block_name, data_movement in dict_.items():
            if child_block_name == table_name: # skip the table_name entry
                continue
            elif child_block_name == block_name:  # for now skip the self movement (Requires scratch pad)
                continue
            elif data_movement == "":
                continue
            else:
                hardware_graph_dict[block_name][child_block_name] = float(data_movement)

    return hardware_graph_dict


# ------------------------------
# Functionality:
#   parse the csv file and return a dictionary containing the task to hardware mapping
# ------------------------------
def parse_task_to_hw_mapping(task_to_hw_mapping_file):
    if not os.path.exists(task_to_hw_mapping_file):
        return ""
    reader = csv.DictReader(open(task_to_hw_mapping_file, 'r'))
    dict_list = []
    for line in reader:
        dict_list.append(line)

    # generate the task graph dictionary, a 2d dictionary where
    # the first coordinate is the parent and the second coordinate is the child
    hardware_graph_dict = defaultdict(dict)
    table_name = "Block Name"
    block_names = []
    for dict_ in dict_list:
        block_names.extend([key for key in dict_.keys()])
    block_names = list(set(block_names))

    block_mapping = {}
    for dict_ in dict_list:
        for blk, task in dict_.items():
            if task == "":
                continue
            elif ("->") in task:
                task = [el.strip() for el in task.split("->")]   # split the producer consumer and get rid of extra spaces
            else:
                task = [task, task]

            if blk not in block_mapping.keys():
                block_mapping[blk] = [task]
            else:
                block_mapping[blk].append(task)

    return block_mapping

# ------------------------------
# Functionality:
#   parse the csv file and return a dictionary containing the task graph
# ------------------------------
def parse_task_graph_data_movement(task_graph_file_addr):
    reader = csv.DictReader(open(task_graph_file_addr, 'r'))
    dict_list = []
    for line in reader:
        dict_list.append(line)

    # generate the task graph dictionary, a 2d dictionary where
    # the first coordinate is the parent and the second coordinate is the child
    task_graph_data_movement_dict = defaultdict(dict)
    table_name = "Task Name"
    for dict_ in dict_list:
        task_name = [dict_[key] for key in dict_.keys() if key == table_name][0]
        task_graph_data_movement_dict[task_name] = {}
        for child_task_name, data_movement in dict_.items():
            if child_task_name == table_name: # skip the table_name entry
                continue
            elif child_task_name == task_name:  # for now skip the self movement (Requires scratch pad)
                continue
            elif data_movement == "":
                continue
            else:
                task_graph_data_movement_dict[task_name][child_task_name] = float(data_movement)

    return task_graph_data_movement_dict


# ------------------------------
# Functionality:
#   file finding helper
# ------------------------------
def get_full_file_name(partial_name, file_list):
    for file_name in file_list:
        if partial_name == file_name:
            return file_name
    print("file with the name of :" + partial_name + " doesnt exist")


def get_block_clock_freq(library_dir, input_file_name):
    input_file_addr = os.path.join(library_dir, input_file_name)

    df = pd.read_csv(input_file_addr)

    # eval the expression
    def evaluate(value):
        replaced_value_1 = value.replace("^", "**")
        replaced_value_2 = replaced_value_1.replace("=", "")
        return eval(replaced_value_2)

    misc_data = {}
    for index, row in df.iterrows():
        temp_dict = row.to_dict()
        misc_data[list(temp_dict.values())[0]] = evaluate(list(temp_dict.values())[1])

    hardware_sub_type = ["sram", "dram", "ip", "gpp", "ic"]
    block_sub_type_clock_freq = {}
    for key in misc_data.keys() :
       if "clock" in key:
           for type in hardware_sub_type:
               if type in key:
                   block_sub_type_clock_freq[type] = misc_data[key]

    return block_sub_type_clock_freq

# ------------------------------
# Functionality:
#   parse the task graph csv and generate FARSI digestible task graph
# ------------------------------
def gen_task_graph(library_dir, prefix, misc_knobs):
    tasksL: List[TaskL] = []

    # get files
    file_list = [f for f in os.listdir(library_dir) if os.path.isfile(os.path.join(library_dir, f))]
    data_movement_file_name = get_full_file_name(prefix + "Task Data Movement.csv", file_list)
    IP_perf_file_name = get_full_file_name(prefix + "Task PE Performance.csv", file_list)
    Block_char_file_name = get_full_file_name("misc_database - "+ "Block Characteristics.csv", file_list)

    # collect data movement data
    data_movement_file_addr = os.path.join(library_dir, data_movement_file_name)
    task_graph_dict = parse_task_graph_data_movement(data_movement_file_addr)

    # collect number of instructions for each tasks
    work_dict = gen_task_graph_work_dict(library_dir, IP_perf_file_name, Block_char_file_name, misc_knobs)
    """ 
    for task_name, work in work_dict.items():
        print(task_name+","+str(work))
    exit(0)
    """

    if "burst_size_options" in misc_knobs:
        universal_burst_size = misc_knobs["burst_size_options"][0] # this will be tuned by the DSE
    else:
        universal_burst_size = config.default_burst_size

    for task_name_, values in task_graph_dict.items():
        task_ = TaskL(task_name=task_name_, work=work_dict[task_name_])
        task_.set_burst_size(universal_burst_size)
        task_.add_task_work_distribution([(work_dict[task_name_], 1)])
        tasksL.append(task_)

    for task_name_, values in task_graph_dict.items():
        task_ = [taskL for taskL in tasksL if taskL.task_name == task_name_][0]
        for child_task_name, data_movement in values.items():
            child_task = [taskL for taskL in tasksL if taskL.task_name == child_task_name][0]
            task_.add_child(child_task, data_movement, "real")  # eye_tracking_soource t glint_mapping
            task_.add_task_to_child_work_distribution(child_task, [(data_movement, 1)])  # eye_tracking_soource t glint_mapping

    return tasksL,task_graph_dict


# ------------------------------
# Functionality:
#   get teh reference gpp (general purpose processor) data (properties)
# ------------------------------
def get_ref_gpp_data(dict_list):
    for dict_ in dict_list:
        # get all the properties
        speedup = [float(value) for key, value in dict_.items() if key == "speed up"][0]
        if speedup == 1:
            return dict_
    print("couldn't find the reference gpp")
    exit(0)


# ------------------------------
# Functionality:
#   get number of iterations for a task
# ------------------------------
def parse_task_itr_cnt(library_dir, task_itr_cnt_file_name):
    if task_itr_cnt_file_name == None :
        print(" Task Itr Cnt for the workload is not provided")
        return {}
    elif not os.path.exists(os.path.join(library_dir, task_itr_cnt_file_name)):
        print(" Task Itr Cnt for the workload is not provided")
        return {}

    reader = csv.DictReader(open(os.path.join(library_dir, task_itr_cnt_file_name), 'r'))
    dict_list = []
    task_itr_cnt = {}
    for line in reader:
        dict_list.append(line)

    task_metric_dict = {}
    for dict_ in dict_list:
        task_name = [dict_[key] for key in dict_.keys() if key == "Task Name"][0]
        itr_cnt = int([dict_[key] for key in dict_.keys() if key == "number of iterations"][0])
        task_itr_cnt[task_name] = itr_cnt

    return task_itr_cnt

# ------------------------------
# Functionality:
#   get each tasks PPA (perf, pow, area) data
# ------------------------------
def parse_task_PPA(library_dir, IP_file_name, gpp_names):
    reader = csv.DictReader(open(os.path.join(library_dir, IP_file_name), 'r'))
    dict_list = []
    for line in reader:
        dict_list.append(line)

    task_metric_dict = {}
    for dict_ in dict_list:
        task_name = [dict_[key] for key in dict_.keys() if key == "Task Name"][0]
        task_metric_dict[task_name] = {}
        for key, value in dict_.items():
            if key == "Task Name" or value == "":
                continue
            if key not in gpp_names:
                ip_name_modified = task_name + "_" + key
            else:
                ip_name_modified = key

            task_metric_dict[task_name][ip_name_modified] = float(value)
    return task_metric_dict


# ------------------------------
# Functionality:
#   get the performance of the task on the gpp
# ------------------------------
def parse_task_perf_on_ref_gpp(library_dir, IP_perf_file_name, Block_char_file_name):
    reader = csv.DictReader(open(os.path.join(library_dir, IP_perf_file_name), 'r'))
    dict_list = []
    for line in reader:
        dict_list.append(line)

    ref_gpp_dict = parse_ref_block_values(library_dir, Block_char_file_name, ("pe", "gpp"))

    task_perf_on_ref_dict = defaultdict(dict)
    for dict_ in dict_list:
        task_name = [dict_[key] for key in dict_.keys() if key == "Task Name"][0]
        task_perf_on_ref_dict[task_name] = {}
        time = [dict_[key] for key in dict_.keys() if ref_gpp_dict['Name'] in key][0]
        task_perf_on_ref_dict[task_name] = float(time)
    return task_perf_on_ref_dict


# ------------------------------
# Functionality:
#   parse and find the hardware blocks of a certain type
# ------------------------------
def parse_block_based_on_types(library_dir, Block_char_file_name, type_sub_type):
    type = type_sub_type[0]
    sub_type = type_sub_type[1]
    ref_gpp_dict = {}
    reader = csv.DictReader(open(os.path.join(library_dir, Block_char_file_name), 'r'))
    blck_dict_list = []
    for line in reader:
        blck_dict_list.append(line)

    blck_dict = {}
    for dict_ in blck_dict_list:
        if dict_['Type'] == type and dict_['Subtype'] == sub_type:
            blck_dict[dict_['Name']] = {}
            for key, value in dict_.items():
                if value.isdigit():
                    blck_dict[dict_['Name']][key] = float(value)
                else:
                    blck_dict[dict_['Name']][key] = value
    return blck_dict


# ------------------------------
# Functionality:
#   parse reference block blocks values
# ------------------------------
def parse_ref_block_values(library_dir, Block_char_file_name, type_sub_type):
    blck_dict = parse_block_based_on_types(library_dir, Block_char_file_name, type_sub_type)
    for blck_name, blck_dict_ in blck_dict.items():
        for key_, value_ in blck_dict_.items():
            if key_ == "Ref" and value_ == "yes":
                return blck_dict_

    print("need to at least have one ref gpp")
    exit(0)


# ------------------------------
# Functionality:
#   generate task graph and populate it with work values
# ------------------------------
def gen_task_graph_work_dict(library_dir, IP_perf_file_name, Block_char_file_name, misc_knobs):
    #correction_values = gen_correction_values(workmisc_knobs)

    gpp_file_addr = os.path.join(library_dir, Block_char_file_name)
    IP_perf_file_addr = os.path.join(library_dir, IP_perf_file_name)
    gpp_perf_file_addr = os.path.join(library_dir, Block_char_file_name)

    #  parse the file and collect in a dictionary
    reader = csv.DictReader(open(IP_perf_file_addr, 'r'))
    dict_list = []
    for line in reader:
        dict_list.append(line)

    #  collect data on ref gpp  (from ref_gpp perspective)
    ref_gpp_dict = parse_ref_block_values(library_dir, Block_char_file_name, ("pe", "gpp"))
    # calculate the work (basically number of instructions per task)
    task_perf_on_ref_gpp_dict = parse_task_perf_on_ref_gpp(library_dir, IP_perf_file_name, Block_char_file_name)

    work_dict = {}  # per task, what is the work (number of instruction processed)

    # right now, the data is reported in cycles
    for task_name, time in task_perf_on_ref_gpp_dict.items():
        # the following two lines can be omitted once the time is reported in seconds
        cycles = time  # we have to do this, because at the moment, the data is reported in cycles
        time = cycles/float(ref_gpp_dict["Freq"]) # we have to do this, because at the moment, the data is reported in cycles

        # don't need to use frequency correction values for work since frequency cancels out
        work_dict[task_name] = time* float(ref_gpp_dict["dhrystone_IPC"])*float(ref_gpp_dict["Freq"])

    return work_dict


# ------------------------------
# Functionality:
#   find all the ips (accelerators) for a task
# ------------------------------
def deduce_IPs(task_PEs, gpp_names):
    ip_dict = {}
    for task_PE in task_PEs:
        for PE, cycles in task_PE.items():
            if PE in gpp_names or PE in ip_dict.keys():
                continue
            ip_dict[PE] = {}
            ip_dict[PE]["Freq"] = 100000000

    return ip_dict


def convert_energy_to_power(task_PPA_dict):
    task_names = task_PPA_dict["perf"].keys()
    task_PPA_dict["power"] = {}
    for task_name in task_names:
        task_PPA_dict["power"][task_name] = {}
        blocks = task_PPA_dict["perf"][task_name].keys()
        for block in blocks:
            if task_PPA_dict["perf"][task_name][block] == 0 or (block not in task_PPA_dict["energy"][task_name].keys()):
                task_PPA_dict["power"][task_name][block] = 0
            else:
                task_PPA_dict["power"][task_name][block] = task_PPA_dict["energy"][task_name][block]/task_PPA_dict["perf"][task_name][block]


# based on various knobs correct for the parsed data.
# if the input doesn't require any correction, then no changes are applied
def gen_correction_values(workload, misc_knobs):
    # instantiate the dictionary
    correction_dict = {}
    correction_dict["ip"] = {}
    correction_dict["gpp"] = {}
    correction_dict["dram"] = {}
    correction_dict["sram"] = {}
    correction_dict["ic"] = {}

    # initilize the correction values (in case misc knobs do not contain these values)
    ip_freq_correction_ratio = 1
    gpp_freq_correction_ratio = 1
    dram_freq_correction_ratio = 1
    sram_freq_correction_ratio = 1
    ic_freq_correction_ratio = 1
    tech_node_SF = {}
    tech_node_SF["perf"] =1
    tech_node_SF["energy"] = {"gpp":1, "non_gpp":1}
    tech_node_SF["area"] = {"mem":1, "non_mem":1, "gpp":1}

    # if any of hte above values found in misc_knobs, over write
    if "ip_freq_correction_ratio" in misc_knobs.keys():
        ip_freq_correction_ratio = misc_knobs["ip_freq_correction_ratio"]

    if "gpp_freq_correction_ratio" in misc_knobs.keys():
        gpp_freq_correction_ratio = misc_knobs["gpp_freq_correction_ratio"]

    if "dram_freq_correction_ratio" in misc_knobs.keys():
        dram_freq_correction_ratio = misc_knobs["dram_freq_correction_ratio"]

    if "sram_freq_correction_ratio" in misc_knobs.keys():
        sram_freq_correction_ratio = misc_knobs["sram_freq_correction_ratio"]

    if "sram_freq_correction_ratio" in misc_knobs.keys():
        ic_freq_correction_ratio = misc_knobs["ic_freq_correction_ratio"]

    if "tech_node_SF" in misc_knobs.keys():
        tech_node_SF = misc_knobs["tech_node_SF"]

    # populate the correction dictionary
    correction_dict["ip"]["work_rate"] = (1/tech_node_SF["perf"])*ip_freq_correction_ratio
    correction_dict["gpp"]["work_rate"] = (1/tech_node_SF["perf"])*gpp_freq_correction_ratio
    correction_dict["dram"]["work_rate"] = (1/tech_node_SF["perf"])*dram_freq_correction_ratio
    correction_dict["sram"]["work_rate"] = (1/tech_node_SF["perf"])*sram_freq_correction_ratio
    correction_dict["ic"]["work_rate"] = (1/tech_node_SF["perf"])*ic_freq_correction_ratio

    correction_dict["ip"]["work_over_energy"] = (1/tech_node_SF["energy"]["non_gpp"])*1
    correction_dict["gpp"]["work_over_energy"] = (1/tech_node_SF["energy"]["gpp"])*1
    correction_dict["sram"]["work_over_energy"] = (1/tech_node_SF["energy"]["non_gpp"])*1
    correction_dict["dram"]["work_over_energy"] = (1/tech_node_SF["energy"]["non_gpp"])*1
    correction_dict["ic"]["work_over_energy"] = (1/tech_node_SF["energy"]["non_gpp"])*1


    correction_dict["ip"]["work_over_area"] = (1/tech_node_SF["area"]["non_mem"])*1
    correction_dict["gpp"]["work_over_area"] = (1/tech_node_SF["area"]["gpp"])*1
    correction_dict["sram"]["work_over_area"] = (1/tech_node_SF["area"]["mem"])*1
    correction_dict["dram"]["work_over_area"] = (1/tech_node_SF["area"]["mem"])*1
    correction_dict["ic"]["work_over_area"] = (1/tech_node_SF["area"]["non_mem"])*1

    correction_dict["ip"]["one_over_area"] = (1 / tech_node_SF["area"]["non_mem"]) * 1
    correction_dict["gpp"]["one_over_area"] = (1 / tech_node_SF["area"]["gpp"]) * 1
    correction_dict["sram"]["one_over_area"] = (1 / tech_node_SF["area"]["mem"]) * 1
    correction_dict["dram"]["one_over_area"] = (1 / tech_node_SF["area"]["mem"]) * 1
    correction_dict["ic"]["one_over_area"] = (1 / tech_node_SF["area"]["non_mem"]) * 1

    return correction_dict

# ------------------------------
# Functionality:
#   parse the hardware library
# ------------------------------
def parse_hardware_library(library_dir, IP_perf_file_name,
                           IP_energy_file_name, IP_area_file_name,
                           Block_char_file_name, task_itr_cnt_file_name, workload, misc_knobs):

    def gen_freq_range(misc_knobs, block_sub_type):
        assert(block_sub_type in ["ip", "mem", "ic"])
        if block_sub_type+"_spawn" not in misc_knobs.keys():
            result = [1]
        else:
            spawn = misc_knobs[block_sub_type+"_spawn"]
            result = spawn[block_sub_type+"_freq_range"]
            #upper_bound = spawn[block_sub_type+"_freq_range"]["upper_bound"]
            #incr = spawn[block_sub_type+"_freq_range"]["incr"]
            #result = list(range(1, int(upper_bound), int(incr)))
        return result


    def gen_loop_itr_range(task_name, task_itr_cnt, misc_knobs):
        max_num_itr = 1
        max_spawn_ip_by_loop_itr = 1
        loop_itr_incr = 1

        # base cases
        if task_name not in task_itr_cnt:
            return range(1, 2)
        else:
            max_num_itr = task_itr_cnt[task_name]
        if max_num_itr == 1:
            return range(1, 2)
        if "ip_spawn" not in misc_knobs.keys():
            return range(1,2)

        # sanity check
        assert misc_knobs["ip_spawn"]["ip_loop_unrolling"]["spawn_mode"]  in ["arithmetic", "geometric"]
        if misc_knobs["ip_spawn"]["ip_loop_unrolling"]["spawn_mode"] == "geometric":
            assert(misc_knobs["ip_spawn"]["ip_loop_unrolling"]["incr"] > 1)

        # get parameters
        if "incr" in misc_knobs["ip_spawn"]["ip_loop_unrolling"].keys():
            loop_itr_incr = misc_knobs["ip_spawn"]["ip_loop_unrolling"]["incr"]

        if "max_spawn_ip" in misc_knobs["ip_spawn"]["ip_loop_unrolling"].keys():
            max_spawn_ip_by_loop_itr = misc_knobs["ip_spawn"]["ip_loop_unrolling"]["max_spawn_ip"]
        else:
            max_spawn_ip_by_loop_itr = max_num_itr

        # use arithmetic or geometric progression to spawn ips
        if misc_knobs["ip_spawn"]["ip_loop_unrolling"]["spawn_mode"] == "arithmetic":
            num_ips_perspective_2 = int(max_num_itr / loop_itr_incr)
            result = list(range(1, int(max_num_itr), int(loop_itr_incr)))
        elif misc_knobs["ip_spawn"]["ip_loop_unrolling"]["spawn_mode"] == "geometric":
            num_ips_perspective_2 = int(math.log(max_num_itr, loop_itr_incr))
            result = [loop_itr_incr** (n) for n in range(0, num_ips_perspective_2+ 1)]

        # cap the result by het maximum_spawn_ip
        if len(result) > max_spawn_ip_by_loop_itr:
            result = copy.deepcopy(result[:max_spawn_ip_by_loop_itr-1])

        # add the maximum as well
        if max_num_itr not in result:
            result.append(max_num_itr)
        return result
        # return the range

    # add corrections
    correction_values = gen_correction_values(workload, misc_knobs)

    hardware_library_dict = {}
    # parse IPs
    gpps = parse_block_based_on_types(library_dir, Block_char_file_name, ("pe", "gpp"))
    ip_template = parse_block_based_on_types(library_dir, Block_char_file_name, ("pe", "ip")) # this is just to collect clock freq for ips
    srams = parse_block_based_on_types(library_dir, Block_char_file_name, ("mem", "sram"))
    drams = parse_block_based_on_types(library_dir, Block_char_file_name, ("mem", "dram"))
    mems = {**drams, **srams}
    ics = parse_block_based_on_types(library_dir, Block_char_file_name, ("ic", "ic"))
    task_work_dict = gen_task_graph_work_dict(library_dir, IP_perf_file_name, Block_char_file_name, misc_knobs)

    task_PPA_dict = {}
    gpp_names = list(gpps.keys())
    task_PPA_dict["perf_in_cycles"] = parse_task_PPA(library_dir, IP_perf_file_name, gpp_names)  # are provided in cycles at the moment,

    ips = deduce_IPs(list(task_PPA_dict["perf_in_cycles"].values()), gpp_names)

    task_PPA_dict["perf"] = copy.deepcopy(task_PPA_dict["perf_in_cycles"])
    for task, task_PE in task_PPA_dict["perf_in_cycles"].items():
        for PE, cycles in  task_PE.items():
            if PE in ips:
                block_freq = ips[PE]["Freq"]
            elif PE in gpps:
                block_freq = gpps[PE]["Freq"]
            task_PPA_dict["perf"][task][PE] = float(cycles)/block_freq

    task_PPA_dict["energy"] = parse_task_PPA(library_dir, IP_energy_file_name, gpp_names)

    # generate power here
    convert_energy_to_power(task_PPA_dict)
    task_PPA_dict["area"] = parse_task_PPA(library_dir, IP_area_file_name, gpp_names)
    task_itr_cnt = parse_task_itr_cnt(library_dir, task_itr_cnt_file_name)

    for task_name in task_work_dict.keys():
        IP_perfs =  task_PPA_dict["perf"][task_name]
        IP_energy =  task_PPA_dict["energy"][task_name]  # reported in miliwatt at the moment
        IP_area =  task_PPA_dict["area"][task_name]
        IP_names = list(task_PPA_dict["perf"][task_name].keys())
        for IP_name in IP_names:
            if IP_name in hardware_library_dict.keys():
                hardware_library_dict[IP_name]["mappable_tasks"].append(task_name)
                continue
            if IP_name in gpps:
                hardware_library_dict[IP_name] = {}
                hardware_library_dict[IP_name]["work_rate"] = correction_values["gpp"]["work_rate"]*float(gpps[IP_name]['Freq'])*float(gpps[IP_name]["dhrystone_IPC"])
                hardware_library_dict[IP_name]["work_over_energy"] = correction_values["gpp"]["work_over_energy"]*float(gpps[IP_name]['Inst_per_joul'])
                hardware_library_dict[IP_name]["work_over_area"] = correction_values["gpp"]["work_over_area"]*(1.0)/float(gpps[IP_name]['Gpp_area'])
                hardware_library_dict[IP_name]["one_over_area"] = correction_values["gpp"]["one_over_area"]*(1.0)/float(gpps[IP_name]['Gpp_area']) # convention is that workoverarea is 1/area for fix areas (like IPs and GPPs)
                hardware_library_dict[IP_name]["mappable_tasks"] = [task_name]
                hardware_library_dict[IP_name]["type"] = "pe"
                hardware_library_dict[IP_name]["sub_type"] = "gpp"
                hardware_library_dict[IP_name]["clock_freq"] = gpps[IP_name]["Freq"]
                hardware_library_dict[IP_name]["BitWidth"] = gpps[IP_name]["BitWidth"]
                hardware_library_dict[IP_name]["loop_itr_cnt"] = 0
                hardware_library_dict[IP_name]["loop_max_possible_itr_cnt"] = 0
                hardware_library_dict[IP_name]["hop_latency"] = gpps[IP_name]["hop_latency"]
                hardware_library_dict[IP_name]["pipe_line_depth"] = gpps[IP_name]["pipe_line_depth"]
                #print("taskname: " + str(task_name) + ", subtype: gpp, power is"+ str(hardware_library_dict[IP_name]["work_rate"]/hardware_library_dict[IP_name]["work_over_energy"] ))
            else:
                loop_itr_range_ = gen_loop_itr_range(task_name, task_itr_cnt, misc_knobs)
                ip_freq_range = gen_freq_range(misc_knobs, "ip")
                for loop_itr_cnt, ip_freq in itertools.product(loop_itr_range_, ip_freq_range):
                    IP_name_refined = IP_name +"_"+str(loop_itr_cnt) + "_" + str(ip_freq)
                    hardware_library_dict[IP_name_refined] = {}
                    hardware_library_dict[IP_name_refined]["work_rate"] = (ip_freq*loop_itr_cnt*correction_values["ip"]["work_rate"])*(task_work_dict[task_name]/(IP_perfs[IP_name]))
                    hardware_library_dict[IP_name_refined]["work_over_energy"] = (correction_values["ip"]["work_over_energy"]/loop_itr_cnt)*(task_work_dict[task_name]/(float(IP_energy[IP_name]*float((10**-15)))))
                    hardware_library_dict[IP_name_refined]["work_over_area"] = (correction_values["ip"]["work_over_area"]/loop_itr_cnt)*(task_work_dict[task_name])/(IP_area[IP_name]*(10**-12))
                    hardware_library_dict[IP_name_refined]["one_over_area"] = (correction_values["ip"]["one_over_area"]/loop_itr_cnt)*(1.0)/(IP_area[IP_name]*(10**-12)) # convention is that workoverarea is 1/area for fix areas (like IPs and GPPs)
                    hardware_library_dict[IP_name_refined]["mappable_tasks"] = [task_name]
                    hardware_library_dict[IP_name_refined]["type"] = "pe"
                    hardware_library_dict[IP_name_refined]["sub_type"] = "ip"
                    hardware_library_dict[IP_name_refined]["clock_freq"] = ip_template["IP"]["Freq"]*ip_freq
                    hardware_library_dict[IP_name_refined]["BitWidth"] = ip_template["IP"]["BitWidth"]
                    hardware_library_dict[IP_name_refined]["loop_itr_cnt"] = loop_itr_cnt
                    hardware_library_dict[IP_name_refined]["loop_max_possible_itr_cnt"] = task_itr_cnt[task_name]
                    hardware_library_dict[IP_name_refined]["hop_latency"] = ip_template["IP"]["hop_latency"]
                    hardware_library_dict[IP_name_refined]["pipe_line_depth"] = ip_template["IP"]["pipe_line_depth"]
                    #print("taskname: " + str(task_name) + ", subtype: ip, power is"+ str(hardware_library_dict[IP_name]["work_rate"]/hardware_library_dict[IP_name]["work_over_energy"] ))

    for blck_name, blck_value in mems.items():
        mem_freq_range = gen_freq_range(misc_knobs, "mem")
        for freq in mem_freq_range:
            IP_name_refined = blck_value['Name']+ "_" + str(freq)
            hardware_library_dict[IP_name_refined] = {}
            #hardware_library_dict[blck_value['Name']] = {}
            hardware_library_dict[IP_name_refined]["work_rate"] = freq*correction_values[blck_value["Subtype"]]["work_rate"]*float(blck_value['BitWidth'])*float(blck_value['Freq'])
            hardware_library_dict[IP_name_refined]["work_over_energy"] = correction_values[blck_value["Subtype"]]["work_over_energy"]*float(blck_value['Byte_per_joul'])
            hardware_library_dict[IP_name_refined]["work_over_area"] = correction_values[blck_value["Subtype"]]["work_over_area"]*float(blck_value['Byte_per_m'])
            hardware_library_dict[IP_name_refined]["one_over_area"] = correction_values[blck_value["Subtype"]]["one_over_area"]*float(blck_value['Byte_per_m'])  # not gonna be used so doesn't matter how to populate
            hardware_library_dict[IP_name_refined]["mappable_tasks"] = 'all'
            hardware_library_dict[IP_name_refined]["type"] = "mem"
            hardware_library_dict[IP_name_refined]["sub_type"] = blck_value['Subtype']
            hardware_library_dict[IP_name_refined]["clock_freq"] = freq*blck_value["Freq"]
            hardware_library_dict[IP_name_refined]["BitWidth"] = blck_value["BitWidth"]
            hardware_library_dict[IP_name_refined]["loop_itr_cnt"] = 0
            hardware_library_dict[IP_name_refined]["loop_max_possible_itr_cnt"] = 0
            hardware_library_dict[IP_name_refined]["hop_latency"] = blck_value["hop_latency"]
            hardware_library_dict[IP_name_refined]["pipe_line_depth"] = blck_value["pipe_line_depth"]

    for blck_name, blck_value in ics.items():
        ic_freq_range = gen_freq_range(misc_knobs, "ic")
        for freq in ic_freq_range:
            IP_name_refined = blck_value['Name']+ "_" + str(freq)
            hardware_library_dict[IP_name_refined] = {}
            hardware_library_dict[IP_name_refined]["work_rate"] = freq*correction_values[blck_value["Subtype"]]["work_rate"]*float(blck_value['BitWidth'])*float(blck_value['Freq'])
            hardware_library_dict[IP_name_refined]["work_over_energy"] = correction_values[blck_value["Subtype"]]["work_over_energy"]*float(blck_value['Byte_per_joul'])
            hardware_library_dict[IP_name_refined]["work_over_area"] = correction_values[blck_value["Subtype"]]["work_over_area"]*float(blck_value['Byte_per_m'])
            hardware_library_dict[IP_name_refined]["one_over_area"] = correction_values[blck_value["Subtype"]]["one_over_area"]*float(blck_value['Byte_per_m']) # not gonna be used so doesn't matter how to populate
            hardware_library_dict[IP_name_refined]["mappable_tasks"] = 'all'
            hardware_library_dict[IP_name_refined]["type"] = "ic"
            hardware_library_dict[IP_name_refined]["sub_type"] = "ic"
            hardware_library_dict[IP_name_refined]["clock_freq"] = freq*blck_value["Freq"]
            hardware_library_dict[IP_name_refined]["BitWidth"] = blck_value["BitWidth"]
            hardware_library_dict[IP_name_refined]["loop_itr_cnt"] = 0
            hardware_library_dict[IP_name_refined]["loop_max_possible_itr_cnt"] = 0
            hardware_library_dict[IP_name_refined]["hop_latency"] = blck_value["hop_latency"]
            hardware_library_dict[IP_name_refined]["pipe_line_depth"] = blck_value["pipe_line_depth"]

    return hardware_library_dict

# collect budget values for each workload
def collect_budgets(workloads_to_consider, budget_misc_knobs, library_dir, prefix=""):
    if "base_budget_scaling" not in budget_misc_knobs.keys():
        base_budget_scaling = {"latency":1, "power":1, "area":1}
    else:
        base_budget_scaling = budget_misc_knobs["base_budget_scaling"]

    # get files
    file_list = [f for f in os.listdir(library_dir) if os.path.isfile(os.path.join(library_dir, f))]
    misc_file_name = get_full_file_name(prefix + "Budget.csv", file_list)

    # get the time profile
    df = pd.read_csv(os.path.join(library_dir, misc_file_name))
    workloads = df['Workload']
    workload_last_task = {}

    budgets_dict = {}
    budgets_dict = defaultdict(dict)
    other_values_dict = defaultdict(dict)
    other_values_dict["glass"] = {}
    budgets_dict["glass"]["latency"] = {}

    for metric in config.budgetted_metrics:
        if metric in ["power", "area"] and not len(workloads_to_consider) == 1:
            budgets_dict["glass"][metric] = (df.loc[df['Workload'] == "all"])[metric].values[0]
            budgets_dict["glass"][metric] *= float(base_budget_scaling[metric])
            # this is a hack for now. change later.
            # but used for budget sweep for now
            #budgets_dict["glass"][metric] = config.budget_dict["glass"][metric]
        elif metric in ["latency"] or len(workloads_to_consider)==1:
            for idx in range(0, len(workloads)):
                workload_name = workloads[idx]
                if workload_name == "all" or workload_name not in workloads_to_consider:
                    continue
                if metric == "latency":
                    budgets_dict["glass"][metric][workload_name] = (df.loc[df['Workload'] == workload_name])[metric].values[0]
                    budgets_dict["glass"][metric][workload_name] *= float(base_budget_scaling[metric])
                else:
                    budgets_dict["glass"][metric] = (df.loc[df['Workload'] == workload_name])[metric].values[0]
                    budgets_dict["glass"][metric] *= float(base_budget_scaling[metric])

    for metric in config.other_metrics:
        other_values_dict["glass"][metric] = (df.loc[df['Workload'] == "all"])[metric].values[0]

    return budgets_dict, other_values_dict

# get the last task for each workload
def collect_last_task(workloads_to_consider, library_dir, prefix=""):
    blocksL: List[BlockL] = []  # collection of all the blocks
    pe_mapsL: List[TaskToPEBlockMapL] = []
    pe_schedulesL: List[TaskScheduleL] = []


    # get files
    file_list = [f for f in os.listdir(library_dir) if os.path.isfile(os.path.join(library_dir, f))]
    misc_file_name = get_full_file_name(prefix + "Last Tasks.csv", file_list)

    # get the time profile
    df = pd.read_csv(os.path.join(library_dir, misc_file_name))
    workloads = df['workload']
    last_tasks = df['last_task']
    workload_last_task = {}
    for idx in range(0, len(workloads)):
        workload = workloads[idx]
        if workload not in workloads_to_consider:
            continue
        workload_last_task[workloads[idx]] = last_tasks[idx]

    return workload_last_task

# ------------------------------
# Functionality:
#   generate the hardware graph with light libraries
# ------------------------------
def gen_hardware_graph(library_dir, prefix = "") :
    # get files
    file_list = [f for f in os.listdir(library_dir) if os.path.isfile(os.path.join(library_dir, f))]
    hardware_graph_file_name = get_full_file_name(prefix + "Hardware Graph.csv", file_list)
    task_to_hardware_mapping_file_name = get_full_file_name(prefix + "Task to Hardware Mapping.csv", file_list)
    hardware_graph_file_addr =  os.path.join(library_dir, hardware_graph_file_name)
    hardware_graph_dict = parse_hardware_graph(hardware_graph_file_addr)

    return hardware_graph_dict


# ------------------------------
# Functionality:
#   generate the hardware graph with light libraries
# ------------------------------
def gen_task_to_hw_mapping(library_dir, prefix = "") :
    # get files
    file_list = [f for f in os.listdir(library_dir) if os.path.isfile(os.path.join(library_dir, f))]
    task_to_hardware_mapping_file_name = get_full_file_name(prefix + "Task To Hardware Mapping.csv", file_list)
    task_to_hardware_mapping_file_addr =  os.path.join(library_dir, task_to_hardware_mapping_file_name)
    task_to_hardware_mapping = parse_task_to_hw_mapping(task_to_hardware_mapping_file_addr)

    return task_to_hardware_mapping


# ------------------------------
# Functionality:
#   generate the hardware library
# ------------------------------
def gen_hardware_library(library_dir, prefix, workload, misc_knobs={}):
    blocksL: List[BlockL] = []  # collection of all the blocks
    pe_mapsL: List[TaskToPEBlockMapL] = []
    pe_schedulesL: List[TaskScheduleL] = []


    # get files
    file_list = [f for f in os.listdir(library_dir) if os.path.isfile(os.path.join(library_dir, f))]
    data_movement_file_name = get_full_file_name(prefix + "Task Data Movement.csv", file_list)
    IP_perf_file_name = get_full_file_name(prefix + "Task PE Performance.csv", file_list)
    IP_energy_file_name = get_full_file_name(prefix +  "Task PE Energy.csv", file_list)
    IP_area_file_name = get_full_file_name(prefix +  "Task PE Area.csv", file_list)
    task_itr_cnt_file_name = get_full_file_name(prefix+ "Task Itr Count.csv", file_list)
    Block_char_file_name = get_full_file_name("misc_database - "+ "Block Characteristics.csv", file_list)
    common_block_char_file_name =  get_full_file_name("misc_database - "+ "Common Hardware.csv", file_list)

    # get the time profile
    #task_perf_file_addr = os.path.join(library_dir, IP_perf_file_name)
    #ref_gpp_dict = parse_ref_block_values(library_dir, Block_char_file_name, ("pe", "gpp"))
    # calculate the work (basically number of instructions per task)
    task_perf_on_ref_gpp_dict = parse_task_perf_on_ref_gpp(library_dir, IP_perf_file_name, Block_char_file_name)
    # set up the schedules
    for task_name, _ in task_perf_on_ref_gpp_dict.items():
        pe_schedulesL.append(TaskScheduleL(task_name=task_name, starting_time=0))


    # get the mapping and IP library
    hardware_library_dict = parse_hardware_library(library_dir, IP_perf_file_name,
                                                   IP_energy_file_name, IP_area_file_name,
                                                   Block_char_file_name, task_itr_cnt_file_name, workload, misc_knobs)
    block_suptype = "gpp"  # default.
    for IP_name, values in hardware_library_dict.items():
        block_subtype = values['sub_type']
        block_type = values['type']
        blocksL.append(
            BlockL(block_instance_name=IP_name, block_type=block_type, block_subtype=block_subtype,
                   peak_work_rate_distribution = {hardware_library_dict[IP_name]["work_rate"]:1},
                   work_over_energy_distribution = {hardware_library_dict[IP_name]["work_over_energy"]:1},
                   work_over_area_distribution = {hardware_library_dict[IP_name]["work_over_area"]:1},
                   one_over_area_distribution = {hardware_library_dict[IP_name]["one_over_area"]:1},
                   clock_freq=hardware_library_dict[IP_name]["clock_freq"], bus_width=hardware_library_dict[IP_name]["BitWidth"],
                   loop_itr_cnt=hardware_library_dict[IP_name]["loop_itr_cnt"], loop_max_possible_itr_cnt=hardware_library_dict[IP_name]["loop_max_possible_itr_cnt"],
                   hop_latency=hardware_library_dict[IP_name]["hop_latency"], pipe_line_depth=hardware_library_dict[IP_name]["pipe_line_depth"],))

        if block_type == "pe":
            for mappable_tasks in hardware_library_dict[IP_name]["mappable_tasks"]:
                task_to_block_map_ = TaskToPEBlockMapL(task_name=mappable_tasks, pe_block_instance_name=IP_name)
                pe_mapsL.append(task_to_block_map_)

    return blocksL, pe_mapsL, pe_schedulesL
