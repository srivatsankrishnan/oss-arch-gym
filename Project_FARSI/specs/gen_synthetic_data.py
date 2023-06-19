#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

from specs.LW_cl import *
from specs.parse_libraries.parse_library import  *



def gen_tg_with_hops(task_name_intensity_type, others_task_cnt, parallel_task_cnt, serial_task_cnt, parallel_task_type, num_of_NoCs):
    task_names = [task_name for task_name, intensity in task_name_intensity_type]
    task_name_position = []
    name = "synthetic_"
    parents = [name + "souurce"]
    task_name_position.append((task_names[0], parents))

    parents = [name + "0"]
    task_name_position.append((task_names[1], parents))
    last_task = 2

    """
    parents = [name + "0"]
    task_name_position.append((task_names[2], parents))
    last_task = 3

    parents = [name + "1", name + "2"]
    task_name_position.append((task_names[3], parents))
    last_task =4
    """

    for i in range(0, others_task_cnt + serial_task_cnt - 3):
        parents =  [name + str(last_task - 1)]
        task_name_position.append((task_names[last_task], parents))
        last_task +=1

    for i in range(0, num_of_NoCs-2):
        parents = [name + str(last_task - 1)]
        task_name_position.append((task_names[last_task], parents))
        last_task += 1
    return task_name_position,"_"


def gen_tg_core_improved(task_name_intensity_type, others_task_cnt, parallel_task_cnt, serial_task_cnt, parallel_task_type, mode, num_of_NoCs):
    task_names = [task_name for task_name, intensity in task_name_intensity_type]
    task_name_position = []
    name = "synthetic_"
    #parallel_task_type = "edge_detection"
    #parallel_task_type = "audio"
    hoppy_tasks = []

    for i in range(0,2):
        if i == 0:
            parents = [name+"souurce"]
            task_name_position.append((task_names[i],parents))
        if i in [1]:
            parents = [name +str(0)]
            task_name_position.append((task_names[i], parents))


    num_of_NoCs_added_tasks = max(num_of_NoCs-2, 0)

    # spacial case
    if parallel_task_cnt == 0:
        last_task = 2
        for i in range(0, others_task_cnt - 4 - num_of_NoCs_added_tasks):
            parents =  [name + str(last_task- 1)]
            task_name_position.append((task_names[last_task], parents))
            last_task +=1

        if mode == "hop":
            for i in range(0, num_of_NoCs_added_tasks):
                parents = [name + str(last_task-1)]
                task_name_position.append((task_names[last_task], parents))
                hoppy_tasks.append(task_names[last_task])
                last_task += 1

        return task_name_position,{},hoppy_tasks




    parallel_task_cnt -=1 # by default has one
    # set up parallel type of audio
    parallel_task_names = {}
    parallel_task_names[0] = []
    parallel_task_names[0].append(name + str(1))
    last_task = 2
    if parallel_task_type == "audio_style":
        parallel_offset =1
        last_serial = serial_offset = 1
        for i in range(0, parallel_task_cnt):
            parents = [name + str(parallel_offset-1)]
            task_name_position.append((task_names[1 + i+1], parents))
            parallel_task_names[0].append(name + str(i+2))
            last_task +=1

    # set up serial
    if parallel_task_type == "edge_detection_style":
        serial_task_cnt += int(parallel_task_cnt/2)


    for i in range(0, serial_task_cnt):
        if i ==0:
            parents = [el for el in parallel_task_names[0]]
        else:
            parents =  [name + str(last_task - 1)]
        task_name_position.append((task_names[last_task], parents))
        last_task +=1

    if serial_task_cnt == 0:
        parents = [el for el in parallel_task_names[0]]
    else:
        parents = [name + str(last_task-1)]
    task_name_position.append((task_names[last_task], parents))
    last_right = last_task

    # set up parallel type of edge
    parents = [name+str(0)]
    last_task +=1
    task_name_position.append((task_names[last_task], parents))
    left_begin =  last_task
    parallel_task_names[0].append(task_names[left_begin])

    if parallel_task_type == "edge_detection_style":
        for i in  range(0, int(parallel_task_cnt/2)):
            parents = [name + str(last_task)]
            last_task += 1
            task_name_position.append((task_names[last_task], parents))
        last_left = last_task
    else:
        last_left = last_task

    # set up last node
    parents = [name + str(last_left), name + str(last_right)]
    last_task +=1
    task_name_position.append((task_names[last_task], parents))


    if parallel_task_type == "edge_detection_style":
        idx =1
        for i in range(2, left_begin):
            parallel_task_names[idx] = []
            parallel_task_names[idx].append(name+str(idx))
            idx +=1
        idx = 1
        for i in range(left_begin, last_left):
            parallel_task_names[idx].append(name + str(i))
            idx+=1


    if mode == "hop":
        for i in range(0, num_of_NoCs_added_tasks):
            parents = [name + str(last_task)]
            task_name_position.append((task_names[last_task+1], parents))
            hoppy_tasks.append(task_names[last_task+1])
            last_task += 1

    return task_name_position,parallel_task_names, hoppy_tasks




def gen_tg_core(task_name_intensity_type, others_task_cnt, parallel_task_cnt, serial_task_cnt):
    task_names = [task_name for task_name, intensity in task_name_intensity_type]
    task_name_position = []
    name = "synthetic_"
    for i in range(0,2):
        if i == 0:
            parents = [name+"souurce"]
            task_name_position.append((task_names[i],parents))
        if i in [1]:
            parents = [name +str(0)]
            task_name_position.append((task_names[i], parents))

    parallel_offset =1
    last_serial = serial_offset = 1
    parallel_task_names = []
    parallel_task_names.append(name + str(1))
    for i in range(0, parallel_task_cnt):
        parents = [name + str(parallel_offset-1)]
        task_name_position.append((task_names[1 + i+1], parents))
        parallel_task_names.append(name + str(i+2))

    for i in range(0, serial_task_cnt):
        if i == 0:
            parents = [el for el in parallel_task_names]
        else:
            parents =   [name + str(1 + i +  parallel_task_cnt)]
        task_name_position.append((task_names[1 + i + 1 + parallel_task_cnt], parents))
        last_serial = serial_offset + i+1


    parents = [name + str(1 + 1 + parallel_task_cnt + serial_task_cnt-1)]
    task_name_position.append((task_names[1 + 1 + parallel_task_cnt + serial_task_cnt], parents))


    parents = [name+str(0)]
    task_name_position.append((task_names[1 + 2 + parallel_task_cnt + serial_task_cnt], parents))

    parents = [name + str(1 + 2 + parallel_task_cnt + serial_task_cnt), name + str(1 + 2 + parallel_task_cnt + serial_task_cnt-1)]
    task_name_position.append((task_names[1 + 2 + parallel_task_cnt + serial_task_cnt+1], parents))



    return task_name_position


# -----------
# Functionality:
#      split tasks to what can run in parallel and
#      what should run in serial
# Variables:
#      task_name_intensity_type: list of tuples : (task name, intensity (memory intensive, comp intensive))
# -----------
def cluster_tasks(task_name_intensity_type, avg_parallelism):
    task_names = [task_name for task_name, intensity in task_name_intensity_type]
    task_name_position = []
    state = "par"
    pos_ctr = 0
    state_ctr = avg_parallelism
    for idx, task in enumerate(task_names):
        if state == "par" and state_ctr > 0:
            state_ctr -= 1
        elif state == "par" and state_ctr == 0:
            state = "ser"
            pos_ctr += 1
        elif state == "ser":
            state = "par"
            state_ctr = avg_parallelism
            state_ctr -= 1
            pos_ctr += 1
        task_name_position.append((task, pos_ctr))
    return task_name_position


# -----------
# Functionality:
#      generate synthetic work (instructions, bytes) for tasks
# Variables:
#      task_name_intensity_type: list of tuples : (task name, intensity (memory intensive, comp intensive))
# -----------
def generate_synthetic_work(task_exec_intensity_type, general_task_type_char):

    work_dict = {}
    for task, intensity in task_exec_intensity_type:
        if "siink" in task or "souurce" in task:
            work_dict[task] = 0
        else:
            work_dict[task] = general_task_type_char[intensity]["exec"]

    return work_dict


def generate_synthetic_datamovement_asymetric_tg(task_exec_intensity_type, others_task_cnt, parallel_task_cnt, serial_task_cnt , parallel_task_type, exec_intensity_scaling_factor, num_of_NoCs):
    # hardcoded data.
    # TODO: move these into a config file later
    general_task_type_char = {}
    general_task_type_char["memory_intensive"] = {}
    general_task_type_char["comp_intensive"] = {}
    general_task_type_char["dummy_intensive"] = {}
    general_task_type_char["memory_intensive"]["read_bytes"] = exec_intensity_scaling_factor*500000
    general_task_type_char["memory_intensive"]["write_bytes"] = exec_intensity_scaling_factor*500000 # hardcoded, delete later
    general_task_type_char["comp_intensive"]["read_bytes"] = exec_intensity_scaling_factor*80000
    general_task_type_char["comp_intensive"]["write_bytes"] = exec_intensity_scaling_factor*80000

    general_task_type_char["memory_intensive"]["exec"] = exec_intensity_scaling_factor*50000
    general_task_type_char["comp_intensive"]["exec"] = exec_intensity_scaling_factor*10*50000*3

    general_task_type_char["dummy_intensive"]["read_bytes"] = 64
    general_task_type_char["dummy_intensive"]["write_bytes"] = 64
    general_task_type_char["dummy_intensive"]["exec"] = 64

    #general_task_type_char["memory_intensive"]["read_bytes"] = math.floor((exec_intensity_scaling_factor * 500000)/256)*256
    #general_task_type_char["memory_intensive"]["write_bytes"] = math.floor((exec_intensity_scaling_factor * 500000)/256)*256

    # find a family task (parent, child or sibiling)
    def find_family(task_name_position, task_name, relationship):
        task_position = [position for task, position in task_name_position if task == task_name][0]
        if relationship == "parent":
            parents = [task_name for task_name, position in task_name_position if position ==(task_position -1)]
            return parents
        elif relationship == "child":
            children = [task_name for task_name, position in task_name_position if position ==(task_position +1)]
            return children
        else:
            print('relationsihp ' + relationship + " is not defined")
            exit(0)

    def find_family_asymetric_tg(task_name_position, task_name, relationship):
        task_parents = [parents for task, parents in task_name_position if task == task_name]
        if relationship == "parent":
            return task_parents[0]
        elif relationship == "child":
            children = []
            for task_name_, parents in task_name_position:
                for parent in parents:
                    if parent == task_name:
                        children.append(task_name_)
            return children
        else:
            print('relationsihp ' + relationship + " is not defined")
            exit(0)

    data_movement = {}
    if num_of_NoCs == 1:
        task_name_position, parallel_task_names, hoppy_tasks = gen_tg_core_improved(task_exec_intensity_type, others_task_cnt, parallel_task_cnt, serial_task_cnt, parallel_task_type, "non_hop", 1)
    else:
        task_name_position, parallel_task_names, hoppy_tasks = gen_tg_core_improved(task_exec_intensity_type, others_task_cnt, parallel_task_cnt, serial_task_cnt, parallel_task_type, "hop", num_of_NoCs)
        #task_name_position, parallel_task_names = gen_tg_with_hops(task_exec_intensity_type, others_task_cnt, parallel_task_cnt, serial_task_cnt, parallel_task_type, "hop", num_of_NoCs)

    task_name_position.insert(0,("synthetic_souurce", [""]))
    all_idx = []
    for el, pos in task_name_position:
        if "souurce" in el:
            continue
        all_idx.append(int(el.split('_')[1]))
    last_task_name = max(all_idx)
    parent = ["synthetic_"+str(last_task_name)]
    task_name_position.append(("synthetic_siink", parent))
    task_exec_intensity_type.insert(0, ("synthetic_souurce", (task_exec_intensity_type[0])[1]))
    task_exec_intensity_type.append(("synthetic_siink", task_exec_intensity_type[0][1]))

    for task,_ in task_name_position:
        data_movement[task] = {}
        tasks_parents = find_family_asymetric_tg(task_name_position, task, "parent")
        tasks_children = find_family_asymetric_tg(task_name_position, task, "child")
        for child in tasks_children:
            exec_intensity = [intensity  for task_, intensity in task_exec_intensity_type if task_== task][0]
            #if child == "synthetic_siink":  # hardcoded delete later
            #    data_movement[task][child] = 1
            #else:
            data_movement[task][child] = general_task_type_char[exec_intensity]["write_bytes"]
    return data_movement,general_task_type_char, parallel_task_names,hoppy_tasks



# -----------
# Functionality:
#      generate synthetic datamovement between tasks
# Variables:
#      task_exec_intensity_type: memory or computational intensive
#      average number of tasks that can run in parallel
# -----------
def generate_synthetic_datamovement(task_exec_intensity_type, avg_parallelism):
    # hardcoded data.
    # TODO: move these into a config file later
    general_task_type_char = {}
    general_task_type_char["memory_intensive"] = {}
    general_task_type_char["comp_intensive"] = {}
    general_task_type_char["memory_intensive"]["read_bytes"] = 100000
    general_task_type_char["memory_intensive"]["write_bytes"] = 100000 # hardcoded, delete later
    general_task_type_char["comp_intensive"]["read_bytes"] = 100
    general_task_type_char["comp_intensive"]["write_bytes"] = 100

    general_task_type_char["memory_intensive"]["exec"] = 100
    general_task_type_char["comp_intensive"]["exec"] = 10000000

    # find a family task (parent, child or sibiling)
    def find_family(task_name_position, task_name, relationship):
        task_position = [position for task, position in task_name_position if task == task_name][0]
        if relationship == "parent":
            parents = [task_name for task_name, position in task_name_position if position ==(task_position -1)]
            return parents
        elif relationship == "child":
            children = [task_name for task_name, position in task_name_position if position ==(task_position +1)]
            return children
        else:
            print('relationsihp ' + relationship + " is not defined")
            exit(0)



    data_movement = {}
    task_name_position = cluster_tasks(task_exec_intensity_type, avg_parallelism)
    task_name_position.insert(0,("synthetic_souurce", -1))
    task_name_position.append(("synthetic_siink", max([pos for el, pos in task_name_position])+1))
    task_exec_intensity_type.insert(0, ("synthetic_souurce", (task_exec_intensity_type[0])[1]))
    task_exec_intensity_type.append(("synthetic_siink", task_exec_intensity_type[0][1]))

    for task,_ in task_name_position:
        data_movement[task] = {}
        tasks_parents = find_family(task_name_position, task, "parent")
        tasks_children = find_family(task_name_position, task, "child")
        for child in tasks_children:
            exec_intensity = [intensity  for task_, intensity in task_exec_intensity_type if task_== task][0]
            #if child == "synthetic_siink":  # hardcoded delete later
            #    data_movement[task][child] = 1
            #else:
            data_movement[task][child] = general_task_type_char[exec_intensity]["write_bytes"]

    return data_movement


# we generate very simple scenarios for now.
def generate_synthetic_task_graphs_for_asymetric_graphs(num_of_tasks,  others_task_cnt, parallel_task_cnt, serial_task_cnt, parallel_task_type, intensity_params, num_of_NoCs):

    exec_intensity = intensity_params[0]
    exec_intensity_scaling_factor = intensity_params[1]  # scaling with respect to the referece (amount of data movement)
    intensity_ratio = intensity_params[2] # what percentage of the tasks are memory intensive

    #----------------------
    # assigning memory bounded ness or compute boundedness to tasks
    #----------------------
    num_of_NoCs_added_tasks = max(num_of_NoCs-2, 0)
    opposite_intensity_task_cnt = int((num_of_tasks - 2 - num_of_NoCs_added_tasks)*(1-intensity_ratio))
    opposite_intensity = list({"memory_intensive", "comp_intensive"}.difference(set([exec_intensity])))[0]

    last_idx = 0
    task_exec_intensity = []
    for idx in range(0, num_of_tasks - 2 - opposite_intensity_task_cnt - num_of_NoCs_added_tasks):
        task_exec_intensity.append(("synthetic_"+str(idx), exec_intensity))
        last_idx = idx+1

    for idx in range(0, opposite_intensity_task_cnt):
        task_exec_intensity.append(("synthetic_"+str(last_idx), opposite_intensity))
        last_idx +=1

    # for dummy tasks taht are used for hops
    for idx in range(0, num_of_NoCs_added_tasks):
        task_exec_intensity.append(("synthetic_"+str(last_idx + idx), "dummy_intensive"))


    # generate task graph and data movement
    task_graph_dict, general_task_type_char, parallel_task_names, hoppy_task_names = generate_synthetic_datamovement_asymetric_tg(task_exec_intensity, num_of_tasks, parallel_task_cnt, serial_task_cnt, parallel_task_type, exec_intensity_scaling_factor, num_of_NoCs)

    # collect number of instructions for each tasks
    work_dict = generate_synthetic_work(task_exec_intensity, general_task_type_char)
    for task,work in work_dict.items():
        intensity_ = "none"
        for task_, intensity__ in task_exec_intensity:
            if task_ == task:
                intensity_ = intensity__
                break
        if intensity_ == "comp_intensive":
            children_cnt = len(list(task_graph_dict[task].values()))
            work_dict[task] = work_dict[task]*(children_cnt+1)



    tasksL: List[TaskL] = []
    for task_name_, values in task_graph_dict.items():
        task_ = TaskL(task_name=task_name_, work=work_dict[task_name_])
        task_.add_task_work_distribution([(work_dict[task_name_], 1)])
        tasksL.append(task_)

    for task_name_, values in task_graph_dict.items():
        task_ = [taskL for taskL in tasksL if taskL.task_name == task_name_][0]
        for child_task_name, data_movement in values.items():
            if child_task_name in hoppy_task_names or  child_task_name in ["synthetic_siink"]:
                data_movement = 64
            child_task = [taskL for taskL in tasksL if taskL.task_name == child_task_name][0]
            task_.add_child(child_task, data_movement, "real")  # eye_tracking_soource t glint_mapping
            task_.add_task_to_child_work_distribution(child_task, [(data_movement, 1)])  # eye_tracking_soource t glint_mapping

    return tasksL,task_graph_dict, work_dict, parallel_task_names,hoppy_task_names


#generate_synthetic_task_graphs_for_asymetric_graphs(10, 3, "memory_intensive")


# we generate very simple scenarios for now.
def generate_synthetic_task_graphs(num_of_tasks, avg_parallelism, exec_intensity):
    assert(num_of_tasks > avg_parallelism)

    tasksL: List[TaskL] = []

    # generate a list of task names and their exec intensity (i.e., compute or memory intensive)
    task_exec_intensity = []
    for idx in range(0, num_of_tasks - 2):
        task_exec_intensity.append(("synthetic_"+str(idx), exec_intensity))

    # collect data movement data
    task_graph_dict = generate_synthetic_datamovement(task_exec_intensity, avg_parallelism)

    # collect number of instructions for each tasks
    work_dict = generate_synthetic_work(task_exec_intensity)

    for task_name_, values in task_graph_dict.items():
        task_ = TaskL(task_name=task_name_, work=work_dict[task_name_])
        task_.add_task_work_distribution([(work_dict[task_name_], 1)])
        tasksL.append(task_)

    for task_name_, values in task_graph_dict.items():
        task_ = [taskL for taskL in tasksL if taskL.task_name == task_name_][0]
        for child_task_name, data_movement in values.items():
            child_task = [taskL for taskL in tasksL if taskL.task_name == child_task_name][0]
            task_.add_child(child_task, data_movement, "real")  # eye_tracking_soource t glint_mapping
            task_.add_task_to_child_work_distribution(child_task, [(data_movement, 1)])  # eye_tracking_soource t glint_mapping

    return tasksL,task_graph_dict, work_dict

# generate a synthetic hardware library to generate systems from
def generate_synthetic_hardware_library(task_work_dict, library_dir, Block_char_file_name):

    blocksL: List[BlockL] = []  # collection of all the blocks
    pe_mapsL: List[TaskToPEBlockMapL] = []
    pe_schedulesL: List[TaskScheduleL] = []

    gpps = parse_block_based_on_types(library_dir, Block_char_file_name, ("pe", "gpp"))
    gpp_names = list(gpps.keys())
    mems = parse_block_based_on_types(library_dir, Block_char_file_name, ("mem", "sram"))
    mems.update(parse_block_based_on_types(library_dir, Block_char_file_name, ("mem", "dram")))
    ics  = parse_block_based_on_types(library_dir, Block_char_file_name, ("ic", "ic"))


    hardware_library_dict = {}
    for task_name in task_work_dict.keys():
        for IP_name in gpp_names:
            if IP_name in hardware_library_dict.keys():
                hardware_library_dict[IP_name]["mappable_tasks"].append(task_name)
                continue
            hardware_library_dict[IP_name] = {}
            if IP_name in gpps:
                hardware_library_dict[IP_name]["work_rate"] = float(gpps[IP_name]['Freq'])*float(gpps[IP_name]["dhrystone_IPC"])
                hardware_library_dict[IP_name]["work_over_energy"] = float(gpps[IP_name]['Inst_per_joul'])
                hardware_library_dict[IP_name]["work_over_area"] = 1/float(gpps[IP_name]['Gpp_area'])
                hardware_library_dict[IP_name]["mappable_tasks"] = [task_name]
                hardware_library_dict[IP_name]["type"] = "pe"
                hardware_library_dict[IP_name]["sub_type"] = "gpp"
                hardware_library_dict[IP_name]["clock_freq"] =  gpps[IP_name]['Freq']
                hardware_library_dict[IP_name]["bus_width"] =  "NA"
                #print("taskname: " + str(task_name) + ", subtype: gpp, power is"+ str(hardware_library_dict[IP_name]["work_rate"]/hardware_library_dict[IP_name]["work_over_energy"] ))

    for blck_name, blck_value in mems.items():
        hardware_library_dict[blck_value['Name']] = {}
        hardware_library_dict[blck_value['Name']]["work_rate"] = float(blck_value['BitWidth'])*float(blck_value['Freq'])
        hardware_library_dict[blck_value['Name']]["work_over_energy"] = float(blck_value['Byte_per_joul'])
        hardware_library_dict[blck_value['Name']]["work_over_area"] =  float(blck_value['Byte_per_m'])
        hardware_library_dict[blck_value['Name']]["mappable_tasks"] = 'all'
        hardware_library_dict[blck_value['Name']]["type"] = "mem"
        hardware_library_dict[blck_value['Name']]["sub_type"] = blck_value['Subtype']
        hardware_library_dict[blck_value['Name']]["clock_freq"] = blck_value['Freq']
        hardware_library_dict[blck_value['Name']]["bus_width"] = blck_value['BitWidth']*8

    for blck_name, blck_value in ics.items():
        hardware_library_dict[blck_value['Name']] = {}
        hardware_library_dict[blck_value['Name']]["work_rate"] = float(blck_value['BitWidth'])*float(blck_value['Freq'])
        hardware_library_dict[blck_value['Name']]["work_over_energy"] = float(blck_value['Byte_per_joul'])
        hardware_library_dict[blck_value['Name']]["work_over_area"] =  float(blck_value['Byte_per_m'])
        hardware_library_dict[blck_value['Name']]["mappable_tasks"] = 'all'
        hardware_library_dict[blck_value['Name']]["type"] = "ic"
        hardware_library_dict[blck_value['Name']]["sub_type"] = "ic"
        hardware_library_dict[blck_value['Name']]["clock_freq"] = blck_value['Freq']
        hardware_library_dict[blck_value['Name']]["bus_width"] = blck_value['BitWidth']*8


    block_suptype = "gpp"  # default.
    for IP_name, values in hardware_library_dict.items():
        block_subtype = values['sub_type']
        block_type = values['type']
        blocksL.append(
            BlockL(block_instance_name=IP_name, block_type=block_type, block_subtype=block_subtype,
                   peak_work_rate_distribution = {hardware_library_dict[IP_name]["work_rate"]:1},
                   work_over_energy_distribution = {hardware_library_dict[IP_name]["work_over_energy"]:1},
                   work_over_area_distribution = {hardware_library_dict[IP_name]["work_over_area"]:1},
                   one_over_area_distribution = {1/hardware_library_dict[IP_name]["work_over_area"]:1},
                   clock_freq=hardware_library_dict[IP_name]["clock_freq"], bus_width=hardware_library_dict[IP_name]['bus_width']))

        if block_type == "pe":
            for mappable_tasks in hardware_library_dict[IP_name]["mappable_tasks"]:
                task_to_block_map_ = TaskToPEBlockMapL(task_name=mappable_tasks, pe_block_instance_name=IP_name)
                pe_mapsL.append(task_to_block_map_)

    return blocksL, pe_mapsL, pe_schedulesL