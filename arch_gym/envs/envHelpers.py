import os
import sys
import json
import random
import math
import configparser
import numpy as np
import yaml
os.sys.path.insert(0, os.path.abspath('/../../configs'))

#from configs import configs
from configs import arch_gym_configs
import shutil
from sims.Timeloop.process_params import TimeloopConfigParams
from subprocess import Popen, PIPE
import pandas as pd
from math import ceil

class CustomListDumper(yaml.Dumper):
    def increase_indent(self, flow=False, *args, **kwargs):
        return super(CustomListDumper, self).increase_indent(flow, False)


class helpers():
    def __init__(self):
        self.mem_control_basepath = arch_gym_configs.dram_mem_controller_config
        self.sniper_basepath = arch_gym_configs.sniper_config
        self.timeloop_param_obj = TimeloopConfigParams(arch_gym_configs.timeloop_parameters)
    
    def action_mapper(self, action, param):
        """
        RL agent outputs actions in [0,1]

        This function maps the action space to the actual values
        we split the action space (0,1) into equal parts depending on the number of valid actions each parameter can take
        We then bin the action to the appropriate range
        """
        num_bins = len(param)
        action_bin = 2/num_bins
        
        # create boundries for each bin
        boundries = np.arange(-1, 1, action_bin)
        
        
        # find the index in the boundries array that the action falls in
        try:
            action_index = np.where(boundries <= round(action))[0][-1]
        except Exception as e:
            print(action)
      
        return action_index
    
    def action_mapper_timeloop(self, action, param):
        """
        RL agent outputs actions in [0,1]
        This function maps the action space to the actual values
        we split the action space (0,1) into equal parts depending on the number of valid actions each parameter can take
        We then bin the action to the appropriate range
        """
        action_binned = []
        for idx in range(len(param)):
            each_action  = action[idx]
            num_bins = param[idx]

            action_bin = 2/num_bins

            # create boundries for each bin
            boundries = np.arange(-1, 1, action_bin)

            # find the index in the boundries array that the action falls in
            try:
                action_index = np.where(boundries <= round(each_action))[0][-1]
                action_binned.append(action_index)
            except Exception as e:
                print(action)

        return action_binned

    def action_mapper_FARSI(self, action, encoding_dictionary):
        mapped_action_vector = []
        # print(action)
        lb=encoding_dictionary["encoding_flattened_lb"]
        ub=encoding_dictionary["encoding_flattened_ub"]
        delta = [(x-y) for x,y in zip(ub, lb)]
        # print(ub)
        # print(lb)
        for id, el in enumerate(action):
            num_bins = delta[id] + 1
            bin_size = delta[id] / num_bins
            curr_bin = lb[id]
            for i in range(num_bins):
                if el <= (curr_bin + bin_size):
                    mapped_action_vector.append(lb[id] + i)
                    break
                curr_bin  += bin_size
        print("Action: " + str(mapped_action_vector))
        #mapped_action_vector = [0, 0, 0, 0, 0, 1, 2, 3, 3, 1, 0, 1, 2, 1, 1, 0, 2, 1, 2, 0, 1, 1, 1, 0, 2, 0, 2, 2, 0, 0, 0, 2, 2, 1]
        return mapped_action_vector

    def action_decoder_FARSI(self, action_encoded, encoding_dictionary):

        action = action_encoded
        pe_allocation = action[encoding_dictionary["delimiters"]["pe_allocation"][0]: encoding_dictionary["delimiters"]["pe_allocation"][1]+1]
        mem_allocation = action[encoding_dictionary["delimiters"]["mem_allocation"][0]: encoding_dictionary["delimiters"]["mem_allocation"][1]+1]
        bus_allocation = action[encoding_dictionary["delimiters"]["bus_allocation"][0]: encoding_dictionary["delimiters"]["bus_allocation"][1]+1]

        pe_to_bus_connection = action[encoding_dictionary["delimiters"]["pe_to_bus_connection"][0]: encoding_dictionary["delimiters"]["pe_to_bus_connection"][1]+1]
        bus_to_bus_connection = action[encoding_dictionary["delimiters"]["bus_to_bus_connection"][0]: encoding_dictionary["delimiters"]["bus_to_bus_connection"][1]+1]
        bus_to_mem_connection = action[encoding_dictionary["delimiters"]["bus_to_mem_connection"][0]: encoding_dictionary["delimiters"]["bus_to_mem_connection"][1]+1]

        task_to_pe_mapping = action[encoding_dictionary["delimiters"]["task_to_pe_mapping"][0]: encoding_dictionary["delimiters"]["task_to_pe_mapping"][1]+1]
        task_to_mem_mapping = action[encoding_dictionary["delimiters"]["task_to_mem_mapping"][0]: encoding_dictionary["delimiters"]["task_to_mem_mapping"][1]+1]

        action_decoded = {}
        action_decoded["pe_allocation"] = pe_allocation
        action_decoded["mem_allocation"] = mem_allocation
        action_decoded["bus_allocation"] = bus_allocation
        action_decoded["pe_to_bus_connection"]  = pe_to_bus_connection
        action_decoded["bus_to_bus_connection"] = bus_to_bus_connection
        action_decoded["bus_to_mem_connection"] = bus_to_mem_connection
        action_decoded["task_to_pe_mapping"] = task_to_pe_mapping
        action_decoded["task_to_mem_mapping"] = task_to_mem_mapping
        action_decoded["task_number_name_encoding"] = encoding_dictionary["task_number_name_encoding"]
        action_decoded["pe_number_name_encoding"] = encoding_dictionary["pe_number_name_encoding"]
        action_decoded["mem_number_name_encoding"] = encoding_dictionary["mem_number_name_encoding"]
        action_decoded["ic_number_name_encoding"] = encoding_dictionary["ic_number_name_encoding"]

        return action_decoded
    
    def transform_pow_2(self, value):
        return 2**round(math.log(value, 2))   

    # get design space information and generate the design space bounds
    def gen_SOC_design_space(self, env, design_space_mode, **kwargs):
        if env is not None: 
            dse = env.dse_hndlr.dse
        else: 
            dse = kwargs["dse"]
        database = dse.database



        #--------------------------
        # get all the tasks and blocks
        #--------------------------
        all_tasks = database.get_tasks()  # software tasks to include in the design.
        all_blocks = database.get_blocks()
        all_pes = [el for el in all_blocks if el.type == "pe"]
        all_mems = [el for el in all_blocks if el.type == "mem"]
        all_ics = [el for el in all_blocks if el.type == "ic"]
        all_ips = [el for el in all_blocks if el.subtype == "ip"]
        all_gpps = [el for el in all_blocks if el.subtype == "gpp"]

        #--------------------------
        # set the upper bounds
        #--------------------------
        # allocation
        # test
        if design_space_mode == "comprehensive":
            max_pe_cnt = max_mem_cnt = max_bus_cnt = task_cnt = len(all_tasks)
            all_pe_cnt = len(all_pes)
            all_mem_cnt = len(all_mems)
            all_ic_cnt = len(all_ics)
            pe_allocation_lb_value = mem_allocation_lb_value = bus_allocation_lb_value = -1


        if design_space_mode == "limited":
            max_pe_cnt = max_mem_cnt = max_bus_cnt = 3
            task_cnt = len(all_tasks)
            all_pe_cnt = 2
            all_mem_cnt = all_ic_cnt = 4 # number of PE types
            pe_allocation_lb_value = mem_allocation_lb_value = bus_allocation_lb_value = 0

        #--------------------------
        # specify the upper/lower bounds for all the design stages
        #--------------------------
        # allocation
        pe_allocation_ub = [all_pe_cnt-1  for el in range(0, max_pe_cnt)]
        pe_allocation_lb = [pe_allocation_lb_value  for el in range(0, max_pe_cnt)]
        mem_allocation_ub = [all_mem_cnt-1  for el in range(0, max_mem_cnt)]
        mem_allocation_lb = [mem_allocation_lb_value  for el in range(0, max_mem_cnt)]
        bus_allocation_ub = [all_ic_cnt-1  for el in range(0, max_bus_cnt)]
        bus_allocation_lb = [bus_allocation_lb_value  for el in range(0, max_bus_cnt)]

        # topology
        pe_to_bus_connection_ub = [max_bus_cnt-1 for el in range(0, max_pe_cnt)]
        pe_to_bus_connection_lb = [0 for el in range(0, max_pe_cnt)]
        bus_to_bus_connection_ub = [max_bus_cnt-1 for el in range(0, max_bus_cnt)]
        bus_to_bus_connection_lb = [-1 for el in range(0, max_bus_cnt)]
        bus_to_mem_connection_ub = [max_mem_cnt-1 for el in range(0, max_bus_cnt)]
        bus_to_mem_connection_lb = [-1 for el in range(0, max_bus_cnt)]

         # mapping
        task_to_pe_mapping_ub = [max_pe_cnt-1  for el in range(0, task_cnt)]
        task_to_pe_mapping_lb = [0  for el in range(0, task_cnt)]
        task_to_mem_mapping_ub = [max_mem_cnt-1  for el in range(0, task_cnt)]
        task_to_mem_mapping_lb = [0  for el in range(0, task_cnt)]

        DS = {} #design space
        DS["pe_allocation"] = {"ub": pe_allocation_ub, "lb":pe_allocation_lb}
        DS["mem_allocation"] = {"ub": mem_allocation_ub, "lb":mem_allocation_lb}
        DS["bus_allocation"] = {"ub": bus_allocation_ub, "lb":bus_allocation_lb}

        DS["pe_to_bus_connection"] = {"ub": pe_to_bus_connection_ub, "lb": pe_to_bus_connection_lb}
        DS["bus_to_bus_connection"] = {"ub": bus_to_bus_connection_ub, "lb": bus_to_bus_connection_lb}
        DS["bus_to_mem_connection"] = {"ub": bus_to_mem_connection_ub, "lb": bus_to_mem_connection_lb}


        DS["task_to_pe_mapping"] = {"ub": task_to_pe_mapping_ub, "lb": task_to_pe_mapping_lb}
        DS["task_to_mem_mapping"] = {"ub": task_to_mem_mapping_ub, "lb": task_to_mem_mapping_lb}

        return DS

    def gen_SOC_encoding(self, env, DS, design_space_mode = "comprehensive", **kwargs):
        def gen_encoding_dictionary(list_, type_):
            dictionary_ = {}
            for el in range(0, len(list_)):
                if type_ == "block":
                    dictionary_[el] = list_[el].instance_name_without_id
                if type_ == "task":
                    dictionary_[el] = list_[el].name

            return dictionary_

        # where does different encoding start/end
        def gen_encoding_delimters(encoding_sublist_size):
            encoding_delimiters = {}
            prev_up = 0  # previous upper bound
            for k,v in encoding_sublist_size.items():
                encoding_delimiters[k] = [prev_up, prev_up + v - 1]
                prev_up = prev_up + v
            return encoding_delimiters


        if env is not None: 
            dse = env.dse_hndlr.dse
        else: 
            dse = kwargs["dse"] 
        database = dse.database


        #--------------------------
        # get all the tasks and blocks
        #--------------------------
        all_tasks = database.get_tasks()  # software tasks to include in the design.
        all_blocks = database.get_blocks()
        all_pes = [el for el in all_blocks if el.type == "pe"]
        all_mems = [el for el in all_blocks if el.type == "mem"]
        all_ics = [el for el in all_blocks if el.type == "ic"]
        all_ips = [el for el in all_blocks if el.subtype == "ip"]
        all_gpps = [el for el in all_blocks if el.subtype == "gpp"]

        #--------------------------
        #--------------------------
        # generate encoding: name and layout encoding
        #--------------------------
        #--------------------------
        # number to name encoding
        task_number_name_encoding_dictionary = gen_encoding_dictionary(all_tasks, "task")
        #block_encoding_dictionary = gen_encoding_dictionary(all_blocks, "block")
        pe_number_name_encoding_dictionary = gen_encoding_dictionary(all_pes, "block")
        mem_number_name_encoding_dictionary = gen_encoding_dictionary(all_mems, "block")
        ic_number_name_encoding_dictionary = gen_encoding_dictionary(all_ics, "block")

        # layout encoding
        encoding_flattened_ub = []
        encoding_flattened_lb = []
        encoding_sublist_size = {}  # encoding sizes per sublist (sublist such as pe_allocation, mem_allocation, ...)

        # allocation
        encoding_flattened_ub.extend(DS["pe_allocation"]["ub"])
        encoding_flattened_lb.extend(DS["pe_allocation"]["lb"])
        encoding_sublist_size["pe_allocation"] = len(DS["pe_allocation"]["ub"])

        encoding_flattened_ub.extend(DS["mem_allocation"]["ub"])
        encoding_flattened_lb.extend(DS["mem_allocation"]["lb"])
        encoding_sublist_size["mem_allocation"] = len(DS["mem_allocation"]["ub"])

        encoding_flattened_ub.extend(DS["bus_allocation"]["ub"])
        encoding_flattened_lb.extend(DS["bus_allocation"]["lb"])
        encoding_sublist_size["bus_allocation"] = len(DS["bus_allocation"]["ub"])

        # extend with topology
        encoding_flattened_ub.extend(DS["pe_to_bus_connection"]["ub"])
        encoding_flattened_lb.extend(DS["pe_to_bus_connection"]["lb"])
        encoding_sublist_size["pe_to_bus_connection"] = len(DS["pe_to_bus_connection"]["ub"])

        encoding_flattened_ub.extend(DS["bus_to_bus_connection"]["ub"])
        encoding_flattened_lb.extend(DS["bus_to_bus_connection"]["lb"])
        encoding_sublist_size["bus_to_bus_connection"] = len(DS["bus_to_bus_connection"]["ub"])

        encoding_flattened_ub.extend(DS["bus_to_mem_connection"]["ub"])
        encoding_flattened_lb.extend(DS["bus_to_mem_connection"]["lb"])
        encoding_sublist_size["bus_to_mem_connection"] = len(DS["bus_to_mem_connection"]["ub"])

        # extend with mapping
        encoding_flattened_ub.extend(DS["task_to_pe_mapping"]["ub"])
        encoding_flattened_lb.extend(DS["task_to_pe_mapping"]["lb"])
        encoding_sublist_size["task_to_pe_mapping"] = len(DS["task_to_pe_mapping"]["ub"])

        encoding_flattened_ub.extend(DS["task_to_mem_mapping"]["ub"])
        encoding_flattened_lb.extend(DS["task_to_mem_mapping"]["lb"])
        encoding_sublist_size["task_to_mem_mapping"] = len(DS["task_to_mem_mapping"]["ub"])

        # this to help with the above issue
        lower_bounds = {"allocation": -1, "topology":-1, "mapping": 0}

        # find the delimeters
        encoding_delimeters = gen_encoding_delimters(encoding_sublist_size) # where does different encoding start/end

        # ------------------------------
        # compile all the information in one dicionary for one output
        # ------------------------------
        encoding_dictionary = {} # contain all the encoding information
        # number name encoding
        encoding_dictionary["task_number_name_encoding"] = task_number_name_encoding_dictionary
        #encoding_dictionary["block_encoding"] = block_encoding_dictionary
        encoding_dictionary["pe_number_name_encoding"] = pe_number_name_encoding_dictionary
        encoding_dictionary["mem_number_name_encoding"] = mem_number_name_encoding_dictionary
        encoding_dictionary["ic_number_name_encoding"] = ic_number_name_encoding_dictionary

        # upper/lower
        # allocation
        encoding_dictionary["pe_allocation_ub"] = DS["pe_allocation"]["ub"]
        encoding_dictionary["mem_allocation_ub"] = DS["mem_allocation"]["ub"]
        encoding_dictionary["bus_allocation_ub"] = DS["bus_allocation"]["ub"]
        encoding_dictionary["pe_allocation_lb"] = DS["pe_allocation"]["lb"]
        encoding_dictionary["mem_allocation_lb"] = DS["mem_allocation"]["lb"]
        encoding_dictionary["bus_allocation_lb"] = DS["bus_allocation"]["lb"]

        # topology
        encoding_dictionary["pe_to_bus_connection_ub"] = DS["pe_to_bus_connection"]["ub"]
        encoding_dictionary["bus_to_bus_connection_ub"] = DS["bus_to_bus_connection"]["ub"]
        encoding_dictionary["bus_to_mem_connection_ub"] = DS["bus_to_mem_connection"]["ub"]
        encoding_dictionary["pe_to_bus_connection_lb"] = DS["pe_to_bus_connection"]["lb"]
        encoding_dictionary["bus_to_bus_connection_lb"] = DS["bus_to_bus_connection"]["lb"]
        encoding_dictionary["bus_to_mem_connection_lb"] = DS["bus_to_mem_connection"]["lb"]

        # mapping
        encoding_dictionary["task_to_pe_mapping_ub"] = DS["task_to_pe_mapping"]["ub"]
        encoding_dictionary["task_to_mem_mapping_ub"] = DS["task_to_mem_mapping"]["ub"]
        encoding_dictionary["task_to_pe_mapping_lb"] = DS["task_to_pe_mapping"]["lb"]
        encoding_dictionary["task_to_mem_mapping_lb"] = DS["task_to_mem_mapping"]["lb"]

        # flattened
        encoding_dictionary["encoding_flattened_lb"] = encoding_flattened_lb
        encoding_dictionary["encoding_flattened_ub"] = encoding_flattened_ub

        # print(enco)
        # temp_lb = np.array(encoding_flattened_lb)
        # temp_ub = np.array(encoding_flattened_ub)
        # diff = (temp_ub - temp_lb)
        # product = np.int64(1)
        # print(diff)
        # for item in diff:
        #     product *= (item + 1)
        # print('{:.5E}'.format((product)))
        # sys.exit()
        # print("---------------------LB---------------------------")
        # print(encoding_dictionary["encoding_flattened_lb"])
        # print("---------------------UB---------------------------")
        # print(encoding_dictionary["encoding_flattened_ub"])
        # print(len(encoding_dictionary['encoding_flattened_ub'])== (len(encoding_dictionary['encoding_flattened_ub'])))
        # print(encoding_delimeters)
        # sys.exit()
        # meta data
        encoding_dictionary["encoding_lb_shortcut"] = lower_bounds
        encoding_dictionary["delimiters"] = encoding_delimeters

        return encoding_dictionary


    def extract_encoding_from_flattened_encoding(self, encoding_dictionary):
            # get the upper bound/lower bounds
            pe_allocation_ub = [encoding_dictionary["encoding_ub"][idx] for idx in range(encoding_dictionary["delimiters"]["pe_allocation"][0], encoding_dictionary["delimiters"]["pe_allocation"][1]+1)]
            mem_allocation_ub = [encoding_dictionary["encoding_ub"][idx] for idx in range(encoding_dictionary["delimiters"]["mem_allocation"][0], encoding_dictionary["delimiters"]["mem_allocation"][1]+1)]
            bus_allocation_ub = [encoding_dictionary["encoding_ub"][idx] for idx in range(encoding_dictionary["delimiters"]["bus_allocation"][0], encoding_dictionary["delimiters"]["bus_allocation"][1]+1)]

            pe_to_bus_connection_ub = [encoding_dictionary["encoding_ub"][idx] for idx in range(encoding_dictionary["delimiters"]["pe_to_bus_connection"][0], encoding_dictionary["delimiters"]["pe_to_bus_connection"][1]+1)]
            bus_to_bus_connection_ub = [encoding_dictionary["encoding_ub"][idx] for idx in range(encoding_dictionary["delimiters"]["bus_to_bus_connection"][0], encoding_dictionary["delimiters"]["bus_to_bus_connection"][1]+1)]
            bus_to_mem_connection_ub = [encoding_dictionary["encoding_ub"][idx] for idx in range(encoding_dictionary["delimiters"]["bus_to_mem_connection"][0], encoding_dictionary["delimiters"]["bus_to_mem_connection"][1]+1)]

            task_to_pe_mapping_ub = [encoding_dictionary["encoding_ub"][idx] for idx in range(encoding_dictionary["delimiters"]["task_to_pe_mapping"][0], encoding_dictionary["delimiters"]["task_to_pe_mapping"][1])]
            task_to_mem_mapping_ub = [encoding_dictionary["encoding_ub"][idx] for idx in range(encoding_dictionary["delimiters"]["task_to_mem_mapping"][0], encoding_dictionary["delimiters"]["task_to_mem_mapping"][1])]

            lower_bound = encoding_dictionary["encoding_lb_shortcut"]

            return pe_allocation_ub, mem_allocation_ub, bus_allocation_ub,  pe_to_bus_connection_ub, bus_to_bus_connection_ub, bus_to_mem_connection_ub, task_to_pe_mapping_ub, task_to_mem_mapping_ub, lower_bound

    def run_system_checks(self, env, encoding_dictionary, pe_allocation, mem_allocation, bus_allocation, pe_to_bus_connection, bus_to_bus_connection, bus_to_mem_connection, task_to_pe_mapping, task_to_mem_mapping):

        dse = env.dse_hndlr.dse
        dse_hndlr = env.dse_hndlr
        database = dse.database

        pe_number_name_encoding = encoding_dictionary["pe_number_name_encoding"]
        mem_number_name_encoding = encoding_dictionary["mem_number_name_encoding"]
        ic_number_name_encoding = encoding_dictionary["ic_number_name_encoding"]
        task_number_name_encoding = encoding_dictionary["task_number_name_encoding"]


        system_validity = True
        invalidity_reason = ""
        # instantiate all the hardware blocks
        #-----------------------
        # generate allocation
        #-----------------------
        pes =[]
        for el in pe_allocation:
            if el == -1:
                pes.append(-1)
                continue
            pe_name = pe_number_name_encoding[el]
            pe_ = dse_hndlr.database.get_block_by_name(pe_name)
            pe_instance = dse_hndlr.database.sample_similar_block(pe_)
            ordered_SOCsL = sorted(dse_hndlr.database.SOCsL, key=lambda SOC: SOC.get_budget("latency"))
            pe_instance.set_SOC(ordered_SOCsL[0].type, dse_hndlr.database.SOC_id)
            pes.append(pe_instance)

        mems =[]
        for el in mem_allocation:
            if el == -1:
                mems.append(-1)
                continue
            mem_name = mem_number_name_encoding[el]
            mem_ = dse_hndlr.database.get_block_by_name(mem_name)
            mem_instance = dse_hndlr.database.sample_similar_block(mem_)
            ordered_SOCsL = sorted(dse_hndlr.database.SOCsL, key=lambda SOC: SOC.get_budget("latency"))
            mem_instance.set_SOC(ordered_SOCsL[0].type, dse_hndlr.database.SOC_id)
            mems.append(mem_instance)

        ics =[]
        for el in bus_allocation:
            if el == -1:
                ics.append(-1)
                continue
            ic_name = ic_number_name_encoding[el]
            ic_ = dse_hndlr.database.get_block_by_name(ic_name)
            ic_instance = dse_hndlr.database.sample_similar_block(ic_)
            ordered_SOCsL = sorted(dse_hndlr.database.SOCsL, key=lambda SOC: SOC.get_budget("latency"))
            ic_instance.set_SOC(ordered_SOCsL[0].type, dse_hndlr.database.SOC_id)
            ics.append(ic_instance)

        #-----------------------
        # alloction checks
        #-----------------------
        # negative encoding (none existance of a block) must be at the end
        found_negative_pe = False
        for pe in pes:
            if not(pe == -1) and found_negative_pe:
                system_validity = False
                invalidity_reason = 'pe_order'
                break
            if pe == -1:
                found_negative_pe = True

        found_negative_mem = False
        for mem in mems:
            if not(mem== -1) and found_negative_mem:
                system_validity = False
                invalidity_reason = 'mem_order'
                break
            if mem == -1:
                found_negative_mem = True

        found_negative_ic = False
        for ic in ics:
            if not(ic== -1) and found_negative_ic:
                system_validity = False
                invalidity_reason = 'ic_order'
                break
            if ic == -1:
                found_negative_ic = True

        # at least one block (from each block tier (pe, mem, bus) must exist)
        if all([pe == - 1 for pe in pes]):
            system_validity = False
            invalidity_reason = 'no_pes'
        if all([mem == - 1 for mem in mems]):
            invalidity_reason = 'no_mems'
            system_validity = False
        if all([ic == - 1 for ic in ics]):
            invalidity_reason = 'no_ics'
            system_validity = False


        #-----------------------
        # topology checks
        #-----------------------
        for idx, connected_bus_idx in enumerate(pe_to_bus_connection):
            bus_exist = not(ics[connected_bus_idx] == -1)
            pe_exist = not(pes[idx] == -1)
            connection_exist = not(connected_bus_idx == -1)

            if not connection_exist:
                continue
            if (not(bus_exist) or not(pe_exist)) and connection_exist:
                system_validity = False
                invalidity_reason = 'connection_without_allocation'
                break
            elif (not(bus_exist) or not(pe_exist))  and not(connection_exist):
                continue
            pes[idx].connect(ics[connected_bus_idx])

        for idx, connected_bus_idx in enumerate(bus_to_bus_connection):
            bus_exist = not(ics[idx] == -1)
            connected_bus_exist = not(ics[connected_bus_idx] == -1)
            connection_exist = not(connected_bus_idx == -1)
            if not connection_exist:
                continue
            if (not(bus_exist) or not(connected_bus_exist)) and connection_exist:
                system_validity = False
                invalidity_reason = 'connection_without_allocation'
                break
            elif (not(bus_exist) or not(connected_bus_exist))  and not(connection_exist):
                continue
            elif idx == connected_bus_idx:
                system_validity = False
                invalidity_reason = 'bus_to_itself_connection'
            ics[idx].connect(ics[connected_bus_idx])

        for idx, connected_mem_idx in enumerate(bus_to_mem_connection):
            # system checks
            bus_exist = not(ics[idx] == -1)
            mem_exist = not(mems[connected_mem_idx] == -1)
            connection_exist = not(connected_mem_idx == -1)
            if not connection_exist:
                continue
            if (not(bus_exist) or not(mem_exist)) and connection_exist:
                system_validity = False
                invalidity_reason = 'connection_without_allocation'
                break
            elif  (not(bus_exist) or not(mem_exist))  and not(connection_exist):
                continue
            ics[idx].connect(mems[connected_mem_idx])

        if all([connection == - 1 for connection in bus_to_mem_connection]):
            system_validity = False
            invalidity_reason = 'no_bus_is_connected_to_mem'

        # each memory can only have one ic neighbour
        for mem in mems:
            if mem == -1:
                continue
            ics_ = [el for el in mem.get_neighs() if el.type =="ic"]
            if len(ics_) > 1:
                system_validity = False
                invalidity_reason = 'mem_multiple_neighbour'
                break

        if not system_validity:
            return False, invalidity_reason

        #-----------------------
        # mapping checks
        #-----------------------
        tasks =[]
        for task_idx in range(0, len(task_to_pe_mapping)):
            task_name = task_number_name_encoding[task_idx]
            task_instance_ = dse_hndlr.database.get_task_by_name(task_name)
            tasks.append(task_instance_)

        for task_idx, pe_mapping in enumerate(task_to_pe_mapping):
            # each task must be mapped to something
            if pe_mapping == -1:
                system_validity = False
                invalidity_reason = 'task_no_pe_mapping'
                break
            # pe must exist
            if pes[pe_mapping] == -1:
                system_validity = False
                invalidity_reason = 'task_pe_mapping_with_no_allocation'
                break
            pe = pes[pe_mapping]
            #get_work_ratio = self.database.get_block_work_ratio_by_task_dir
            task = tasks[task_idx]

            # check pe compatibility
            compatible_blocks = [el.instance_name_without_id for el in dse_hndlr.database.find_all_compatible_blocks("pe", [task])]
            if not pe.instance_name_without_id in compatible_blocks:
                system_validity = False
                invalidity_reason = 'task_pe_compatibility'
                break
            pe.load_improved(task, task)

        for task_idx, mem_mapping in enumerate(task_to_mem_mapping):
            # each task must be mapped to something
            if mem_mapping == -1:
                invalidity_reason = 'task_no_mem_mapping'
                system_validity = False
                break
            if mems[mem_mapping] == -1:
                invalidity_reason = 'task_mem_mapping_with_no_allocation'
                system_validity = False
                break
            mem = mems[mem_mapping]
            #get_work_ratio = self.database.get_block_work_ratio_by_task_dir
            task = tasks[task_idx]
            for task_child in task.get_children():
                mem.load_improved(task, task_child)  # load memory with tasks

        return system_validity, invalidity_reason



    def random_walk_FARSI_array_style(self,  env, encoding_dictionary, check_system = False):

        if not check_system:
            action = [random.choice(range(el[0], el[1])) for el in zip(encoding_dictionary["encoding_flattened_lb"], encoding_dictionary["encoding_flattened_ub"])]
        else:
            system_valid = False
            while not system_valid:
                pe_allocation = [random.choice(range(encoding_dictionary["pe_allocation_lb"][idx], encoding_dictionary["pe_allocation_ub"][idx])) for idx,_ in enumerate(encoding_dictionary["pe_allocation_ub"])]
                mem_allocation = [random.choice(range(encoding_dictionary["mem_allocation_lb"][idx], encoding_dictionary["mem_allocation_ub"][idx])) for idx, _ in enumerate(encoding_dictionary["mem_allocation_ub"])]
                bus_allocation = [random.choice(range(encoding_dictionary["mem_allocation_lb"][idx], encoding_dictionary["bus_allocation_ub"][idx])) for idx,_ in enumerate(encoding_dictionary["bus_allocation_ub"])]

                pe_to_bus_connection = [random.choice(range(encoding_dictionary["pe_to_bus_connection_lb"][idx], encoding_dictionary["pe_to_bus_connection_ub"][idx])) for idx,_ in enumerate(encoding_dictionary["pe_to_bus_connection_ub"])]
                bus_to_bus_connection = [random.choice(range(encoding_dictionary["bus_to_bus_connection_lb"][idx],encoding_dictionary["bus_to_bus_connection_ub"][idx])) for idx,_ in enumerate(encoding_dictionary["bus_to_bus_connection_ub"])]
                bus_to_mem_connection= [random.choice(range(encoding_dictionary["bus_to_mem_connection_lb"][idx], encoding_dictionary["bus_to_mem_connection_ub"][idx])) for idx,_ in enumerate(encoding_dictionary["bus_to_mem_connection_ub"])]

                task_to_pe_mapping = [random.choice(range(encoding_dictionary["task_to_pe_mapping_lb"][idx], encoding_dictionary["task_to_pe_mapping_ub"][idx])) for idx,_ in enumerate(encoding_dictionary["task_to_pe_mapping_ub"])]
                task_to_mem_mapping = [random.choice(range(encoding_dictionary["task_to_mem_mapping_lb"][idx], encoding_dictionary["task_to_mem_mapping_ub"][idx])) for idx,_ in enumerate(encoding_dictionary["task_to_mem_mapping_ub"])]

                # check validity
                system_valid, invalidity_reason = self.run_system_checks(env, encoding_dictionary, pe_allocation, mem_allocation, bus_allocation, pe_to_bus_connection, bus_to_bus_connection, bus_to_mem_connection, task_to_pe_mapping, task_to_mem_mapping)


                if system_valid:
                    action = []
                    action.extend(pe_allocation)
                    action.extend(mem_allocation)
                    action.extend(bus_allocation)

                    action.extend(pe_to_bus_connection)
                    action.extend(bus_to_bus_connection)
                    action.extend(bus_to_mem_connection)
                    action.extend(task_to_pe_mapping)
                    action.extend(task_to_mem_mapping)
                else:
                    print("invalidty reason", invalidity_reason)
        return action


    def random_walk_FARSI_array_style_old(self,  encoding_dictionary):
        while True:
            action = [random.choice(range(el[0], el[1])) for el in zip(encoding_dictionary["encoding_lb"], encoding_dictionary["encoding_ub"])]
            bus_to_mem_connection_ub = [encoding_dictionary["encoding_ub"][idx] for idx in range(encoding_dictionary["delimiters"]["bus_to_mem_connection"][0], encoding_dictionary["delimiters"]["bus_to_mem_connection"][1])]
            lower_bound = encoding_dictionary["encoding_lb_shortcut"]
            bus_to_mem_connection = action[encoding_dictionary["delimiters"]["bus_to_mem_connection"][0]: encoding_dictionary["delimiters"]["bus_to_mem_connection"][1]+1]

            # simple check ensuring not multiple bus connection for memory
            if len(set(bus_to_mem_connection)) < len(bus_to_mem_connection):
                continue
            else:
                break
        return action


    def random_walk_FARSI(self, env):
        dse = env.dse_hndlr.dse
        dse_hndlr = env.dse_hndlr
        database = dse.database
        boost_SOC = 1
        cur_ex_dp = env.cur_ex_dp
        cur_sim_dp = env.cur_sim_dp

        new_ex_dp = cPickle.loads(cPickle.dumps(cur_ex_dp, -1))

        # exploration does one simple sampling
        dse_hndlr.prepare_for_exploration(boost_SOC, "FARSI_des_passed_in", new_ex_dp)
        des_tup = [new_ex_dp, cur_sim_dp]


        safety_chk_passed = False
        # iterate and continuously generate moves, until one passes some sanity check
        while not safety_chk_passed:
            move_to_try, total_transformation_cnt = dse.sel_moves(des_tup, "dist_rank")
            safety_chk_passed = move_to_try.safety_check(new_ex_dp)
            move_to_try.populate_system_improvement_log()

        ##move_to_try, total_transformation_cnt = dse.sel_moves(dse_tup, "dist_rank")
        return move_to_try
        #des_tup_new, possible_des_cnt = self.gen_neigh_and_eval(des_tup)
        #dse_handler.explore()
    
    
    def action_decoder_rl(self, act_encoded, rl_form):
        """
        Decode the action space for the RL agent
        """
        print("[Action Encoded]", act_encoded)
        act_decoded = {}
        
        # simle Encoding for string action space for memory controller
        page_policy_mapper = {0:"Open", 1:"OpenAdaptive", 2:"Closed", 3:"ClosedAdaptive"}
        scheduler_mapper = {0:"Fifo", 1:"FrFcfsGrp", 2:"FrFcfs"}
        schedulerbuffer_mapper = {0:"Bankwise", 1:"ReadWrite", 2:"Shared"}
        request_buffer_size_mapper = {0:1, 1:2, 2:4, 3:8, 4:16, 5:32, 6:64, 7:128}
        respqueue_mapper = {0:"Fifo", 1:"Reorder"}
        refreshpolicy_mapper = {0:"NoRefresh", 1:"AllBank"}
        refreshmaxpostponed_mapper = {0:1, 1:2, 2:4, 3:8}
        refreshmaxpulledin_mapper = {0:1, 1:2, 2:4, 3:8}
        arbiter_mapper = {0:"Simple", 1:"Fifo", 2:"Reorder"}
        max_active_transactions_mapper = {0:1, 1:2, 2:4, 3:8, 4:16, 5:32, 6:64, 7:128}

        if(rl_form == 'sa' or rl_form == 'macme_continuous'):
            act_decoded["PagePolicy"] =  page_policy_mapper[self.action_mapper(act_encoded[0], page_policy_mapper)]
            act_decoded["Scheduler"]  =  scheduler_mapper[self.action_mapper(act_encoded[1], scheduler_mapper)]
            act_decoded["SchedulerBuffer"]  =  schedulerbuffer_mapper[self.action_mapper(act_encoded[2], schedulerbuffer_mapper)]
            act_decoded["RequestBufferSize"]  =  request_buffer_size_mapper[self.action_mapper(act_encoded[3], 
                                                                                        request_buffer_size_mapper)]
            act_decoded["RespQueue"]  =  respqueue_mapper[self.action_mapper(act_encoded[4], respqueue_mapper)]
            act_decoded["RefreshPolicy"]  =  refreshpolicy_mapper[self.action_mapper(act_encoded[5], refreshpolicy_mapper)]
            act_decoded["RefreshMaxPostponed"]  =  refreshmaxpostponed_mapper[self.action_mapper(act_encoded[6],
                                                                                        refreshmaxpostponed_mapper)]
            act_decoded["RefreshMaxPulledin"]  =  refreshmaxpulledin_mapper[self.action_mapper(act_encoded[7], 
                                                                                            refreshmaxpulledin_mapper)]
            act_decoded["Arbiter"] =  arbiter_mapper[self.action_mapper(act_encoded[8], arbiter_mapper)]
            act_decoded["MaxActiveTransactions"] =  max_active_transactions_mapper[self.action_mapper(act_encoded[9],
                                                                                    max_active_transactions_mapper)]
        elif (rl_form == 'macme'):
            print("[Action Decoder]", act_encoded)
            
            act_decoded["PagePolicy"] =  page_policy_mapper[act_encoded[0]]
            act_decoded["Scheduler"]  =  scheduler_mapper[act_encoded[1]]
            act_decoded["SchedulerBuffer"]  =  schedulerbuffer_mapper[act_encoded[2]]
            act_decoded["RequestBufferSize"]  =  request_buffer_size_mapper[act_encoded[3]]
            act_decoded["RespQueue"]  =  respqueue_mapper[act_encoded[4]]
            act_decoded["RefreshPolicy"]  =  refreshpolicy_mapper[act_encoded[5]]
            act_decoded["RefreshMaxPostponed"]  =  refreshmaxpostponed_mapper[act_encoded[6]]
            act_decoded["RefreshMaxPulledin"]  =  refreshmaxpulledin_mapper[act_encoded[7]]
            act_decoded["Arbiter"] =  arbiter_mapper[act_encoded[8]]
            act_decoded["MaxActiveTransactions"] =  max_active_transactions_mapper[act_encoded[9]]
        elif(rl_form == 'tdm'):
            print("[Action Decoder]", act_encoded)
            
            act_decoded["PagePolicy"] =  page_policy_mapper[np.clip(act_encoded[0], 0, len(page_policy_mapper)-1)]
            act_decoded["Scheduler"]  =  scheduler_mapper[np.clip(act_encoded[1], 0, len(scheduler_mapper)-1)]
            act_decoded["SchedulerBuffer"]  =  schedulerbuffer_mapper[np.clip(act_encoded[2], 0, len(schedulerbuffer_mapper)-1)]
            act_decoded["RequestBufferSize"]  =  request_buffer_size_mapper[np.clip(act_encoded[3], 0, len(request_buffer_size_mapper)-1)]
            act_decoded["RespQueue"]  =  respqueue_mapper[np.clip(act_encoded[4], 0, len(respqueue_mapper)-1)]
            act_decoded["RefreshPolicy"]  =  refreshpolicy_mapper[np.clip(act_encoded[5], 0, len(refreshpolicy_mapper)-1)]
            act_decoded["RefreshMaxPostponed"]  =  refreshmaxpostponed_mapper[np.clip(act_encoded[6], 0, len(refreshmaxpostponed_mapper)-1)]
            act_decoded["RefreshMaxPulledin"]  =  refreshmaxpulledin_mapper[np.clip(act_encoded[7], 0, len(refreshmaxpulledin_mapper)-1)]
            act_decoded["Arbiter"] =  arbiter_mapper[np.clip(act_encoded[8], 0, len(arbiter_mapper)-1)]
            act_decoded["MaxActiveTransactions"] =  max_active_transactions_mapper[np.clip(act_encoded[9], 0, len(max_active_transactions_mapper)-1)]
        else:
            print("Invalid RL form")
            sys.exit()
        print("[Action Decoder]", act_decoded)
        return act_decoded


    def action_decoder_ga(self, act_encoded):
        print(act_encoded)
        act_decoded = {}
         # simle Encoding for string action space for memory controller
        page_policy_mapper = {0:"Open", 1:"OpenAdaptive", 2:"Closed", 3:"ClosedAdaptive"}
        scheduler_mapper = {0:"Fifo", 1:"FrFcfsGrp", 2:"FrFcfs"}
        schedulerbuffer_mapper = {0:"Bankwise", 1:"ReadWrite", 2:"Shared"}
        respqueue_mapper = {0:"Fifo", 1:"Reorder"}
        refreshpolicy_mapper = {0:"NoRefresh", 1:"AllBank"}#, 2:"PerBank", 3:"SameBank"}
        arbiter_mapper = {0:"Simple", 1:"Fifo", 2:"Reorder"}
        
        act_decoded["PagePolicy"] =  page_policy_mapper[int(act_encoded[0])]
        act_decoded["Scheduler"]  =  scheduler_mapper[int(act_encoded[1])]
        act_decoded["SchedulerBuffer"]  =  schedulerbuffer_mapper[int(act_encoded[2])]
        act_decoded["RequestBufferSize"]  =  int(act_encoded[3])
        act_decoded["RespQueue"]  =  respqueue_mapper[int(act_encoded[4])]
        act_decoded["RefreshPolicy"]  =  refreshpolicy_mapper[int(act_encoded[5])]
        act_decoded["RefreshMaxPostponed"]  =  int(act_encoded[6])
        act_decoded["RefreshMaxPulledin"]  =  int(act_encoded[7])
        act_decoded["Arbiter"]  =  arbiter_mapper[int(act_encoded[8])]
        act_decoded["MaxActiveTransactions"]  =  int(act_encoded[9])

        return act_decoded

    
    def action_decoder_ga_astraSim(self, act_encoded):
        act_decoded = {"network": {}, "system": {}}
        # Network decoding
        topologyName_mapper = {0:"Hierarchical"}
        topologiesPerDim_mapper = {0:"Ring", 1:"FullyConnected", 2:"Switch"}
        dimensionType_mapper = {0:"N", 1:"P", 2:"T"}

        # System decoding
        schedulePolicy_mapper = {0: "LIFO", 1: "FIFO"}
        implementation_mapper = {0: "Ring", 1: "direct", 2: "doubleBinaryTree", 3: "oneRing", 4: "oneDirect"}
        collectiveOptimization_mapper = {0: "baseline", 1: "localBWAware"}

        # Link Counts: 
        links_count = {"Ring": 2, "FullyConnected": 7, "Switch": 1}
        print(f"act_encoded: {act_encoded}")

        # hardcoded, except link count
        act_decoded["network"]["topology-name"] = "Hierarchical"
        act_decoded["network"]["nic-latency"] = [0, 0, 0]
        act_decoded["network"]["dimensions-count"] = 3
        act_decoded["system"]["active-chunks-per-dimension"] = 1


        # NETWORK: 0-2) topologies per dim, 3-5) dimension type
        act_decoded["network"]["topologies-per-dim"] = [topologiesPerDim_mapper[int(act_encoded[0])], 
                                topologiesPerDim_mapper[int(act_encoded[1])], topologiesPerDim_mapper[int(act_encoded[2])]]
        act_decoded["network"]["dimension-type"] = [dimensionType_mapper[int(act_encoded[3])], dimensionType_mapper[int(act_encoded[4])], dimensionType_mapper[int(act_encoded[5])]]

        # 6-8) unitsCount, 9-11) linkLatency
        act_decoded["network"]["units-count"] = [int(act_encoded[6]), int(act_encoded[7]), int(act_encoded[8])]
        act_decoded["network"]["link-latency"] = [int(act_encoded[9]), int(act_encoded[10]), int(act_encoded[11])]

        # Links Count
        act_decoded["network"]["links-count"] = [int(links_count[act_decoded["network"]["topologies-per-dim"][i]])
                        for i in range(act_decoded["network"]["dimensions-count"])]

        # 12-14) linkBandwidth, 15-17) routerLatency
        act_decoded["network"]["link-bandwidth"] = [int(act_encoded[12]), int(act_encoded[13]), int(act_encoded[14])]
        act_decoded["network"]["router-latency"] = [int(act_encoded[15]), int(act_encoded[16]), int(act_encoded[17])]

        # 18-20) hbmLatency, 21-23) hbmBandwidth, 24-26) hbmScale
        act_decoded["network"]["hbm-latency"] = [int(act_encoded[18]), int(act_encoded[19]), int(act_encoded[20])]
        act_decoded["network"]["hbm-bandwidth"] = [int(act_encoded[21]), int(act_encoded[22]), int(act_encoded[23])]
        act_decoded["network"]["hbm-scale"] = [int(act_encoded[24]), int(act_encoded[25]), int(act_encoded[26])]

        # SYSTEM: 27) schedulingPolicy, 28) endpointDelay
        act_decoded["system"]["scheduling-policy"] = schedulePolicy_mapper[int(act_encoded[27])]
        act_decoded["system"]["endpoint-delay"] = int(act_encoded[28])
        
        # 29) preferredDatasetSplits, 30) boostMode
        act_decoded["system"]["preferred-dataset-splits"] = int(act_encoded[29])
        act_decoded["system"]["boost-mode"] = int(act_encoded[30])

        # 31-33) allReduceImplementation
        act_decoded["system"]["all-reduce-implementation"] = implementation_mapper[int(act_encoded[31])] + "_" + implementation_mapper[int(act_encoded[32])] + "_" + implementation_mapper[int(act_encoded[33])]
        # 34-36) allGatherImplementation
        act_decoded["system"]["all-gather-implementation"] = implementation_mapper[int(act_encoded[34])] + "_" + implementation_mapper[int(act_encoded[35])] + "_" + implementation_mapper[int(act_encoded[36])]
        # 37-39) reduceScatterImplementation
        act_decoded["system"]["reduce-scatter-implementation1"] = implementation_mapper[int(act_encoded[37])] + "_" + implementation_mapper[int(act_encoded[38])] + "_" + implementation_mapper[int(act_encoded[39])]
        # 40-42) allToAllImplementation
        act_decoded["system"]["all-to-all-implementation"] = implementation_mapper[int(act_encoded[40])] + "_" + implementation_mapper[int(act_encoded[41])] + "_" + implementation_mapper[int(act_encoded[42])]
        # 43) collectiveOptimization
        act_decoded["system"]["collective-optimization"] = collectiveOptimization_mapper[int(act_encoded[43])]

        return act_decoded

        # action_dict = {"system": {}, "network": {}}
        # action_dict["network"]["topology-name"] = "Hierarchical"
        # action_dict["network"]["topologies-per-dim"] = ["Ring", "Ring", "Ring"]
        # action_dict["network"]["dimension-type"] = ["N", "N", "N"]
        # # DIMENSION COUNT MUST BE SET TO 3 FOR NOW
        # action_dict["network"]["dimensions-count"] = 3  
        # action_dict["network"]["units-count"] = [4, 4, 4]
        # action_dict["network"]["link-latency"] = [1, 1, 1]
        # action_dict["network"]["link-bandwidth"] = [32, 16, 16]
        # action_dict["network"]["nic-latency"] = [0, 0, 0]
        # action_dict["network"]["router-latency"] = [0, 0, 0]
        # action_dict["network"]["hbm-latency"] = [500, 500, 500]
        # action_dict["network"]["hbm-bandwidth"] = [370, 370, 370]
        # action_dict["network"]["hbm-scale"] = [0, 0, 0]
        # action_dict["network"]["links-count"] = [2, 2, 2]

        # # system attributes
        # action_dict["system"]["scheduling-policy"] = "LIFO"
        # action_dict["system"]["endpoint-delay"] = 1
        # action_dict["system"]["active-chunks-per-dimension"] = 1
        # action_dict["system"]["preferred-dataset-splits"] = 4
        # action_dict["system"]["boost-mode"] = 0
      
        # action_dict["system"]["all-reduce-implementation"] = "ring_ring_ring"
        # action_dict["system"]["all-gather-implementation"] = "ring_ring_ring"
        # action_dict["system"]["reduce-scatter-implementation"] = "ring_ring_ring"
        # action_dict["system"]["all-to-all-implementation"] = "ring_ring_ring"
        # action_dict["system"]["collective-optimization"] = "baseline"

        # return action_dict


    def random_walk(self):
        '''
                configurations are ordered in this fashion

                keys = ["PagePolicy", "Scheduler", "SchedulerBuffer", "RequestBufferSize", 
                "CmdMux", "RespQueue", "RefreshPolicy", "RefreshMaxPostponed", 
                "RefreshMaxPulledin", "PowerDownPolicy", "Arbiter", "MaxActiveTransactions" ]
        '''

        pagepolicy = random.randint(0,3)
        scheduler = random.randint(0,2)
        schedulerbuffer = random.randint(0,2)
        reqest_buffer_size = random.randint(1,8)
        respqueue = random.randint(0,1)
        refreshpolicy = random.randint(0,1)
        refreshmaxpostponed = random.randint(1,8)
        refreshmaxpulledin = random.randint(1,8)
        powerdownpolicy = random.randint(0,2)
        arbiter = random.randint(0,2)
        maxactivetransactions = random.randint(1,128)
        #max_buffer_depth = 128
        
        #rand_idx = random.randint(1,math.log2(max_buffer_depth))
        #maxactivetransactions = int(pow(2,rand_idx))

        rand_actions = [pagepolicy, scheduler, schedulerbuffer, reqest_buffer_size,
                        respqueue, refreshpolicy, refreshmaxpostponed, 
                        refreshmaxpulledin, arbiter, maxactivetransactions]

        #rand_actions_decoded = self.action_decoder(rand_actions)

        return rand_actions
    
    def read_modify_write_dramsys(self, action):
        print("[envHelpers][Action]", action)
        op_success = False
        mem_ctrl_file = arch_gym_configs.dram_mem_controller_config_file
        
        try:
            with open (mem_ctrl_file, "r") as JsonFile:
                data = json.load(JsonFile)
                data['mcconfig']['PagePolicy'] = action['PagePolicy']
                data['mcconfig']['Scheduler'] = action['Scheduler']
                data['mcconfig']['SchedulerBuffer'] = action['SchedulerBuffer']
                data['mcconfig']['RequestBufferSize'] = action['RequestBufferSize']
                data['mcconfig']['RespQueue'] = action['RespQueue']
                data['mcconfig']['RefreshPolicy'] = action['RefreshPolicy']
                data['mcconfig']['RefreshMaxPostponed'] = action['RefreshMaxPostponed']
                data['mcconfig']['RefreshMaxPulledin'] = action['RefreshMaxPulledin']
                data['mcconfig']['Arbiter'] = action['Arbiter']
                data['mcconfig']['MaxActiveTransactions'] = action['MaxActiveTransactions']

                with open (mem_ctrl_file, "w") as JsonFile:
                    json.dump(data,JsonFile)
                op_success = True
        except Exception as e:
            print(str(e))
            op_success = False
        return op_success

    def writemem_ctrlr(self,action_dict):
        mem_ctrl_filename = arch_gym_configs.dram_mem_controller_config_file
        write_success = False
        full_path = os.path.join(self.mem_control_basepath,mem_ctrl_filename)
        mcconfig_dict = {}
        mcconfig_dict ["mcconfig"] = action_dict
        jsonString = json.dumps(mcconfig_dict)
        
        try:
            jsonFile = open(full_path, "w")
            jsonFile.write(jsonString)
            jsonFile.close()
            write_success = True
        except Exception as e:
            print(str(e))
            write_success = False

        return write_success
    
    def read_modify_write_sniper_config(self,action_dict, cfg):
        write_success = False
        parser = configparser.ConfigParser()
        parser.read(cfg)
        print(action_dict)
        parser.set("perf_model/core/interval_timer", "dispatch_width", str(action_dict["core_dispatch_width"]))
        parser.set("perf_model/core/interval_timer", "window_size", str(action_dict["core_window_size"]))
        parser.set("perf_model/core/rob_timer", "outstanding_loads",str(action_dict["core_outstanding_loads"]))
        parser.set("perf_model/core/rob_timer", "outstanding_stores", str(action_dict["core_outstanding_stores"]))
        parser.set("perf_model/core/rob_timer", "commit_width", str(action_dict["core_commit_width"]))
        parser.set("perf_model/core/rob_timer", "rs_entries", str(action_dict["core_rs_entries"]))
        parser.set("perf_model/l1_icache", "cache_size", str(action_dict["l1_icache_size"]))
        parser.set("perf_model/l1_dcache", "cache_size", str(action_dict["l1_dcache_size"]))
        parser.set("perf_model/l2_cache", "cache_size", str(action_dict["l2_cache_size"]))
        parser.set("perf_model/l3_cache", "cache_size", str(action_dict["l3_cache_size"]))
        try:
            with open(cfg,'w') as configfile:
                parser.write(configfile)
                write_success = True
        except Exception as e:
            print(str(e))
            write_success = False
        
        return write_success
    
    def create_agent_configs(self,agent_ids, cfg):
        
        shutil.copy(cfg, 'arch_gym_x86_agent_{}.cfg'.format(agent_ids))

        # return absolute paths to the config files
        return os.path.abspath('arch_gym_x86_agent_{}.cfg'.format(agent_ids))
    
    def decode_timeloop_action(self, action):
        '''Transforms action indexes to action dictionary yaml accepted by timeloop'''
        new_arch_params = self.timeloop_param_obj.get_arch_param_template()
        all_params = self.timeloop_param_obj.get_all_params()
        it = 0

        # Assuming ordered dict behavior (insertion order) in python 3.6+
        for param in all_params.keys():
            if isinstance(all_params[param], dict):
                for subparam in all_params[param].keys():
                    new_arch_params[param][subparam] = all_params[param][subparam][int(
                        action[it] - 1)]

                    # fix for block-size and word bits
                    if subparam == 'block-size':
                        if 'memory_width' in new_arch_params[param].keys():
                            new_arch_params[param]['memory_width'] = int(
                                new_arch_params[param]['block-size']) * int(new_arch_params[param]['word-bits'])
                        elif 'width' in new_arch_params[param].keys():
                            # fix for dummy buffer width parameter
                            new_arch_params[param]['width'] = int(
                                new_arch_params[param]['block-size']) * int(new_arch_params[param]['word-bits'])

                    it += 1
            else:
                new_arch_params[param] = all_params[param][int(action[it] - 1)]
                it += 1

        return new_arch_params

    
    def create_timeloop_dirs(self, agent_id, base_script_dir, base_output_dir, 
                             base_arch_dir):
        '''Creates the script, output and arch directories for a given agent_id for timeloop'''
        script_dir_agent = base_script_dir + "/" + str(agent_id)
        output_dir_agent = base_output_dir + "/" + str(agent_id)
        arch_dir_agent = base_arch_dir + "/" + str(agent_id)

        src_script_path = base_script_dir + "/run_timeloop.sh"
        arch_yaml_path = base_arch_dir + "/eyeriss_like.yaml"
        arch_comp_path = base_arch_dir + "/components"
        arch_dest_path = arch_dir_agent + "/components"

        os.makedirs(script_dir_agent, exist_ok=True)
        shutil.copy(src_script_path, script_dir_agent)
        os.makedirs(output_dir_agent, exist_ok=True)
        os.makedirs(arch_dir_agent, exist_ok=True)
        shutil.copy(arch_yaml_path, arch_dir_agent)
        shutil.copytree(arch_comp_path, arch_dest_path)

        return script_dir_agent, output_dir_agent, arch_dir_agent
    
    def remove_dirs(self, dirs):
        '''Removes a list of paths'''
        for path in dirs:
            shutil.rmtree(path)

    def compute_area_maestro(self, num_pe, l1_size, l2_size):
        MAC_AREA_MAESTRO=4470
        L2BUF_AREA_MAESTRO = 4161.536
        L1BUF_AREA_MAESTRO = 4505.1889
        L2BUF_UNIT = 32768
        L1BUF_UNIT = 64
        area = num_pe * MAC_AREA_MAESTRO + ceil(int(l2_size)/L2BUF_UNIT)*L2BUF_AREA_MAESTRO + ceil(int(l1_size)/L1BUF_UNIT)*L1BUF_AREA_MAESTRO * num_pe
        return area
    
    def reset(self):
        # if any csv and m files exists then remove the *.csv file and *.m files

        # get the file path
        file_path = os.path.dirname(os.path.realpath(__file__))
        print(file_path)
        results_file = os.path.join(file_path, self.mapping_file+".csv")
        mapping_file = os.path.join(file_path, self.mapping_file+".m")
        
        # clean up the results and mapping files
        if os.path.exists(results_file):
            print("csv file exists")
            os.remove(results_file)
        if os.path.exists(mapping_file):
            print("m file exists")
            os.remove(mapping_file)
        # return the initial state

        return np.zeros(self.observation_space.shape)

    def decode_cluster(self, idx):
        decoder = {0:'K', 1:'C', 2:'X', 3:'Y'}

        return decoder[idx]

    def decode_action_list(self, action):
        
        print("Action: {}".format(action))
        # convert all the values to int

        if len(action) == 1:
            action = [int(i) for i in action[-1]]
        else:
            action = [int(i) for i in action]
        # l2-S, l2-R, l2-K, l2-C, l2-X, l2-Y
        seed_l2 = action[0]
        seed_l1 = action[-2]
        if (action[-1]<=1):
            num_pe = 2
        else:
            num_pe = action[-1]
        print("Number of PE: {}".format(num_pe))
        print("P:", action)
        l1_df = [['S', action[2]], ['R', action[3]], ['K', action[4]], ['C', action[5]], ['X', action[6]], ['Y', action[7]]]
        

        l2_df = [['S', action[9]], 
            ['R', action[10]], 
            ['K', np.random.randint(1, action[4]) if action[4] > 1 else 1],
            ['C', np.random.randint(1, action[5]) if action[5] > 1 else 1],
            ['X', np.random.randint(1, action[6]) if action[6] > 1 else 1],
            ['Y', np.random.randint(1, action[7]) if action[7] > 1 else 1]]

        # permute the l1_df based on seed_l1
        np.random.seed(seed_l1)
        np.random.shuffle(l1_df)
        
        # permute the l2_df based on seed_l2
        np.random.seed(seed_l2)
        np.random.shuffle(l2_df)
        
        # convert l1_df to dictionary and l2_df to dictionary
        l1_dict = dict(l1_df)
        l2_dict = dict(l2_df)

        # get the cluster
        if (num_pe <= 1):
            num_pe = 2
            parallel_dim_l2 = [str(self.decode_cluster(action[8])), 1]
        else:
            parallel_dim_l2 = [str(self.decode_cluster(action[8])), np.random.randint(1, num_pe)]
        if (l2_dict[self.decode_cluster(action[1])] <= 1):
            print("Fix this!",l2_dict[self.decode_cluster(action[1])])
            parallel_dim_l1 = [str(self.decode_cluster(action[1])), 1]
        else:
            parallel_dim_l1 = [str(self.decode_cluster(action[1])), np.random.randint(1, l2_dict[self.decode_cluster(action[1])])]

        
        # append parallel_dim_l1 to l1_df at the beginning of the list
        l1_df.insert(0, parallel_dim_l1)
        l2_df.insert(0, parallel_dim_l2)

        
        final_df = l2_df + l1_df
        
        return final_df

    def get_dimensions(self, workload, layer_id):
        # add .csv to the workload name
        model_name = workload + ".csv"
        model_path = os.path.join(arch_gym_configs.mastero_model_path, model_name)
        
        # check if model_path exists
        if os.path.exists(model_path):
            print("model_path exists")
            import pandas as pd

            # Read in the csv file
            df = pd.read_csv(model_path)

            # Get user input for row number
            layer_id = layer_id

            # Select the row based on user input
            row = df.iloc[layer_id]

            # convert the row to dictionary
            row_dict = row.to_dict()

            # convert the dictionary to list
            row_list = list(row_dict.values())
            return row_dict, row_list

        else:
            print("model_path does not exist")
            sys.exit()

    def get_CONVtypeShape(self, dimensions, CONVtype=1):
        CONVtype_dicts = {0:"FC", 1:"CONV",2:"DSCONV", 3:"GEMM"}
        CONVtype = CONVtype_dicts[CONVtype]
        if CONVtype == "CONV"or CONVtype=="DSCONV":
            pass
        elif CONVtype == "GEMM" or CONVtype=="SGEMM":
            SzM, SzN, SzK,*a = dimensions
            dimensions = [SzN, SzK, SzM, 1, 1, 1]
        elif CONVtype == "FC":
            SzOut, SzIn, *a = dimensions
            dimensions = [SzOut, SzIn, 1, 1, 1, 1]
        else:
            print("Not supported layer.")
        return dimensions
    
    def write_maestro(self, indv=None, workload= None, layer_id= 0, m_file=None):
        _, dimension = self.get_dimensions(workload, layer_id)
        print("[DEBUG][write_maestro][dimension: {}]", dimension)
        
        m_type_dicts = {0:"CONV", 1:"CONV", 2:"DSCONV", 3:"CONV"}
        
        print("[DEBUG][write_maestro][m_file: {}]", m_file)
        dimensions = [dimension]
        with open("{}.m".format(m_file), "w") as fo:
            fo.write("Network {} {{\n".format(layer_id))
            for i in range(len(dimensions)):
                dimension = dimensions[i]
                m_type = m_type_dicts[int(dimension[-1])]
                dimension = self.get_CONVtypeShape(dimension, int(dimension[-1]))
                print(dimension)
                
                fo.write("Layer {} {{\n".format(m_type))
                fo.write("Type: {}\n".format(m_type))
                fo.write(
                    "Dimensions {{ K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(
                        *dimension))
                fo.write("Dataflow {\n")
                for k in range(0, len(indv), 7):
                    for i in range(k, k + 7):
                        if len(indv[i]) == 2:
                            d, d_sz = indv[i]
                        else:
                            d, d_sz, _ = indv[i]
                        if i % 7 == 0:
                            if k != 0:
                                fo.write("Cluster({},P);\n".format(d_sz))
                        else:
                            sp = "SpatialMap" if d == indv[k][0] or (
                                        len(indv[k]) > 2 and d == indv[k][2]) else "TemporalMap"
                            # MAESTRO cannot take K dimension as dataflow file
                            if not (m_type == "DSCONV"):
                                fo.write("{}({},{}) {};\n".format(sp, d_sz, d_sz, self.get_out_repr(d)))
                            else:
                                if self.get_out_repr(d) == "C" and self.get_out_repr(indv[k][0]) == "K":
                                    fo.write("{}({},{}) {};\n".format("SpatialMap", d_sz, d_sz, "C"))
                                else:
                                    if not (self.get_out_repr(d) == "K"):
                                        fo.write("{}({},{}) {};\n".format(sp, d_sz, d_sz, self.get_out_repr(d)))

                fo.write("}\n")
                fo.write("}\n")
            fo.write("}")

            # return the full path of the m_file
            return os.path.join(os.getcwd(), "{}.m".format(m_file))

    def get_out_repr(self, x):
        out_repr = set(["K", "C", "R", "S"])
        if x in out_repr:
            return x
        else:
            return x + "'"

    def run_maestro(self, exe, m_file, arch_configs):

        NocBW = arch_configs["NocBW"]
        offchipBW = arch_configs["offchipBW"]
        num_pe = arch_configs["num_pe"]
        l1_size = arch_configs["l1_size"]
        l2_size = arch_configs["l2_size"]
        num_pe = arch_configs["num_pe"]

        command = [exe,
           "--Mapping_file={}.m".format(m_file),
           "--full_buffer=false",
           "--noc_bw_cstr={}".format(NocBW),
           "--noc_hops=1",
           "--noc_hop_latency=1",
           "--offchip_bw_cstr={}".format(offchipBW),
           "--noc_mc_support=true",
           "--num_pes={}".format(int(num_pe)),
           "--num_simd_lanes=1",
           "--l1_size_cstr={}".format(l1_size),
           "--l2_size_cstr={}".format(l2_size),
           "--print_res=false",
           "--print_res_csv_file=true",
           "--print_log_file=false",
           "--print_design_space=false",
           "--msg_print_lv=0"]

        print(command)
        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        process.wait() 
        
        try:
            df = pd.read_csv("{}.csv".format(m_file))
            layer_name = df[" Layer Number"]
            runtime = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
            runtime_series = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
            throughput = np.array(df[" Throughput (MACs/Cycle)"]).reshape(-1, 1)
            energy = np.array(df[" Activity count-based Energy (nJ)"]).reshape(-1, 1)
            area = np.array(df[" Area"]).reshape(-1, 1)
            power = np.array(df[" Power"]).reshape(-1, 1)
            l1_size = np.array(df[" L1 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            l2_size = np.array(df["  L2 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            l1_size_series = np.array(df[" L1 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            l2_size_series = np.array(df["  L2 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            l1_input_read = np.array(df[" input l1 read"]).reshape(-1, 1)
            l1_input_write = np.array(df[" input l1 write"]).reshape(-1, 1)
            l1_weight_read = np.array(df["filter l1 read"]).reshape(-1, 1)
            l1_weight_write = np.array(df[" filter l1 write"]).reshape(-1, 1)
            l1_output_read = np.array(df["output l1 read"]).reshape(-1, 1)
            l1_output_write = np.array(df[" output l1 write"]).reshape(-1, 1)
            l2_input_read = np.array(df[" input l2 read"]).reshape(-1, 1)
            l2_input_write = np.array(df[" input l2 write"]).reshape(-1, 1)
            l2_weight_read = np.array(df[" filter l2 read"]).reshape(-1, 1)
            l2_weight_write = np.array(df[" filter l2 write"]).reshape(-1, 1)
            l2_output_read = np.array(df[" output l2 read"]).reshape(-1, 1)
            l2_output_write = np.array(df[" output l2 write"]).reshape(-1, 1)
            mac = np.array(df[" Num MACs"]).reshape(-1, 1)
            
            activity_count = {}
            activity_count["l1_input_read"] = l1_input_read
            activity_count["l1_input_write"] = l1_input_write
            activity_count["l1_weight_read"] = l1_weight_read
            activity_count["l1_weight_write"] = l1_weight_write
            activity_count["l1_output_read"] = l1_output_read
            activity_count["l1_output_write"] = l1_output_write
            activity_count["l2_input_read"] = l2_input_read
            activity_count["l2_input_write"] = l2_input_write
            activity_count["l2_weight_read"] = l2_weight_read
            activity_count["l2_weight_write"] = l2_weight_write
            activity_count["l2_output_read"] = l2_output_read
            activity_count["l2_output_write"] = l2_output_write
            activity_count["mac_activity"] = mac
            area = self.compute_area_maestro(num_pe, l1_size, l2_size)
            self.observation = [np.mean(x) for x in [runtime, throughput, energy, area, l1_size, l2_size, mac, power, num_pe]]
            
        except Exception as e:
            print(e)
            #set all the return values to -1
            runtime = np.array([1e20])
            runtime_series = -1
            throughput = np.array([-1])
            energy = np.array([-1])
            area = np.array([-1])
            power = -1
            l1_size = -1    
            l2_size = -1
            l1_size_series = -1
            l2_size_series = -1
            activity_count = -1
            mac = -1
            self.observation = [np.mean(x) for x in [runtime, throughput, energy, area, l1_size, l2_size, mac, power, num_pe]]
            print("Error in reading csv file")
        
        obs = [runtime, throughput, energy, np.array(area)]
        print("[Env Helpers][Observation: ]", obs)
        flat_obs = np.concatenate([x.flatten() for x in obs])

        # convert to numpy array
        flat_obs = np.asarray(flat_obs)
        return flat_obs
    
    def decode_action_list_multiagent (self, action_list):
        return NotImplementedError

    
    def map_to_discrete(self, action_list, discrete_values):
        discrete_action_list = []
        for i, action in enumerate(action_list):
            num_values = discrete_values[i]
            discrete_action = int(action * num_values)
            discrete_action = min(discrete_action, num_values - 1)  # Ensure the index doesn't go out of bounds
            discrete_action_list.append(discrete_action)
        return discrete_action_list

    def decode_action_list_rl (self, action_list, dimensions):

        '''
        Convert the continuous action list to a discrete action list depending upon the dimensions
        of the network layer
        '''
        print("Action List: ", action_list)
        print("Dimensions: ", dimensions)
        discrete_values = [720, 4, 2, 2, dimensions["K"], dimensions["C"], dimensions["X"],
                            dimensions["Y"], 4, 2, 2, dimensions["K"], dimensions["C"], dimensions["X"],
                              dimensions["Y"], 720, 1024]
        
        discrete_action_list = self.map_to_discrete(action_list, discrete_values)
        
        return discrete_action_list

    def generate_maestro_parameter_set(self, dimensions):
        print("Dimensions: ", dimensions)

        params = {}
        params["seed_l2"] = [i for i in range(0, 720)]
        params["ckxy_l2"] = [0, 1, 2, 3]
        params["s_l2"] = [i for i in range(dimensions["S"]-1, dimensions["S"])]
        params["r_l2"] = [i for i in range(dimensions["R"]-1, dimensions["R"])]
        params["k_l2"] = [i for i in range(1, dimensions["K"])]
        params["c_l2"] = [i for i in range(1, dimensions["C"])]
        params["x_l2"] = [i for i in range(1, dimensions["X"])]
        params["y_l2"] = [i for i in range(1, dimensions["Y"])]
        params["ckxy_l1"] = [0, 1, 2, 3]
        params["s_l1"] = [i for i in range(dimensions["S"]-1, dimensions["S"])]
        params["r_l1"] = [i for i in range(dimensions["R"]-1, dimensions["R"])]
        params["k_l1"] = [i for i in range(1, dimensions["K"])]
        params["c_l1"] = [i for i in range(1, dimensions["C"])]
        params["x_l1"] = [i for i in range(1, dimensions["X"])]
        params["y_l1"] = [i for i in range(1, dimensions["Y"])]
        params["seed_l1"] = [i for i in range(0, 720)]
        params["num_pe"] = [i for i in range(1, 1024)]

        return params

    def custom_list_representer(self, dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    def generate_aco_maestro_config(self, yaml_file, params_dict):
        write_ok = False
        print("YAML file: ", yaml_file)
        print(os.path.exists(yaml_file))
        dumper = CustomListDumper(yaml.SafeDumper)
        yaml.add_representer(list, self.custom_list_representer, Dumper=dumper)

        try:
            with open(yaml_file, 'r') as file:
                yaml_data = yaml.safe_load(file)

            # Update the ArchParamsNode attributes with new_values
            arch_params_node = yaml_data['Nodes']['ArchParamsNode']['attributes']
            
            for key, value in params_dict.items():
                if key in arch_params_node:
                    print("Key: ", key, " Value: ", value)
                    arch_params_node[key] = value

            print("YAML data: ", yaml_data)
            
            # Save the modified YAML data back to the file
            with open(yaml_file, 'w') as file:
               yaml.dump(yaml_data, file, Dumper=CustomListDumper)

            write_ok = True
        except Exception as e:
            print(e)
            write_ok = False
        return write_ok

# For testing 
if __name__ == "__main__":   
    print("Hello!")
    
    helper = helpers()
    action_dict = {}
    action_dict["core_dispatch_width"] = 8
    action_dict["core_window_size"] = 512
    action_dict["l1_icache_size"] = 128
    action_dict["l1_dcache_size"] = 128
    action_dict["l2_cache_size"] = 2048
    action_dict["l3_cache_size"] = 8192

    helper.read_modify_write_sniper_config(action_dict)
    

    
