#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import json
import os
import itertools
from settings import config
from typing import List, Tuple
from design_utils.components.workload import *
import copy
import math
import time
from collections import deque


# This class emulates hardware queues.
# At the moment, we are using the pipe class to the same thing
# so not using this class
class HWQueue():
    def __init(self, max_size, buffer_size):
        self.max_size = max_size
        self.q_data = deque()
        self.buffer_size = buffer_size

    def enqueue(self, data):
        if self.is_full():
            return False
        self.q_data.insert(0, data)
        return True

    def dequeue(self):
        if self.is_empty():
            return False
        self.q_data.pop()
        return True

    def peek(self):
        if self.is_empty():
            return False
        return self.q_data[-1]

    def size(self):
        return len(self.q_data)

    def is_empty(self):
        return (self.size() == 0)

    def is_full(self):
        total_data = 0
        for front_ in self.q_data:
            total_data += front_.total_work
        return total_data >= self.max_size

        #return self.size() == self.max_size

    def __str__(self):
        return str(self.q_data)


if config.simulation_method == "power_knobs":
    from specs import database_input_powerKnobs as database_input
elif config.simulation_method == "performance":
    from specs import database_input
else:
    raise NameError("Simulation method unavailable")


# This class emulates the behavior of a hardware block
class Block:
    id_counter = 0
    block_numbers_seen = []
    def __init__(self, db_input, hw_sampling, instance_name, type, subtype,
                 peak_work_rate_distribution, work_over_energy_distribution, work_over_area_distribution,
                 one_over_area_distribution, clock_freq, bus_width, loop_itr_cnt, loop_max_possible_itr_cnt, hop_latency, pipe_line_depth,
                 leakage_power="", power_knobs="",
                 SOC_type="", SOC_id=""):
        self.db_input = db_input  # data base input
        self.__instance_name = instance_name  # name of the block instance
        self.subtype = subtype    # sub type of the block (e.g, gpp, or ip)

        # specs
        self.peak_work_rate_distribution = peak_work_rate_distribution   # peak work rate (work over time).
                                                                         # The concept of work rate
                                                                         # is different depending on the block.
                                                                         # concretely, for PE work rate is IPC
                                                                         # and for memory/buses its Bandwidth
        self.hop_latency = hop_latency
        self.pipe_line_depth = pipe_line_depth
        self.clock_freq = clock_freq
        self.bus_width = bus_width
        self.loop_itr_cnt = loop_itr_cnt
        self.loop_max_possible_itr_cnt = loop_max_possible_itr_cnt
        self.work_over_energy_distribution = work_over_energy_distribution  # work over energy
        self.work_over_area_distribution = work_over_area_distribution  # work over area
        self.one_over_area_distribution = one_over_area_distribution
        self.set_rates(hw_sampling)  # set the above rates based on the hardware sampling. Note that
                                     # the above rate can vary if they are a distribution rather than
                                     # one value
        self.leakage_power = leakage_power

        # power knobs of the block
        self.power_knobs = power_knobs
        # Indices of power knobs that can be used for the turbo mode (increase performance)
        self.turbo_power_knob_indices = []
        # Indices of power knobs that can be used for the slow-down mode (decrease performance)
        self.slow_down_power_knob_indices = []
        # divide power knobs to turbo and slow down ones
        #self.categorize_power_knobs()

        self.pipes = []  # these are the queues that connect different block
        self.pipe_clusters = []
        self.type = type  # type of the block (i.e., pe, mem, ic)
        self.neighs: List[Block] = []  # neighbours, i.e., the connected blocks
        self.__task_name_dir_list = []  #  tasks on the block

        # task_dir is the tuple specifying which task is making a request to access memory and in which direction.
        self.__tasks_dir_work_ratio: Dict[(Task, str): float] = {}  # tasks on the block and their work ratio.
        self.id = Block.id_counter
        Block.id_counter += 1
        if Block.id_counter in Block.block_numbers_seen:
            raise Exception("can not have two blocks with id:" + str(Block.id_counter))
        Block.block_numbers_seen.append(Block.id_counter)
        self.area = 0
        self.SOC_type = SOC_type
        self.SOC_id = SOC_id

        self.PA_prop_dict = {}   # Props used for PA (platform architect) design generation.
        self.PA_prop_auto_tuning_list = []   # list of variables to be auto tuned

        self.area_list = [0]  # list of areas for different task calls
        self.area_in_bytes_list = [0]  # list of areas for different task calls
        # only for memory
        self.area_task_dir_list = []
        self.task_mem_map_dict  = {}  # memory map associated with different tasks for memory
        self.system_bus_behavior = False # if true, the block is system bus

    def set_system_bus_behavior(self, qualifier):
        system_bus_behavior = qualifier

    def is_system_bus(self):
        return self.system_bus_behavior

    def get_block_bus_width(self):
        return self.bus_width

    def get_loop_itr_cnt(self):
        return self.loop_itr_cnt

    def get_loop_max_possible_itr_cnt(self):
        return self.loop_max_possible_itr_cnt


    def get_block_freq(self):
        return self.clock_freq

    def get_hop_latency(self):
        return self.hop_latency

    def get_pipe_line_depth(self):
        return self.pipe_line_depth

    # ---------------
    # Functionality:
    #   Return peak_work_rate. Note that work-rate definition varies based on the
    #   hardware type. work-rate for PE is IPS, where as work-rate for bus/mem means BW.
    # ---------------
    def get_peak_work_rate(self, pk_id=0):
        if (pk_id == 0) or (self.type != "pe"):
            return self.peak_work_rate
        else:
            (perf_change, dyn_power_change, leakage_power_change) = self.power_knobs[pk_id - 1]
            return self.peak_work_rate * perf_change

    # ---------------
    # Functionality:
    #   Return work_over_energy. Note that work definition varies based on the
    #   hardware type. work for PE is instruction, where as work for bus/mem means bytes.
    # ---------------
    def get_work_over_energy(self, pk_id=0):
        if (pk_id == 0) or (self.type != "pe"):
            return self.work_over_energy
        else:
            (perf_change, dyn_power_change, leakage_power_change) = self.power_knobs[pk_id - 1]
            # work over energy is instruction per Joules
            return (perf_change / dyn_power_change) * self.work_over_energy

    # ---------------
    # Functionality:
    #   Return work_over_power. Note that work definition varies based on the
    #   hardware type. work for PE is instruction, where as work for bus/mem means bytes.
    # ---------------
    def get_work_over_power(self, pk_id=0):
        # TODO fix this to actually use work_over_power
        return self.get_work_over_energy(pk_id)

    # ---------------
    # Functionality:
    #   Return work_over_area. Note that work definition varies based on the
    #   hardware type. work for PE is instruction, where as work for bus/mem means bytes.
    # ---------------
    def get_work_over_area(self, pk_id=0):
        if (pk_id == 0) or (self.type != "pe"):
            return self.work_over_area
        else:
            #print("Returning DVFS value")
            (perf_change, dyn_power_change, leakage_power_change) = self.power_knobs[pk_id - 1]
            # Work over area change; area is constant so only amount of work done is changed
            return perf_change * self.work_over_area

    # Return the private variable leakage energy
    # if a power_knob is used then return the leakage energy associated with that knob
    def get_leakage_power(self, pk_id=0):

        if (pk_id == 0) or (self.type != "pe"):
            return self.leakage_power
        else:
            (perf_change, dyn_power_change, leakage_power_change) = self.power_knobs[pk_id - 1]
            # leakage power change
            return leakage_power_change * self.leakage_power

    def get_peak_work_rate_distribution(self):
        return self.peak_work_rate_distribution

    def get_work_over_energy_distribution(self):
        return self.work_over_energy_distribution

    def get_work_over_area_distribution(self):
        return self.work_over_area_distribution

    def get_one_over_area_distribution(self):
        return self.one_over_area_distribution

    # get average of the distribution
    def get_avg(self, distribution_dict):
        avg = 0
        for key, value in distribution_dict.items():
            avg += key*value
        return avg

    # ---------------
    # Functionality:
    #   set rates, i.e., work_rate, work_over_energy, work_over_area
    #   Note that work definition varies based on the
    #   hardware type. work for PE is instruction, where as work for bus/mem means bytes.
    # ---------------
    def set_rates(self, hw_sampling):
        mode = hw_sampling["reduction"]
        accuracy_percentage = hw_sampling["accuracy_percentage"][self.subtype]
        if mode in ["random", "most_likely", "min", "max", "most_likely_with_accuracy_percentage"]:
            if mode == "random":
                time.sleep(.005)
                np.random.seed(datetime.now().microsecond)
                # sample the peak_work_rate
                work_rates = [work_rate for work_rate, work_rate_prob in self.get_peak_work_rate_distribution().items()]
                work_rate_probs = [work_rate_prob for work_rate, work_rate_prob in self.get_peak_work_rate_distribution().items()]
                if not(sum(work_rate_probs) == 1):
                    print("break point")
                work_rate_selected = np.random.choice(work_rates, p=work_rate_probs)
                # get the index and use it to collect other values (since there is a one to one correspondence)
                work_rate_idx = list(self.get_peak_work_rate_distribution().keys()).index(work_rate_selected)
            elif mode == "most_likely" or mode == "most_likely_with_accuracy_percentage": # used when we don't want to do statistical analysis
                # sample the peak_work_rate
                work_rates_sorted = collections.OrderedDict(sorted(self.get_peak_work_rate_distribution().items(), key=lambda kv: kv[1]))
                work_rate_selected = list(work_rates_sorted.keys())[-1]
                work_rate_idx = list(self.get_peak_work_rate_distribution().keys()).index(work_rate_selected)
            elif mode in ["min", "max"]: # used when we don't want to do statistical analysis
                # sort based on the value
                work_rates_sorted = collections.OrderedDict(sorted(self.get_peak_work_rate_distribution().items(), key=lambda kv: kv[0]))
                if mode == "min":  # worse case design, hence smallest workrate
                    work_rate_selected = list(work_rates_sorted.keys())[0]
                elif mode == 'max': # best case design
                    work_rate_selected = list(work_rates_sorted.keys())[-1]
                work_rate_idx = list(self.get_peak_work_rate_distribution().keys()).index(work_rate_selected)

            self.peak_work_rate = list(self.get_peak_work_rate_distribution().keys())[work_rate_idx]
            self.work_over_energy = list(self.get_work_over_energy_distribution().keys())[work_rate_idx]
            self.one_over_power = self.work_over_energy*(1/self.peak_work_rate)
            try:
                self.work_over_area = list(self.get_work_over_area_distribution().keys())[work_rate_idx]
                self.one_over_area = list(self.get_one_over_area_distribution().keys())[work_rate_idx]
            except:
                print("what ")
                exit(0)
        elif mode == "avg":
            self.peak_work_rate = self.get_avg(self.get_peak_work_rate_distribution())
            self.work_over_energy = self.get_avg(self.get_work_over_energy_distribution())
            self.work_over_area = self.get_avg(self.get_work_over_area_distribution())
            self.one_over_area = self.get_avg(self.get_one_over_area_distribution())
            self.one_over_power = self.work_over_energy*(1/self.peak_work_rate)
        else:
            print("mode" + mode + " is not supported for block sampling")
            exit(0)

        if mode == "most_likely_with_accuracy_percentage":
            # use the error in
            self.peak_work_rate *= accuracy_percentage['latency']
            self.work_over_energy *= accuracy_percentage['energy']
            self.work_over_area *= accuracy_percentage['area']
            self.one_over_area *=  accuracy_percentage['one_over_area']
            self.one_over_power *=  accuracy_percentage['energy']

    # -------------------------------------------
    # power-knobs related functions
    # -------------------------------------------
    # Get all the power_knob configurations available
    # each power_knob : ([(performance_change(ips/Bps), active_power_change(ipj/Bpj), leakage_power_change)])
    # for example (1.2,1.2,1.2) shows that by 20% performance increase, dynamic and leakage powers increase by 20%
    def get_power_knob_tuples(self):
        return self.power_knobs

    # Looks into power knobs and adds the one giving performance improvement > 1 to turbo mode ones
    #   and adds the ones with perf improvement <= 1 to slow down ones
    # Please note that it does add only the indices of the power_knob to either list
    def categorize_power_knobs(self):
        for power_knob_idx, power_knob in enumerate(self.power_knobs):
            if power_knob[0] > 1:
                self.turbo_power_knob_indices.append(power_knob_idx)
            elif power_knob[0] < 1:
                self.slow_down_power_knob_indices.append(power_knob_idx)
            else:
                self.turbo_power_knob_indices.append(power_knob_idx)
                self.slow_down_power_knob_indices.append(power_knob_idx)

    # return the turbo mode indices in the power knobs list
    def get_turbo_power_knob_indices(self):
        return self.turbo_power_knob_indices

    # return the slow down mode indices in the power knobs list
    def get_slow_down_power_knob_indices(self):
        return self.slow_down_power_knob_indices

    # A wrapper that gets the mode of power knob and send the list of corresponding indices
    def get_power_knob_indices(self, mode="slow_down"):
        if mode == "slow_down":
            return self.get_slow_down_power_knob_indices()
        elif mode == "turbo":
            return self.get_turbo_power_knob_indices()
        else:
            raise Exception("Power knob mode {} is not available!".format(mode))

    # -------------------------------------------
    # setters
    # -------------------------------------------
    # -----------
    # Functionality:
    #      resetting Platform architect props
    # -----------
    def reset_PA_props(self):
        self.PA_prop_dict = collections.OrderedDict()

    # -----------
    # Functionality:
    #      updating Platform architect props
    # -----------
    def update_PA_props(self, PA_prop_dict):
        self.PA_prop_dict.update(PA_prop_dict)

    def update_PA_auto_tunning_knob_list(self, prop_auto_tuning_list):
        self.PA_prop_auto_tuning_list = prop_auto_tuning_list

    # -----------
    # Functionality:
    #       assign an SOC for the block
    # Variables:
    #       SOC_type: type of SOC
    #       SOC_id: id of the SOC
    # -----------
    def set_SOC(self, SOC_type, SOC_id):
        self.SOC_type = SOC_type
        self.SOC_id = SOC_id

    # ---------------------------
    # Functionality:
    #       get the static area (for the blocks that are not sized on the run time, such as cores and NoCs. Note that memory would be size
    #       on the run time.
    #       convention is that work_over_area is 1/fix_area if you are statically sized
    # --------------------------
    def get_static_size(self):
        return float(1)/self.work_over_area

    # ---------------------------
    # Functionality:
    #       get the area in terms of byte. only for memory. Then we can convert it to mm^2 easily
    # --------------------------
    def get_area_in_bytes(self):
        if not self.type == "mem":
            print("area in byte for non-memory blocks are not defined")
            return 0
        else:
            #area = self.get_area()
            #return math.ceil(area*self.work_over_area)
            max_work = max(self.area_in_bytes_list)
            return math.ceil(max_work)
            #return area*self.db_input.misc_data["ref_mem_work_over_area"]

    # return the number of banks
    def get_num_of_banks(self):
        if self.type == "mem":
            #max_work = max(self.area_list)*self.db_input.misc_data["ref_mem_work_over_area"]
            max_work = max(self.area_in_bytes_list)
            num_of_banks = math.ceil(max_work/self.db_input.misc_data["memory_block_size"])
            return num_of_banks
        else:
            print("asking number of banks for non-memory blocks does not make sense")
            exit(0)

    """
    # get capacity of memories
    def get_capacity(self):
        if self.type == "mem":
            max_work = max(self.area_list)*self.db_input.misc_data["ref_mem_work_over_area"]
            max_work_rounded = math.ceil(max_work/self.db_input.memory_block_size)*self.db_input.memory_block_size
            return max_work_rounded
        else:
            print("asking capacity of non-memory blocks does not make sense")
            exit(0)
    """

    def set_area_directly(self, area):
        self.area_list = [area]
        self.area = area

    # used with cacti
    def update_area_energy_power_rate(self, energy_per_byte, area_per_byte):
        if self.type not in ["mem", "ic"]:
            print("should not be updateing the block values")
            exit(0)
        self.work_over_area = 1/area_per_byte
        self.work_over_energy = 1/max(energy_per_byte,.0000000001)
        self.one_over_power = self.work_over_energy/self.peak_work_rate
        self.one_over_area = self.work_over_area

    # ---------------------------
    # Functionality:
    #       get the area associated with a block.
    # --------------------------
    def get_area(self):
        if self.type == "mem":
            if not config.use_cacti:   # if using cacti, the area is already calculated (note that at the moment, area list doesn't matter for cacti)
                max_work = max(self.area_in_bytes_list)
                max_work_rounded = math.ceil(max_work/self.db_input.misc_data["memory_block_size"])*self.db_input.misc_data["memory_block_size"]
                self.area = max_work_rounded/self.work_over_area
        else:
           self.area =  max(self.area_list)
        return self.area

    def get_leakage_power_calculated_after(self):
        if self.type == "mem":
            max_work = max(self.area_list)*self.db_input.ref_mem_work_over_area
            max_work_rounded = math.ceil(max_work/self.db_input.memory_block_size)*self.db_input.memory_block_size
            self.leakage_power_calculated_after = max_work_rounded/self.db_input.ref_work_over_leakage
        else:
           self.leakage_power_calculated_after = 0
        return self.leakage_power_calculated_after


    def set_instance_name(self, name):
        self.__instance_name = name

    @property
    def instance_name_without_id(self):
        return self.__instance_name


    @property
    def instance_name(self):
        return self.__instance_name + "_"+ str(self.id)

    @instance_name.setter
    def instance_name(self, instance_name):
        self.__instance_name = instance_name

    # ---------------------------
    # Functionality:
    #       Return whether a block only contains a dummy task (i.e., source and sink tasks)
    #       Used for prunning the block later.
    # --------------------------
    def only_dummy_tasks(self):
        num_of_tasks = len(self.get_tasks_of_block())
        if num_of_tasks == 2:
            a = [task.name for task in self.get_tasks_of_block()]
            if any("souurce" in task.name for task in self.get_tasks_of_block()) and \
                    any("siink" in task.name for task in self.get_tasks_of_block()):
                return True
        elif num_of_tasks == 1:
            a = [task.name for task in self.get_tasks_of_block()]
            if any("souurce" in task.name for task in self.get_tasks_of_block()) or \
                    any("siink" in task.name for task in self.get_tasks_of_block()):
                return True
        else:
            return False

    def update_work(self, work):
        self.work_list

    # ---------------------------
    # Functionality:
    #       updating the area when more tasks are assigned to a memory block. Only used for memory.
    # Variables:
    #       area: current area
    #       task_requesting: the task is that is using the memory
    # --------------------------
    def update_area(self, area, task_requesting):  # note that bytes can be negative (if reading from the block) or positive
        if self.type == "mem":
            if area < 0: dir = 'read'
            else: dir = 'write'
            if (self.area_list[-1]+area) < -1*self.db_input.misc_data["area_error_margin"] and not config.use_cacti:
                raise Exception("memory size can not go bellow the error margin")
            #if dir == "write":
            #    self.task_mem_map_dict[task_requesting] = (hex(int(self.area_list[-1])), hex(int(self.area_list[-1]+area)))
            self.area_list.append(self.area_list[-1]+area)
            self.area_task_dir_list.append((task_requesting, dir))
        else:
            if self.only_dummy_tasks():
                area = 0
            self.area_list.append(area)


    def update_area_in_bytes(self, area_in_byte, task_requesting):  # note that bytes can be negative (if reading from the block) or positive
        if self.type == "mem":
            if area_in_byte < 0: dir = 'read'
            else: dir = 'write'
            if (self.area_in_bytes_list[-1]+area_in_byte) < -1*self.db_input.misc_data["area_error_margin"]:
                raise Exception("memory size can not go bellow the error margin")
            #if dir == "write":
            #    self.task_mem_map_dict[task_requesting] = (hex(int(self.area_in_bytes_list[-1])), hex(int(self.area_in_bytes_list[-1]+area_in_byte)))
            self.area_in_bytes_list.append(self.area_in_bytes_list[-1]+area_in_byte)
        else:
            if self.only_dummy_tasks():
                area = 0
            self.area_in_bytes_list.append(area_in_byte)


    def __deepcopy__(self, memo):
        #Block.id_counter = -1
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        #Block.id_counter += 1
        result.area_list = [0]
        result.area_in_bytes_list = [0]
        result.area_task_dir_list = []
        result.task_mem_map_dict = {}
        return result

    # each hardware block has a generic name, so it can be identified
    def get_generic_instance_name(self):
        return self.__instance_name

    # get the type name: {pe, mem, ic}
    def get_block_type_name(self):
        return self.type
    # ------
    # comparisons
    #------
    def __lt__(self, other):
        return self.peak_work_rate < other.peak_work_rate

    # ---------------------------
    # Functionality:
    #       get all task's parents/children and the work ratio between them.
    #       "dir" determines whether the family task is parent or child.
    #       "work_ratio" is the ratio between the work of the family member and the task itself.
    # --------------------------
    def get_tasks_dir_work_ratio_for_printing(self):
        temp_dict = []
        for task_dir, work_ratio in self.__tasks_dir_work_ratio.items():
            print((task_dir[0].name, task_dir[1]), work_ratio)

    # ---------------------------
    # Functionality:
    #       get all task's parents/children and the work ratio between them.
    #       "dir" determines whether the family task is parent or child.
    #       "work_ratio" is the ratio between the work of the family member and the task itself.
    # --------------------------
    def get_tasks_dir_work_ratio(self):
        return self.__tasks_dir_work_ratio

    # ---------------------------
    # Functionality:
    #       return the task if the name matches
    # --------------------------
    def get_tasks_by_name(self, task_name):
        tasks = self.get_tasks_of_block()
        for task in tasks:
            if (task.name == task_name):
                return True, task
        print("erroring out for block" + self.instance_name)
        self.get_tasks_dir_work_ratio_for_printing()
        print("task with the name of " + task_name + "is not loaded on block" + self.instance_name)
        return False, "_"

    # ---------------------------
    # Functionality:
    #      update the work_ratios. Used in jitter modeling, when a new task from the distribution is generated.
    # --------------------------
    def update_all_tasks_dir_work_ratio(self):
        for task_dir, family_tasks_name in self.__tasks_dir_work_ratio.items():
            task, dir = task_dir
            for family_task_name in family_tasks_name.keys():
                work_ratio = task.get_work_ratio_by_family_task_name(family_task_name)
                self.__tasks_dir_work_ratio[task_dir][family_task_name] = work_ratio  # where work ratio is set

        self.__task_name_dir_list = []
        for task_dir, family_tasks_name in self.__tasks_dir_work_ratio.items():
            task, dir = task_dir
            for family_task_name in family_tasks_name.keys():
                self.__task_name_dir_list.append((task.name, dir))

        # the following copule of lines for debugging
        if not len(self.__task_name_dir_list) == sum([len(el.values()) for el in self.__tasks_dir_work_ratio.values()]):
            blah =  len(self.__task_name_dir_list)
            blah2  =  sum([len(el.values()) for el in self.__tasks_dir_work_ratio.values()])
            print("this shoud not happend, so please debug")

    # get the fronts that will be/has been scheduled on the block
    # Note that front is a stream of data/instruction.
    # for PEs, it's just a bunch of instructions sliced up to chunks
    # for memories and buses, it's a stream of data that needs to move from one block (bus, memory) to
    # another block (bus, memory). This class is not used for the moment
    def get_fronts(self, mode="task_name_dir"):
        if mode == "task_name_dir":
            return self.__task_name_dir_list
        elif mode == 'task_dir_work_ratio':
            result = []
            for el in self.__tasks_dir_work_ratio.values():
                result.extend(el.keys())
            return result
        else:
            print("mode:" + mode + " not supported")
            exit(0)
    # ---------------------------
    # Functionality:
    #       get the task_dir
    # --------------------------
    def get_task_dirs_of_block(self):
        tasks_with_possible_duplicates = [task_dir for task_dir in self.__tasks_dir_work_ratio.keys()]
        # the duplicates happens cause  we can have one task that writes and reads into that block
        return list(set(tasks_with_possible_duplicates))

    # ---------------------------
    # Functionality:
    #       get the tasks of a block
    # --------------------------
    def get_tasks_of_block(self):
        tasks_with_possible_duplicates = [task_dir[0] for task_dir in self.__tasks_dir_work_ratio.keys()]
        #if len(tasks_with_possible_duplicates) == 0:
        #    print("what")
        # the duplicates happens cause  we can have one task that writes and reads into that block
        results = list(set(tasks_with_possible_duplicates))
        return results

    def get_tasks_of_block_by_dir(self, dir_):
        tasks_with_possible_duplicates = [task_dir[0] for task_dir in self.__tasks_dir_work_ratio.keys() if task_dir[1] == dir_]
        #if len(tasks_with_possible_duplicates) == 0:
        #    print("what")
        # the duplicates happens cause  we can have one task that writes and reads into that block
        results = list(set(tasks_with_possible_duplicates))
        return results


    # ---------------------------
    # Functionality:
    #       get the tasks work ratio by the family task.
    # Variables:
    #       task: the family task
    # --------------------------
    def get_task_s_work_ratio_by_task(self, task):
        blah = self.__tasks_dir_work_ratio
        work_ratio = [work_ratio for task_dir, work_ratio in self.__tasks_dir_work_ratio.items() if task_dir[0] == task]
        assert(len(work_ratio) < 3), ("task:" + str(task.name) + " can only have one or two (i.e., read and/or write) ratio")
        return work_ratio

    # ---------------------------
    # Functionality:
    #       get the task to task work.
    # Variables:
    #       task: the family task
    #       dir: the direction (read/write) of the family task.
    # --------------------------
    def get_task_s_family_by_task_and_dir(self, task, dir):
        res = []
        task_dir__work_ratio_up = [(task_dir, work_ratio) for task_dir, work_ratio in self.__tasks_dir_work_ratio.items() if task_dir[0] == task]
        for task_dir, work_ratio in task_dir__work_ratio_up:
            dir_ = task_dir[1]
            if dir_ == dir:
                for family_task_ratio in work_ratio:
                    family_task = family_task_ratio
                    res.append(family_task)

        return res

    # ---------------------------
    # Functionality:
    #       get the task to task work ratio.
    # Variables:
    #       task: the family task
    #       dir: the direction (read/write) of the family task.
    # --------------------------
    def get_task_s_work_ratio_by_task_and_dir(self, task, dir):
        res = []
        task_dir__work_ratio_tup = [(task_dir, work_ratio) for task_dir, work_ratio in self.__tasks_dir_work_ratio.items() if task_dir[0] == task]
        for task_dir, work_ratio in task_dir__work_ratio_tup:
            dir_ = task_dir[1]
            if dir_ == dir:
                #for task_, work_ratio_val in work_ratio.items()
                res.append(work_ratio)
        if (len(res) > 1):
            raise Exception('can not happen. Delete this later. for debugging now')
        return res[0]

    # ---------------------------
    # Functionality:
    #       get the task relationship
    # Variables:
    #       task: the family task
    # --------------------------
    def get_task_dir_by_task_name(self, task):
        task_dir = [task_dir for task_dir in self.__tasks_dir_work_ratio.keys() if task_dir[0].name == task.name]
        assert(len(task_dir) < 3), ("task:" + str(task.name) + " can only have one or two (i.e., read and/or write) ratio")
        return task_dir

    # ---------------------------
    # Functionality:
    #       connecting a block to another block.
    # Variables:
    #       neigh: neighbour, the block to connect to.
    # --------------------------
    def connect(self, neigh):
        if neigh not in self.neighs:
            neigh.neighs.append(self)
            self.neighs.append(neigh)

    # ---------------------------
    # Functionality:
    #       disconnecting a block from another block.
    # Variables:
    #       neigh: neighbour, the block to disconnect from.
    # --------------------------
    def disconnect(self, neigh):
        if neigh not in self.neighs:
            return
        self.neighs.remove(neigh)
        neigh.neighs.remove(self)

    # ---------------------------
    # Functionality:
    #       load (map) a task on a block.
    # Variables:
    #       task: task to load (map)
    #       family_task: family task is a task that write/read to another task
    # --------------------------
    def load_improved(self, task, family_task):

        # the following copule of lines for debugging
        if not len(self.__task_name_dir_list) == sum([len(el.values()) for el in self.__tasks_dir_work_ratio.values()]):
            blah =  len(self.__task_name_dir_list)
            blah2  =  sum([len(el.values()) for el in self.__tasks_dir_work_ratio.values()])
            print("this shoud not happend, so please debug")

        # determine relationship between the tasks
        relationship = task.get_relationship(family_task)
        if relationship == "parent": dir= "read"
        elif relationship == "child": dir = "write"
        elif relationship == "self": dir = "loop_back"
        else:
            print("relationship between " + task.name + " and " + family_task.name + " is " + relationship)
            exit(0)
        task_dir = (task, dir)
        work_ratio = task.get_work_ratio(family_task)
        if task_dir in self.__tasks_dir_work_ratio.keys():
            if not family_task.name in self.__tasks_dir_work_ratio[task_dir].keys():
                self.__task_name_dir_list.append((task.name, dir))
            self.__tasks_dir_work_ratio[task_dir][family_task.name] = work_ratio# where work ratio is set
        else:
            self.__tasks_dir_work_ratio[task_dir] = {}
            self.__tasks_dir_work_ratio[task_dir][family_task.name] = work_ratio  # where work ratio is set
            self.__task_name_dir_list.append((task.name, dir))

        # the following copule of lines for debugging
        if not len(self.__task_name_dir_list) == sum([len(el.values()) for el in self.__tasks_dir_work_ratio.values()]):
            blah =  len(self.__task_name_dir_list)
            blah2  =  sum([len(el.values()) for el in self.__tasks_dir_work_ratio.values()])
            print("this should not happen, so please debug")

    # add a pipe for the block.
    def set_pipe(self, pipe_):
        self.pipes.append(pipe_)

    def set_pipe_cluster(self, cluster):
        self.pipe_clusters.append(cluster)

    def reset_pipes(self):
        self.pipes = []

    def reset_clusters(self):
        self.pipe_clusters = []

    def get_pipe_clusters(self, ):
        return self.pipe_clusters

    def get_pipe_clusters_of_task(self, task):
        clusters = []
        for pipe_cluster in self.pipe_clusters:
            if pipe_cluster.is_task_present(task):
                clusters.append(pipe_cluster)
        return clusters


    def get_pipes(self, channel_name):
        block = self
        result = []
        if channel_name == "same":
            dirs = ["read", "write"]
        else:
            dirs = [channel_name]

        for dir_ in dirs:
            pipes_of_channel = list(filter(lambda pipe: pipe.dir == dir_, self.pipes))
            for pipe in pipes_of_channel:
                if block == pipe.get_slave():
                    result.append(pipe)

        #if len(result) ==0:
        #    print("what")
        #assert(len(result)>0)
        return result

    # ---------------------------
    # Functionality:
    #      get the family tasks of a task by the direction
    # Variables:
    #       task_dir: task_dir is read / write corresponding to parent / child
    #                                                       respectively)
    # --------------------------
    def get_tasks_by_direction(self, task_dir):
        assert (task_dir in ["read", "write"])
        return [task_dir for task_dir in self.__tasks_dir_work_ratio if task_dir[1] == task_dir]

    # ---------------------------
    # Functionality:
    #      unload all the tasks that read from the block
    # --------------------------
    def unload_read(self):
        change = True
        delete = [task_dir for task_dir in self.__tasks_dir_work_ratio if task_dir[1] == "read"]
        for el in delete: del self.__tasks_dir_work_ratio[el]

        list_delete = [task_dir for task_dir in self.__task_name_dir_list if task_dir[1] == "read"]
        for el in list_delete: self.__task_name_dir_list.remove(el)

        # the following copule of lines are for debugging. get rid of it
        if not len(self.__task_name_dir_list) == sum([len(el.values()) for el in self.__tasks_dir_work_ratio.values()]):
            blah= len(self.__task_name_dir_list)
            blah2=sum([len(el.values()) for el in self.__tasks_dir_work_ratio.values()])
            print("this should not happen, so please debug")

    # ---------------------------
    # Functionality:
    #      unload the task from the block with a certain direction (i.e., read/write)
    # Variables:
    #       task_dir
    # --------------------------
    def unload(self, task_dir):
        task, dir = task_dir
        if  not (task.name, dir) in self.__task_name_dir_list:
            print("what")

        while (task.name, dir) in self.__task_name_dir_list:
            self.__task_name_dir_list.remove((task.name, dir))
        del self.__tasks_dir_work_ratio[task_dir]

        # the following couple of lines are for debugging. get rid of it
        if not len(self.__task_name_dir_list) == sum([len(el.values()) for el in self.__tasks_dir_work_ratio.values()]):
            blah =  len(self.__task_name_dir_list)
            blah2  =  sum([len(el.values()) for el in self.__tasks_dir_work_ratio.values()])
            print("this shoud not happend, so please debug")

    # ---------------------------
    # Functionality:
    #      unload all the tasks.
    # --------------------------
    def unload_all(self):
        self.__task_name_dir_list = []
        self.__tasks_dir_work_ratio = {}

    # ---------------------------
    # Functionality:
    #     disconnect the block from all the blocks.
    # --------------------------
    def disconnect_all(self):
        neighs = self.neighs[:]
        for neigh in neighs:
            self.disconnect(neigh)

    # ---------------------------
    # Functionality:
    #     get all the neighbouring blocks (i.e, all the blocks that are connected to a block)
    # --------------------------
    def get_neighs(self):
        return self.neighs

    def get_metric(self, metric):
        if metric == "latency":
            return self.get_peak_work_rate()
        if metric == "energy":
            return self.get_work_over_energy()
        if metric == "power":
            return self.get_work_over_power()
        if metric == "area":
            return self.get_work_over_area()


# traffic is a stream that moves between two pipes (from one task to another)
class traffic:
    def __init__(self, parent, child, dir, work):
        self.parent = parent  # parent task
        self.child = child  # child task
        self.work = work  # work
        self.dir = dir  # direction (read/write/loop)


    def get_child(self):
        return self.child

    def get_parent(self):
        return self.parent



# physical channels inside the the router
class PipeCluster:
    def __init__(self, ref_block, dir, outgoing_pipe, incoming_pipes, unique_name):
        self.ref_block = ref_block
        self.dir = dir
        self.cluster_type = "regular"
        self.pathlet_phase_work_rate = {}
        self.pathlet_phase_latency = {}
        if outgoing_pipe == None:
            self.outgoing_pipe = None
        else:
            self.outgoing_pipe = outgoing_pipe

        if incoming_pipes == None:
            self.incoming_pipes = []
        else:
            self.incoming_pipes = incoming_pipes
        self.unique_name = unique_name

        self.pathlets = []
        for in_pipe in incoming_pipes:
            self.pathlets.append(pathlet(in_pipe, outgoing_pipe, self.dir))

    def change_to_dummy(self, tasks):
        self.cluster_type = "dummy"
        self.set_dir("same")
        self.dummy_tasks = tasks

    def set_dir(self, dir):
        self.dir = dir

    def get_unique_name(self):
        return self.unique_name

    # the block at the center of the cluster (with source (incoming) pipes  and dest (outgoing) pipe)
    def get_block_ref(self):
        return self.ref_block

    def get_pathlets(self):
        return self.pathlets

    # incoming pipes
    def get_incoming_pipes(self):
        assert not(self.cluster_type == "dummy")
        return self.incoming_pipes

    def get_outgoing_pipe(self):
        assert not(self.cluster_type == "dummy")
        return self.outgoing_pipe

    def get_dir(self):
        return self.dir

    def get_task_work_unit(self, task):
        assert not(self.cluster_type == "dummy")
        work_unit = 0
        for pipe in self.incoming_pipes:
            work_unit += pipe.get_task_work_unit(task)
        return work_unit

    def get_task_work(self, task):
        assert not(self.cluster_type == "dummy")
        work = 0
        for pipe in self.incoming_pipes:
            work += pipe.get_task_work(task)
        return work

    def is_task_present(self, task):
        if self.get_block_ref().type == "pe":
            return task.get_name() in [el.get_name() for el in self.dummy_tasks]
        elif self.get_block_ref().type in ["ic"]:
            return self.outgoing_pipe.is_task_present(task)
        else:
            return any([pipe.is_task_present(task ) for pipe in self.incoming_pipes])

    def get_info(self):
        incoming_pipes = []
        for el in self.incoming_pipes:
            incoming_pipes.append(el.get_info())
        if self.outgoing_pipe == None:
            outgoing = " None "
            #outgoing_tasks = "None"
        else:
            outgoing = self.outgoing_pipe.get_info()


        return "block:" + self.get_block_ref().instance_name + " incoming_pipes:" +str(incoming_pipes) + "  outgoing_pipes:"+str(outgoing)

    # for path's within the pipe cluster set the work rate
    def set_pathlet_phase_work_rate(self, pathlet, phase_num, work_rate):
        in_pipe = pathlet.get_in_pipe()
        out_pipe = pathlet.get_out_pipe()
        if in_pipe not in self.incoming_pipes or out_pipe not in [self.outgoing_pipe]:
            print("pipe should already exist")
            exit(0)
        else:
            if pathlet not in self.pathlet_phase_work_rate.keys():
                self.pathlet_phase_work_rate[pathlet] = {}
            if phase_num not in self.pathlet_phase_work_rate[pathlet].keys():
                self.pathlet_phase_work_rate[pathlet][phase_num] = 0
            self.pathlet_phase_work_rate[pathlet][phase_num] += work_rate



    def get_pathlet_phase_work_rate(self):
        return self.pathlet_phase_work_rate

    def get_pathlet_last_phase_work_rate(self):
        pathlet_last_phase_work_rate = {}
        for pathlet, phase_work_rate in self.get_pathlet_phase_work_rate().items():
            last_phase = sorted(phase_work_rate.keys())[-1]
            pathlet_last_phase_work_rate[pathlet] = phase_work_rate[last_phase]
        return pathlet_last_phase_work_rate, last_phase

    def set_pathlet_latency(self, pathlet, phase_num, latency_dict):
        if pathlet not in self.pathlet_phase_latency.keys():
            self.pathlet_phase_latency[pathlet] = {}
        for latency_typ, trv_dir_val in latency_dict.items():
            for trv_dir, val in trv_dir_val.items():
                if trv_dir not in self.pathlet_phase_latency[pathlet].keys():
                    self.pathlet_phase_latency[pathlet][trv_dir] = {}
                if phase_num not in self.pathlet_phase_latency[pathlet][trv_dir].keys():
                    self.pathlet_phase_latency[pathlet][trv_dir][phase_num] = 0
                self.pathlet_phase_latency[pathlet][trv_dir][phase_num] += val
        return self.pathlet_phase_latency[pathlet]


    def get_pathlet_phase_latency(self):
        return self.pathlet_phase_latency

class pathlet:
    def __init__(self, in_pipe, out_pipe, dir_):
        self.in_pipe = in_pipe
        self.out_pipe = out_pipe
        self.dir = dir_

    def get_out_pipe(self):
        return self.out_pipe

    def get_in_pipe(self):
        return self.in_pipe

    def get_dir(self):
        return self.dir


# A pipe is a queue that connects blocks
class pipe:
    def __init__(self, master, slave, dir_, number, cmd_queue_size, data_queue_size):
        self.traffics = []  # traffic on the pipe
        self.dir = dir_  # direction of the traffic
        self.master = master  # master block
        self.slave = slave  # slave block
        self.number = number  # an assigned id
        self.cmd_queue_size = cmd_queue_size
        self.data_queue_size = data_queue_size

    def get_cmd_queue_size(self):
        return self.cmd_queue_size

    def set_cmd_queue_size(self, size):
        self.size = size

    def get_data_queue_size(self):
        return self.data_queue_size

    def set_data_queue_size(self, size):
        self.data_queue_size = size

    def get_dir(self):
        return self.dir

    def get_master(self):
        return self.master

    def get_slave(self):
        return self.slave

    def update_traffic_and_dir(self, parent , child, work, dir):
        if dir not in self.dir:
            print("this should not happen. Same direction should be assigned all the time")
            exit(0)

        self.traffics.append(traffic(parent, child, dir, work))

    def get_task_work_unit(self, task):
        traffics_ = []
        for traffic_ in self.traffics:
            if traffic_.dir == "read" and task.name == traffic_.child.name:
                traffics_.append(traffic_)
            elif traffic_.dir == "write" and task.name == traffic_.parent.name:
                traffics_.append(traffic_)

        work = 0
        for traffic_ in traffics_:
            if traffic_.dir == "read":
                work += task.get_self_to_family_task_work_unit(traffic_.parent)
            elif traffic_.dir  == "write":
                work += task.get_self_to_family_task_work_unit(traffic_.child)

        return work

    def get_task_work(self, task):
        traffics_ = []
        for traffic_ in self.traffics:
            if traffic_.dir == "read" and task.name == traffic_.child.name:
                traffics_.append(traffic_)
            elif traffic_.dir == "write" and task.name == traffic_.parent.name:
                traffics_.append(traffic_)

        work = 0
        for traffic_ in traffics_:
            if traffic_.dir == "read":
                work += task.get_self_to_family_task_work(traffic_.parent)
            elif traffic_.dir  == "write":
                work += task.get_self_to_family_task_work(traffic_.child)

        return work

    # is task on the pipe
    def is_task_present(self, task):
        result = False
        if "write" in self.dir:
            result = result or (task.name in (set([el.parent.name for el in self.traffics])))
        if "read" in self.dir:
            result = result or (task.name in (set([el.child.name for el in self.traffics])))

        return result

    def get_tasks(self):
        if "write" in self.dir:
            results = set([el.parent.name for el in self.traffics])
        if "read" in self.dir:
            results = set([el.child.name for el in self.traffics])
        return results

    def get_traffic(self):
        return self.traffics


    def get_traffic_names(self):
        result = []
        for el in self.traffics:
            result.append(el.parent.name+ " " + el.child.name+ "   -  ")
        return result

    def get_info(self):
        return "m:"+self.master.instance_name + " s:"+self.slave.instance_name

# This class is the graph with nodes denoting the blocks and the edges denoting
# the relationship (parent/child) between the nodes.
# generation_mode either tool_generated or user_generated. Different checks are applied
# for different scenarios.
class HardwareGraph:
    def __init__(self, block_to_prime_with:Block, generation_mode="tool_generated"):
        self.pipes = []
        self.pipe_clusters = []
        self.last_pipe_assigned_number = 0  # this is used for setting up the pipes.
        self.last_cluster_assigned_number = 0
        self.blocks = self.traverse_neighs_recursively(block_to_prime_with, []) # all the blocks in the graph
        self.task_graph = TaskGraph(self.get_all_tasks())  # the task graph
        self.config_code = str(-1) # used to differentiat between systems (focuses on type/number of elements)
        self.SOC_design_code= "" # used to differentiat between systems (focuses on type/number of elements)
        self.simplified_topology_code = str(-1)  # used to differential between systems (focuses on number of elements)
        self.pipe_design()  # set up the pipes
        self.generation_mode = generation_mode



    def get_blocks(self):
        return self.blocks

    # get a pipe, given it's master and slave
    def get_pipe_with_master_slave(self, master_, slave_, dir_):
        for pipe in self.pipes:
            if pipe.get_master() == master_ and pipe.get_slave() == slave_ and dir_ in pipe.get_dir():
                return pipe

        print("this pipe was not found. something wrong")
        master_to_slave_path = self.get_path_between_two_vertecies(master_, slave_)
        exit(0)

    # do sanity check on the pipe
    def pipe_is_sane(self, pipe_):
        if pipe_.master.type == "mem":
            return False
        if pipe_.slave.type == "pe":
            return False
        return True

    # traverse the hardware graph and assign the pipes between them
    def traverse_and_assign_pipes(self, block, blocks_visited):
        if block in blocks_visited:
            return None
        blocks_visited.append(block)
        for neigh in block.neighs:
            dirs = [["read"], ["write"]]
            for dir_ in dirs:
                pipe_ = pipe(block, neigh, dir_, self.last_pipe_assigned_number)
                if self.pipe_is_sane(pipe_):
                    self.last_pipe_assigned_number += 1
                    block.set_pipe(pipe_)
                    neigh.set_pipe(pipe_)
                    self.pipes.append(pipe_)
            self.traverse_and_assign_pipes(neigh, blocks_visited)
        return blocks_visited

    # ---------------------------
    # Functionality:
    #     traverse all the neighbours of a block recursively.
    # Variables:
    #     block: block to get neighbours for
    #     blocks_visited: blocks already visited in the depth first search. Prevent double counting certain blocks.
    # --------------------------
    def traverse_neighs_recursively(self, block, blocks_visited): # depth first search
        if block in blocks_visited:
            return None
        blocks_visited.append(block)
        for neigh in block.neighs:
            self.traverse_neighs_recursively(neigh, blocks_visited)
        return blocks_visited

    # a node is unnecessary when no task lives on it.
    # this happens when we apply moves (e.g., migrate tasks around)
    def prune_unnecessary_nodes(self, block_to_prime_with): # depth first search
        blocks = self.traverse_neighs_recursively(block_to_prime_with, [])
        for block in blocks:
            if block.type == "ic":
                connectd_pes = [block_ for block_ in block.get_neighs() if block_.type == "pe"]
                connectd_mems = [block_ for block_ in block.get_neighs() if block_.type == "mem"]
                connected_ics = [block_ for block_ in block.get_neighs() if block_.type == "ic"]
                #only_one_pe_one_ic =  len(connectd_mems) == 0 and len(connectd_pes) == 1 and len(connected_ics) == 1
                no_mem_one_ic =  len(connectd_mems) == 0 and len(connected_ics) == 1
                no_pe_one_ic_no_system_bus = len(connectd_pes) == 0 and len(connected_ics) == 1 and (True or not block.is_system_ic())
                no_pe_no_mem = (len(connectd_pes) == 0 and len(connectd_mems) == 0)

                if no_mem_one_ic:
                    ic_to_connect_to = connected_ics[0]
                    for pe in connectd_pes:
                        pe.connect(ic_to_connect_to)
                        pe.disconnect(block)
                elif no_pe_one_ic_no_system_bus:
                    ic_to_connect_to = connected_ics[0]
                    for mem in connectd_mems:
                        mem.connect(ic_to_connect_to)
                        mem.disconnect(block)
                # if either of above true, take care of ics as well
                if no_mem_one_ic or no_pe_one_ic_no_system_bus or no_pe_no_mem:
                    if len(connected_ics) == 1:
                        block.disconnect(connected_ics[0])
                    else:
                        ic_to_connect_to = connected_ics[0]
                        block.disconnect(connected_ics[0])
                        for ic in connected_ics[1:]:
                            ic.connect(ic_to_connect_to)
                            block.disconnect(ic)

    # ---------------------------
    # Functionality:
    #     sample a task (from the task distribution) in the task graph. Used for jitter modeling.
    # --------------------------
    def sample(self, hw_sampling):
        # sample tasksz
        for task_ in self.get_all_tasks():
           task_.update_task_work(task_.sample_self_task_work())
           for child_task in task_.get_children():
               task_.update_task_to_child_work(child_task, task_.sample_self_to_child_task_work(child_task))

        # update blocks with the sampled tasks
        blocks = self.traverse_neighs_recursively(self.get_root(), [])
        for block in blocks:
            block.update_all_tasks_dir_work_ratio()

        # sample blocks
        for block in blocks:
            if hw_sampling["mode"] in ["error_integration", "exact"]: # including the error
                block.set_rates(hw_sampling)
            else:
                print("hw_sampling_mode" + hw_sampling["mode"] +" is not defined")
                exit(0)
    # ---------------------------
    # Functionality:
    #     get all the tasks associated with the hardware graph (basically all the blocks tasks)
    # --------------------------
    def get_all_tasks(self):
        all_tasks = []
        for block in self.blocks:
            all_tasks.extend(block.get_tasks_of_block())
            all_tasks = list(set(all_tasks))  # get rid of duplicates
        return all_tasks

    def get_blocks_by_type(self, type_):
        return [block for block in self.blocks if block.type == type_]

    # get all the blocks that hos the task
    def get_blocks_of_task_by_name(self, task_name):
        all_blocks = []
        for block in self.blocks:
            task_names = [el.get_name() for el in block.get_tasks_of_block()]
            if task_name in task_names:
                all_blocks.append(block)

        return all_blocks


    # get all the blocks that hos the task
    def get_blocks_of_task(self, task):
        all_blocks = []
        for block in self.blocks:
            if task in block.get_tasks_of_block() :
                all_blocks.append(block)

        return all_blocks

    # this is just a number that helps us encode a design topology/block type
    def set_config_code(self):
        pes = str(sorted(['_'.join(blck.instance_name.split("_")[:-1]) for blck in self.get_blocks_by_type("pe")]))
        mems = str(sorted(['_'.join(blck.instance_name.split("_")[:-1]) for blck in self.get_blocks_by_type("mem")]))
        ics = str(sorted(['_'.join(blck.instance_name.split("_")[:-1]) for blck in self.get_blocks_by_type("ic")]))

        self.config_code = str(len(self.get_blocks_by_type("ic"))) + "_" + \
                           str(len(self.get_blocks_by_type("mem"))) + "_" + \
                           str(len(self.get_blocks_by_type("pe")))
        self.config_code = pes+"__"+mems +"__" + ics

    # this code (value) uniquely specifies a design.
    # Usage: prevent regenerating/reevaluting the design for example.
    def set_SOC_design_code(self):
        # sort based on the blk name and task names tuples
        def sort_blk_tasks(blks):
            blk_tasks = []
            for blk in blks:
                blk_tasks_sorted = str([el.name for el in sorted(blk.get_tasks_of_block(), key=lambda x: x.name)])
                blk_name_stripped = '_'.join(blk.instance_name.split("_")[:-1])
                blk_tasks.append((blk, blk_tasks_sorted + "__tasks_one_blk_" + blk_name_stripped))
            blk_sorted = sorted(blk_tasks, key=lambda x: x[1])
            return blk_sorted

        self.SOC_design_code= "" # used to differentiat between systems (focuses on type/number of elements)
        hg_string = ""

        # sort the PEs
        pes_sorted = sort_blk_tasks(self.get_blocks_by_type("pe"))
        # iterate through PEs
        for pe, string_ in pes_sorted:
            hg_string += string_
            # sort the neighbours
            neighs_sorted = sort_blk_tasks(pe.get_neighs())
            # iterate through neighbours
            for neigh, string_ in neighs_sorted:
                hg_string +=  string_
                neigh_s_neighs_sorted = sort_blk_tasks(neigh.get_neighs())
                for neigh_s_neigh, string_ in neigh_s_neighs_sorted:
                    if not neigh_s_neigh.type == "mem":
                        continue
                    hg_string += string_

        # task graph based id
        TG = sorted([tsk.name for tsk in self.get_task_graph().get_all_tasks()])
        for tsk_name in TG:
            task = self.get_task_graph().get_task_by_name(tsk_name)
            blks_hosting_task = self.get_blocks_of_task(task)
            blks_hosting_task_sorted = str(sorted(['_'.join(blk.instance_name.split("_")[:-1]) for blk in blks_hosting_task]))
            self.SOC_design_code += tsk_name + "_" + blks_hosting_task_sorted + "___"
        self.SOC_design_code += hg_string

    # this is just a number that helps us encode a design topology/block type
    def get_SOC_design_code(self):
        return self.SOC_design_code

    # just a string to specify the simplified_topology
    def set_simplified_topology_code(self):
        self.simplified_topology_code = str(len(self.get_blocks_by_type("ic"))) + "_" + \
                           str(len(self.get_blocks_by_type("mem"))) + "_" + \
                           str(len(self.get_blocks_by_type("pe")))

    def get_simplified_topology_code(self):
        if self.simplified_topology_code == "-1":
            self.set_simplified_topology_code()
        return self.simplified_topology_code

    def get_number_of_channels(self):
        ics = self.get_blocks_by_type("ic")
        total_number_channels = 0
        for blk in ics:
            total_number_channels+= len(blk.get_pipe_clusters())

        return total_number_channels


    def get_routing_complexity(self):
        pes = self.get_blocks_by_type("pe")
        mems = self.get_blocks_by_type("mem")
        ics = self.get_blocks_by_type("ic")

        # a measure of how hard it is to rout,
        # which depends on how many different paths that can be taken between master and slaves
        complexity = 0

        for pe in pes:
            for mem in mems:
                all_paths = self.get_all_paths_between_two_vertecies(pe, mem)
                complexity += len(all_paths)

        # normalized to the number of master slaves
        complexity = complexity/(len(pes)*len(mems))
        return complexity

    def get_config_code(self):
        if self.config_code == "-1":
            self.set_config_code()
        return self.config_code

    # ---------------------------
    # Functionality:
    #       update the graph without prunning. No pruning policy allows for hardware graphs to be directly
    #       (without modifications) absorbed from the input when requeste
    # --------------------------
    def update_graph_without_prunning(self, block_to_prime_with=None):
        if not block_to_prime_with:
            block_to_prime_with = self.get_root()
        self.blocks = self.traverse_neighs_recursively(block_to_prime_with, [])
        self.set_config_code()
        self.set_SOC_design_code()

    # ---------------------------
    # Functionality:
    #       update the graph. Used for jitter modeling after a new task was sampled from the task distribution.
    # --------------------------
    def update_graph(self, block_to_prime_with=None):
        if not block_to_prime_with:
            block_to_prime_with = self.get_root()
        elif block_to_prime_with not in self.get_blocks():
            for blck in self.get_blocks():
                if blck.instance_name == block_to_prime_with.instance_name:
                    block_to_prime_with = blck
                    break
        self.prune_unnecessary_nodes(block_to_prime_with)
        self.blocks = self.traverse_neighs_recursively(block_to_prime_with, [])
        self.set_config_code()
        self.set_SOC_design_code()
        #self.assign_pipes() # rehuild pipes from scratch
        # re assigning pipes

    # assign tasks to the pipes
    def task_the_pipes(self, task, pipes, dir):
        for pipe_ in pipes:
            if dir == "read":
                block = pipe_.get_slave()
                for parent in task.get_parents():  # [par.name for par in task.get_parents()]:
                    parent_names = block.get_task_s_family_by_task_and_dir(task, "read")
                    if parent.name in parent_names:
                        work = parent.get_self_to_family_task_work(task)
                        pipe_.update_traffic_and_dir(parent, task, work, "read")

            elif dir == "write":
                block = pipe_.get_slave()
                for child in task.get_children():
                    children_names = block.get_task_s_family_by_task_and_dir(task, "write")
                    if child.name in children_names:
                        work = task.get_self_to_family_task_work(child)
                        pipe_.update_traffic_and_dir(task, child, work, "write")

    def get_pipes_between_two_blocks(self, blck_1, blck_2, dir_):
        pipes = []
        # get blocks along the way
        master_to_slave_path = self.get_path_between_two_vertecies(blck_1, blck_2)
        # get pipes along the way
        for idx in range(0, len(master_to_slave_path) - 1):
            block_master = master_to_slave_path[idx]
            block_slave = master_to_slave_path[idx + 1]
            pipes.append(self.get_pipe_with_master_slave(block_master, block_slave, dir_))
        return pipes

    def filter_empty_pipes(self):
        empty_pipes = []
        for pipe in self.pipes:
            if pipe.traffics == []:
                empty_pipes.append(pipe)

        for pipe in empty_pipes:
            self.pipes.remove(pipe)

    # assign task to pipes
    def task_all_the_pipes(self):
        def get_blocks_of_task(task):
            blocks = []
            for block in self.blocks:
                if task in block.get_tasks_of_block():
                    blocks.append(block)
            return blocks

        # assign tasks to pipes
        self.task_pipes = {}
        all_tasks = self.get_all_tasks()
        for task in all_tasks:
            pe = [block for block in get_blocks_of_task(task) if  block.type == "pe" ][0]
            mem_reads = [block for block in get_blocks_of_task(task) if  block.type == "mem"  and (task, "read") in block.get_tasks_dir_work_ratio().keys()]
            mem_writes = [block for block in get_blocks_of_task(task) if  block.type == "mem"  and (task, "write") in block.get_tasks_dir_work_ratio().keys()]

            # get all the paths leading from mem reads to pe
            seen_pipes = []
            for mem in mem_reads:
                pipes = self.get_pipes_between_two_blocks(pe, mem, "read")
                pipes_to_consider = []
                for pipe in pipes:
                    if pipe not in seen_pipes:
                        pipes_to_consider.append(pipe)
                        seen_pipes.append(pipe)
                if len(pipes_to_consider) == 0:
                    continue
                else:
                    self.task_the_pipes(task, pipes_to_consider, "read")

            # get all the paths leading from mem reads to pe
            seen_pipes = []
            for mem in mem_writes:
                pipes = self.get_pipes_between_two_blocks(pe, mem, "write")
                pipes_to_consider = []
                for pipe in pipes:
                    if pipe not in seen_pipes:
                        pipes_to_consider.append(pipe)
                        seen_pipes.append(pipe)
                if len(pipes_to_consider) == 0:
                    continue
                else:
                    self.task_the_pipes(task, pipes_to_consider, "write")


    def get_pipe_dir(self, block):
        #if block.type == "pe":
        #    return ["same"]
        #else:
        return ["write", "read"]

    def get_pipe_clusters(self):
        return self.pipe_clusters

    def cluster_pipes(self):
        self.pipe_clusters = []
        for block in self.blocks:
            block.reset_clusters()

        def traffic_overlaps(blck_pipe, neigh_pipe):
            blck_pipe_traffic = blck_pipe.get_traffic_names()
            neigh_pipe_traffic = neigh_pipe.get_traffic_names()
            traffic_non_overlap = list(set(blck_pipe_traffic) - set(neigh_pipe_traffic))
            return  len(traffic_non_overlap) < len(blck_pipe_traffic)

        if (len(self.pipes) == 0):
            print("something is wrong")

        assert(len(self.pipes) > 0),  "you need to assign pipes first"
        pipe_cluster_dict = {}

        # iterate through blocks and neighbours to generate clusters
        for block in self.blocks:
            pipe_cluster_dict[block] = {}
            for neigh in block.get_neighs():
                if neigh.type == "pe": continue
                for dir in self.get_pipe_dir(block):
                    if dir not in pipe_cluster_dict[block]: pipe_cluster_dict[block][dir] = {}

                    # get the pipes
                    block_pipes = block.get_pipes(dir)
                    neigh_pipes = neigh.get_pipes(dir)

                    if block_pipes == [] and block.type == "pe":
                        pipe_cluster_dict[block][dir] = {}
                        for neigh_pipe in neigh_pipes:
                            if neigh_pipe.master == block and neigh_pipe.dir == dir:
                                pipe_cluster_dict[block][dir][neigh_pipe] = []
                    elif block.type == "mem":
                        pipe_cluster_dict[block][dir] = {}
                        pipe_cluster_dict[block][dir][None] = block_pipes
                    elif block.type == "ic":
                        if dir not in pipe_cluster_dict[block]:
                            pipe_cluster_dict[block][dir] = {}
                        for blck_pipe in block_pipes:
                            for neigh_pipe in neigh_pipes:
                                if not blck_pipe.slave == neigh_pipe.master:
                                    continue
                                if not(blck_pipe.dir == neigh_pipe.dir):
                                    continue
                                if traffic_overlaps(blck_pipe, neigh_pipe):
                                    if neigh_pipe not in pipe_cluster_dict[block][dir].keys():
                                        pipe_cluster_dict[block][dir][neigh_pipe] = []
                                    if blck_pipe not in pipe_cluster_dict[block][dir][neigh_pipe]:
                                        pipe_cluster_dict[block][dir][neigh_pipe].append(blck_pipe)

        # now generates the clusters
        for block, dir_outgoing_incoming_pipes in pipe_cluster_dict.items():
            for dir, outgoing_pipe_incoming_pipes in dir_outgoing_incoming_pipes.items():
                for outgoing_pipe, incoming_pipes in outgoing_pipe_incoming_pipes.items():
                    pipe_cluster_ = PipeCluster(block, dir, outgoing_pipe, incoming_pipes, self.last_cluster_assigned_number)
                    if outgoing_pipe:  # for pe and ic
                        if outgoing_pipe.master.type == "pe":  # only push once
                            pipe_cluster_.change_to_dummy(outgoing_pipe.master.get_tasks_of_block())
                            if len(outgoing_pipe.master.get_pipe_clusters()) == 0:
                                outgoing_pipe.master.set_pipe_cluster(pipe_cluster_)
                        else:
                            outgoing_pipe.master.set_pipe_cluster(pipe_cluster_)
                    elif incoming_pipes: # for mem
                        incoming_pipes[0].slave.set_pipe_cluster(pipe_cluster_)
                    self.last_cluster_assigned_number += 1
                    self.pipe_clusters.append(pipe_cluster_)

        pass

    def size_queues(self):
        # set to the default value
        for pipe in self.pipes:
            # by default set the cmd/data size to master queue size
            pipe.set_cmd_queue_size(config.default_data_queue_size)
            pipe.set_data_queue_size(config.default_data_queue_size)

        # actually size the queues
        """
        for pipe in self.pipes:
            # by default set the cmd/data size to master queue size
            pipe_slave = pipe.get_slave()
            # ignore PEs
            if pipe_slave.type == "pe":
                continue
            pipe_line_depth = pipe_slave.get_pipe_line_depth()
            pipe.set_cmd_queue_size(pipe_line_depth)
            pipe.set_data_queue_size(pipe_line_depth)
            pipe_tasks = pipe.get_tasks()
        """


    def generate_pipes(self):
        # assign number to pipes
        self.last_pipe_assigned_number = 0
        pes = self.get_blocks_by_type("pe")
        mems = self.get_blocks_by_type("mem")
        ics = self.get_blocks_by_type("ic")
        def seen_pipe(pipe__):
            for pipe in self.pipes:
                if pipe__.master == pipe.master and pipe__.slave == pipe.slave and pipe__.dir == pipe.dir:
                    return True
            return False

        # iterate through all the blocks and specify their pipes
        for pe in pes:
            for mem  in mems:
                master_to_slave_path = self.get_path_between_two_vertecies(pe, mem)
                if len(master_to_slave_path) > len(ics)+2: # two is for pe and memory
                    print('something has gone wrong with the path calculation')
                    exit(0)
                # get pipes along the way
                for idx in range(0, len(master_to_slave_path) - 1):
                    block_master = master_to_slave_path[idx]
                    block_slave = master_to_slave_path[idx + 1]
                    for dir_ in ["write", "read"]:
                        pipe_ = pipe(block_master, block_slave, dir_, self.last_pipe_assigned_number, 1, 1)
                        if not seen_pipe(pipe_):
                            self.pipes.append(pipe_)
                            self.last_pipe_assigned_number +=1

    def connect_pipes_to_blocks(self):
        for pipe in self.pipes:
            master_block = pipe.get_master()
            slave_block = pipe.get_slave()
            master_block.set_pipe(pipe)
            slave_block.set_pipe(pipe)

    # assign pipes to different blocks (depending on which blocks the pipes are connected to)
    def pipe_design(self):
        for block in self.blocks:
            block.reset_pipes()
            self.pipes = []
            self.pipe_clusters = []

        # generate pipes everywhere
        self.generate_pipes()
        # assign tasks
        self.task_all_the_pipes()
        # filter pipes without tasks
        self.filter_empty_pipes()
        self.connect_pipes_to_blocks()
        self.cluster_pipes()
        self.size_queues()

    # ---------------------------
    # Functionality:
    #       finding all the paths (set of edges) that connect two blocks (nodes) in the hardware graph.
    # Variables:
    #       vertex: source vertex
    #       v_des: destination vertex
    #       vertecies_visited: vertices visited already (avoid circular traversal of the graph)
    #       path: the accumulated path so far. At the end, this will contain the total path.
    # --------------------------
    def get_all_paths(self, vertex, v_des, vertecies_neigh_visited, path):
        paths = self.get_shortest_path_helper(vertex, v_des, vertecies_neigh_visited, path)
        #sorted_paths = sorted(paths, key=len)
        return paths


    # ---------------------------
    # Functionality:
    #       finding the path (set of edges) that connect two blocks (nodes) in the hardware graph.
    # Variables:
    #       vertex: source vertex
    #       v_des: destination vertex
    #       vertecies_visited: vertices visited already (avoid circular traversal of the graph)
    #       path: the accumulated path so far. At the end, this will contain the total path.
    # --------------------------
    def get_shortest_path(self, vertex, v_des, vertecies_neigh_visited, path):
        paths = self.get_shortest_path_helper(vertex, v_des, vertecies_neigh_visited, path)
        sorted_paths = sorted(paths, key=len)
        return sorted_paths[0]

    def get_shortest_path_helper(self, vertex, v_des, vertecies_neigh_visited, path):
        neighs = vertex.get_neighs()
        path.append(vertex)

        # iterate through neighbours and remove the ones that you have already visited
        neighs_to_ignore = []
        for neigh in neighs:
            if (vertex,neigh) in vertecies_neigh_visited:
                neighs_to_ignore.append(neigh)
        neighs_to_look_at = list(set(neighs) - set(neighs_to_ignore))

        if vertex == v_des:
            return [path]
        elif len(neighs_to_look_at) == 0:
            return []
        else:
            for neigh in neighs_to_look_at:
                vertecies_neigh_visited.append((neigh, vertex))

            paths = []
            for vertex_ in neighs_to_look_at:
                paths_  = self.get_shortest_path_helper(vertex_, v_des, vertecies_neigh_visited[:], path[:])
                for path_ in paths_:
                    if len(path) == 0:
                        continue
                    paths.append(path_)

            return paths


    # ---------------------------
    # Functionality:
    #       finding the path (set of edges) that connect two blocks (nodes) in the hardware graph.
    # Variables:
    #       vertex: source vertex
    #       v_des: destination vertex
    #       vertecies_visited: vertices visited already (avoid circular traversal of the graph)
    #       path: the accumulated path so far. At the end, this will contain the total path.
    # --------------------------
    def get_path_helper(self, vertex, v_des, vertecies_visited, path):
        path.append(vertex)
        if vertex in vertecies_visited:
            return []
        if vertex == v_des:
            return path
        else:
            vertecies_visited.append(vertex)
            paths = [self.get_path_helper(vertex_, v_des, vertecies_visited[:], path[:]) for vertex_ in vertex.neighs]
            flatten_path = list(itertools.chain(*paths))
            return flatten_path

    # ---------------------------
    # Functionality:
    #       finding the path (set of edges) that connet two blocks (nodes) in the hardware graph.
    # Variables:
    #       v1: source vertex
    #       v2: destination vertex
    # --------------------------
    def get_path_between_two_vertecies(self, v1, v2):
        #path = self.get_path_helper(v1, v2, [], [])
        #if (len(path)) <= 0:
        #    print("catch this error")
        #assert(len(path) > 0), "no path between the two nodes"
        shortest_path = self.get_shortest_path(v1, v2, [],[])
        #if not shortest_path == path:
        #    print("something gone wrong with path calculation fix this")
        return shortest_path

    # ---------------------------
    # Functionality:
    #       finding all the paths (set of edges) that connet two blocks (nodes) in the hardware graph.
    # Variables:
    #       v1: source vertex
    #       v2: destination vertex
    # --------------------------
    def get_all_paths_between_two_vertecies(self, v1, v2):
        all_paths = self.get_all_paths(v1, v2, [],[])
        return all_paths




    # ---------------------------
    # Functionality:
    #       get root of the hardware graph.
    # --------------------------
    def get_root(self):
        return self.blocks[0]

    # ---------------------------
    # Functionality:
    #       get task's graph
    # --------------------------
    def get_task_graph(self):
        return self.task_graph