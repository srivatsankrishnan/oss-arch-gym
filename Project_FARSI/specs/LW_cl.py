#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

# this file contains light weight version of the design
# components. These classes are used in the database
from typing import List, Tuple
from design_utils.components.hardware import *
from design_utils.components.workload import *
#from design_utils.components.mapping import *
from design_utils.components.scheduling import *
from typing import List
from collections import defaultdict

# This class emulates a hardware block. However,  BlockL is a light weight class that directly talks to the database.
# This is later used by the Block class which is a much more involved
class BlockL:  # block light weight
    def __init__(self, block_instance_name: str, block_type: str, block_subtype, peak_work_rate_distribution,
                 work_over_energy_distribution, work_over_area_distribution, one_over_area_distribution,
                 clock_freq, bus_width, loop_itr_cnt=1, loop_max_possible_itr_cnt=1,
                 hop_latency=1, pipe_line_depth=1,
                 leakage_power = "", power_knobs = ""):
        self.block_instance_name = block_instance_name+"_"+block_type  # block instance name
        self.block_instance_name_without_type = block_instance_name  # without type
        self.block_type = block_type  # type of the block (pe, mem, ic)
        self.block_subtype = block_subtype  # sub type of each block, e.g, for pe: ip or gpp
        self.peak_work_rate_distribution = peak_work_rate_distribution  # peak_work_rate: the fastest that a hardware block can do it's work
                                             # note that work definition varies depending on the hardware type
                                             # e.g., pe work = instructions, mem/ic work = bytes
        self.work_over_energy_distribution = work_over_energy_distribution # how much energy is consume per the amount of work done
        self.work_over_area_distribution = work_over_area_distribution  # how much area is occupied per the amount of work done
        self.leakage_power = leakage_power  # leakage power
        self.power_knobs = power_knobs  # all knobs used for power modulation
        self.one_over_area_distribution = one_over_area_distribution
        self.bus_width =  bus_width
        self.clock_freq = clock_freq
        self.loop_itr_cnt = loop_itr_cnt
        self.loop_max_possible_itr_cnt = loop_max_possible_itr_cnt
        self.hop_latency = hop_latency
        self.pipe_line_depth = pipe_line_depth


# This class emulates the software tasks (e.g., glint detection) within an entire workload. Note that TaskL is a
# light weight class that directory talks to the database. This is later used by the Task class which is much more
# involved.
class TaskL:  # task light weight
    def __init__(self, task_name: str, work: float, iteration=1, type = "latency_based", throughput_info = {}):
        self.task_name = task_name
        self.work = work   #  the amount of work associated with task (at the mement, this is expressed for PEs (as the
                           # reference block). so work = number of instructions.
        self.__task_children = []  # dependent task.
        self.__self_to_children_work = {}  # amount of byte that will be passed from this task to its children.
        self.__self_task_work_distribution = [] # amount of work for this task as a distribution (for jitter modeling).
        self.__self_to_child_task_work_distribution = {}   # amount of bytes passed from this task to its children (as a distribution).
        self.__children_nature_dict = dict()
        self.burst_size = 256
        self.iteration = iteration
        self.throughput_info = throughput_info # can be empty if the task is latency based
        self.type = type
        # hardcoding for testing
        #self.iteration = 2
        """
        if "Smoothing" in self.task_name:
            self.type = "throughput_based"
            self.throughput_info = {"read": 100000000, "write":100000000, "clock_period": 10000}  # clock period in ns
        else:
            self.type = "latency_based"
        """

    def get_throughput_info(self):
        return self.throughput_info

    def get_type(self):
        return self.type

    # ------------------------------
    # Functionality:
    #   adding a child task (i.e., a dependent task)
    # Variables:
    #       taskL: child task
    #       work: amount of work (currently expressed in bytes) imposed on the child task
    # ------------------------------
    def add_child(self, taskL, work, child_nature):
        self.__task_children.append(taskL)
        self.__self_to_children_work[taskL] = work
        self.__children_nature_dict[taskL] = child_nature

    def get_child_nature(self, taskL):
        return self.__children_nature_dict[taskL]

    def set_burst_size(self, burst_size):
        self.burst_size = burst_size

    def get_burst_size(self):
        return self.burst_size

    def get_children_nature(self):
        return self.__children_nature_dict

    def set_children_nature(self, children_nature):
        self.__children_nature_dict = children_nature

    # ------------------------------
    # Functionality:
    #    add work distribution (as opposed to a single value) for the task (for jitter modeling)
    # Variables:
    #       work_dis: work distribution to add
    # ------------------------------
    def add_task_work_distribution(self, work_dis):
        self.__self_task_work_distribution = work_dis

    # ------------------------------
    # Functionality:
    #    add work distribution (as opposed to a single value) for the task's child (for jitter modeling)
    # Variables:
    #       childL: child task
    #       work_dis: work distribution to add
    # ------------------------------
    def add_task_to_child_work_distribution(self, childL, work_dist):
        self.__self_to_child_task_work_distribution[childL] = work_dist


    def set_self_to_children_work_distribution(self, task_to_child_work_distribution):
        self.__self_to_child_task_work_distribution = task_to_child_work_distribution

    def get_self_to_children_work_distribution(self):
        return self.__self_to_child_task_work_distribution

    # ------------------------------
    # Functionality:
    #    get the task's work distribution
    # ------------------------------
    def get_task_work_distribution(self):
        return self.__self_task_work_distribution

    def set_task_work_distribution(self, work_dis):
        self.__self_task_work_distribution = work_dis


    # ------------------------------
    # Functionality:
    #    get the task to child work distribution
    # ------------------------------
    def get_task_to_child_task_work_distribution(self, childL):
        return self.__self_to_child_task_work_distribution[childL]

    # get taskL's dependencies
    def get_children(self):
        return self.__task_children

    # set taskL's dependencies
    def set_children(self, children):
        self.__task_children = children

    # ------------------------------
    # Functionality:
    #    convert the light weight task (i.e., TaskL) to Task class
    # ------------------------------
    def toTask(self):
        return Task(self.task_name, self.work)

    # ------------------------------
    # Functionality:
    #    get work imposed on the child
    # Variables:
    #   child: task's child
    # ------------------------------
    def get_self_to_child_work(self, child):
        assert(child in self.__self_to_children_work.keys())
        return self.__self_to_children_work[child]

    # ------------------------------
    # Functionality:
    #    get work imposed on the children
    # ------------------------------
    def get_self_to_children_work(self):
        return self.__self_to_children_work

    def set_self_to_children_work(self, self_to_children_work):
        self.__self_to_children_work = self_to_children_work


# This class emulates an SOC
class SOCL:
    def __init__(self, type, budget_dict, other_metrics_dict):
        self.budget_dict = budget_dict
        self.other_metrics_dict = other_metrics_dict
        self.type = type
        assert (sorted(list(budget_dict.keys())) ==  sorted(config.budgetted_metrics)), "budgetted metrics need to be the same"

    # ------------------------------
    # Functionality:
    #     get the SOC budget
    # Variables:
    #       metric_name: metric (energy, power, area, latency) to get buget for
    # ------------------------------
    def get_budget(self, metric_name):
        for metric_name_, budget_value in self.budget_dict.items():
            if metric_name_ == metric_name:
                return budget_value
        raise Exception("meteric:" + metric_name + " is not budgetted in the design")


    def set_budget(self, metric_name, metric_value):
        for metric_name_, _ in self.budget_dict.items():
            if metric_name_ == metric_name:
                self.budget_dict[metric_name_]  = metric_value
        raise Exception("meteric:" + metric_name + " is not budgetted in the design")

    def set_other_metrics_ideal_values(self, metric, value):
        for metric_name_, _ in self.other_metrics_dict.items():
            if metric_name_ == metric:
                self.other_metrics_dict[metric_name_] =  value
                return
        raise Exception("meteric:" + metric + " is not in other values in the design")

    def get_other_metrics_ideal_values(self, metric_name):
        for metric_name_, ideal_value in self.other_metrics_dict.items():
            if metric_name_ == metric_name:
                return ideal_value
        raise Exception("meteric:" + metric_name + " is not in other values in the design")

    # ------------------------------
    # Functionality:
    #     get the name of all the metrics that have budgets.
    # ------------------------------
    def get_budgetted_metric_names(self):
        return list(self.budget_dict.keys())


    def get_other_metric_names(self):
        return list(self.other_metrics_dict.keys())

    # get type of the SOC
    def get_type(self):
        return self.type

    def get_budget_dict(self):
        return self.budget_dict


# This class models a mapping from a task to a block
class TaskToPEBlockMapL:
    def __init__(self, task_name: str, pe_block_instance_name):
        self.task_name = task_name  # task name
        self.pe_block_instance_name = pe_block_instance_name+"_"+"pe"  # block name
        self.child_facing_work_ratio_dict = {}  # This is work ratio assiociated with a specific child (task). Note that
                                            # since this is facing a child, it's a write (so write work ratio)
        self.parent_facing_work_ratio_dict = {}  # This is the write work ratio assiociated with a specific child (task)
                                            # since this is facing a parent, it's a read (so read work ratio)

        self.family_work_ratio = defaultdict(dict)   # work ratio associated with the task and it's family (parents/children) members.
        # Work_ratio:
        # self: (PE) = 1
        # parent to child: (mem) = bytes/insts
        self.family_work_ratio['self'][self.task_name] = 1  # this is for PEs, so for every family member, we'll add
                                                # a work_ratio of 1 which is associated with the tasks' work itself

    # ------------------------------
    # Functionality:
    #     adding a family member for the task
    # Variable:
    #       list of family members and their work ratio
    # ------------------------------
    def add_family(self, family_list: List[Tuple[str, str, float]]):
        for family in family_list:
            family_member_name = family[1]
            family_member_work_ratio = family[2]
            relationship = family[0]
            # this is "mem" and "bus" work_ratio
            self.family_work_ratio[relationship][family_member_name] = family_member_work_ratio

    # ------------------------------
    # Functionality:
    #     getting the work ratio
    # ------------------------------
    def get_work_ratio_new(self):
        return self.family_work_ratio
        raise Exception("could not find a task with name:" + task.name + " in this task_to_block_map")

# This class contains scheduling information for task
class TaskScheduleL:
    def __init__(self, task_name: str, starting_time: float):
        self.task_name = task_name
        self.starting_time = starting_time