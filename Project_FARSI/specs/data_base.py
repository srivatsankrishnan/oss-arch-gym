#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

from specs.LW_cl import *
import time
import random
from operator import itemgetter, attrgetter
from settings import config
from datetime import datetime
#from specs import database_input
if config.simulation_method == "power_knobs":
    from specs import database_input_powerKnobs as database_input
elif config.simulation_method == "performance":
    from specs import database_input
else:
    raise NameError("Simulation method unavailable")

from design_utils.components.hardware import  *


# this class is used for populating th library.
# It is a light weight class to help library population
class IPLibraryElement():
    def __init__(self):
        self.blockL = "_"
        self.task = "_"
        self.PPAC_dict = {}

    def set_task(self, task_):
        self.task = task_

    def set_blockL(self, blockL_):
        self.blockL = blockL_

    # generate the PPAC (performance, power, area and cost) dictionary
    def generate(self):
        self.PPAC_dict["latency"] = self.task.get_self_task_work()/self.blockL.peak_work_rate
        self.PPAC_dict["energy"] = self.task.get_self_task_work()/self.blockL.work_over_energy
        self.PPAC_dict["area"] = self.task.get_self_task_work()/self.blockL.work_over_area
        self.PPAC_dict["power"] = self.PPAC_dict["energy"]/self.PPAC_dict["latency"]

    def get_blockL(self):
        return self.blockL

    def get_task(self):
        return self.task

    def get_PPAC(self):
        return self.PPAC_dict


# this class handles the input data base data.
# any class that wants to read the database data needs to talk to this class
class DataBase:
    def __init__(self, db_input, hw_sampling):
        self.cached_blockL_to_block = {}
        self.cached_block_to_blockL = {}
        self.db_input = db_input
        self.tasksL = self.db_input.tasksL   # task
        self.pe_mapsL = self.db_input.pe_mapsL  # pe maps
        self.blocksL = self.db_input.blocksL  # blocks
        self.pe_schedulesL = db_input.pe_schedeulesL  # schedules
        self.workloads_last_task = db_input.workloads_last_task
        self.SOCsL = db_input.SOCsL  # SOC
        self.SOC_id = 0  # soc id.
        self.hw_sampling = hw_sampling   # how to sample hardware

        # cluster blocks
        self.ic_block_list = self.get_blocksL_by_type(block_type="ic")  # list of ICs
        self.mem_block_list = self.get_blocksL_by_type(block_type="mem") # list of memories
        self.pe_block_list = self.get_blocksL_by_type(block_type="pe") # list of PEs

        # generate tasks (TODO: this might need to move out of data_base if we consider a data_base for different workloads)
        self.__tasks = self.parse_and_gen_tasks()  # generate all the tasks (from tasks in the database_input.py)
        self.__blocks = [self.cast(blockL) for blockL in self.get_all_BlocksL()]

        # find mappable blocks for all the tasks
        self.mappable_blocksL_to_tasks_s_name_dict:Dict[str:TaskL] = {}
        self.populate_mappable_blocksL_to_tasks_s_name_dict()

    def set_workloads_last_task(self, workloads_last_task):
        self.workloads_last_task = workloads_last_task

    # used to determine when a workload is done
    def get_workloads_last_task(self):
        return self.workloads_last_task

    def set_hw_sampling(self, hw_sampling):
        self.hw_sampling = hw_sampling

    # ------------------------------
    # Functionality:
    #     input_database to database conversion
    # ------------------------------
    def cast(self, obj, *argv ):
        if len(argv) == 0 and isinstance(obj, BlockL):
            return Block(self.db_input, self.hw_sampling, obj.block_instance_name, obj.block_type, obj.block_subtype,
                         self.get_block_peak_work_rate_distribution(obj), self.get_block_work_over_energy_distribution(obj),
                         self.get_block_work_over_area_distribution(obj), self.get_block_one_over_area_distribution(obj),
                         obj.clock_freq,
                         obj.bus_width,
                         obj.loop_itr_cnt,
                         obj.loop_max_possible_itr_cnt,
                         obj.hop_latency,
                         obj.pipe_line_depth,
                         self.get_block_leakage_power(obj),
                         self.get_block_power_knobs(obj),)
        elif len(argv) == 0 and isinstance(obj, TaskL):
            return Task(obj.task_name, self.get_task_work(obj), self.get_task_iteration(obj), self.get_task_type(obj), self.get_task_throughput_info(obj))
        elif len(argv) == 3 and isinstance(obj, Task) and isinstance(argv[0], Block):
            raise Exception("this is case is deprecated")
            task = obj
            pe_block = argv[0]
            ic = argv[1]
            mem = argv[2]

            task_to_blocks_map = TaskToBlocksMap(task, {})
            work_ratio_read = self.get_block_work_ratio_by_task(task, pe_block, "read")
            # TODO: this needs to be fixed to react to different referencing and also different interconnects, and mem
            task_to_blocks_map.block_workRatio_dict[pe_block] = 1
            task_to_blocks_map.block_workRatio_dict[ic] = work_ratio_read
            task_to_blocks_map.block_workRatio_dict[mem] = work_ratio_read
            return task_to_blocks_map
        elif len(argv) == 1 and isinstance(obj, Task) and isinstance(argv[0], str):
            return TaskToPEBlockSchedule(obj, self.get_task_starting_time(obj))
        else:
            raise Exception("this casting is not acceptable" + str(type(obj)))

    # get the name of the budgeted metrics
    def get_budgetted_metric_names_all_SOCs(self):
        result = []
        for SOCL in self.SOCsL:
            result.extend(SOCL.get_budgetted_metric_names())

        return np.unique(result).tolist()

    def get_other_metric_names_all_SOCs(self):
        result = []
        for SOCL in self.SOCsL:
            result.extend(SOCL.get_other_metric_names())

        return np.unique(result).tolist()

    def get_budgetted_metric_names(self, type):
        for SOCL in self.SOCsL:
            if SOCL.type == type:
                return SOCL.get_other_mertic_names()

    # ------------------------------
    # Functionality:
    #       get the name of budgeted metrics
    # Variables:
    #       type: SOC type_ to querry
    # -------------------------------------------
    def get_budgetted_metric_names(self, type):
        for SOCL in self.SOCsL:
            if SOCL.type == type:
                return SOCL.get_budgetted_metric_names()

    # ------------------------------
    # Functionality:
    #       get the budget value  for a specific metric.
    # Variables:
    #       type: SOC type_
    #       metric: metric to query.
    # -------------------------------------------
    def get_budget(self, metric, type):
        for SOCL in self.SOCsL:
            if SOCL.type == type:
                if not metric in SOCL.get_budgetted_metric_names():
                    print("this metric is not budgget")
                    exit(0)
                return SOCL.get_budget(metric)

    # ideal or desired value is the value that we are targetting
    def get_other_metrics_ideal_value(self, metric, type):
        for SOCL in self.SOCsL:
            if SOCL.type == type:
                if not metric in SOCL.get_other_metrics_names():
                    print("this metric is not included in the other metrics")
                    exit(0)
                return SOCL.get_other_metrics_ideal_value(metric)

    def set_ideal_metric_value(self, metric, type, value):
        for SOCL in self.SOCsL:
            if SOCL.type == type:
                if metric in SOCL.get_budgetted_metric_names():
                    return SOCL.set_budget(metric, value)
                elif metric in SOCL.get_other_metric_names():
                    return SOCL.set_other_metrics_ideal_values(metric, value)

    # get the desired (basically budget) value for various metrics
    def get_ideal_metric_value(self, metric, type):
        for SOCL in self.SOCsL:
            if SOCL.type == type:
                if metric in SOCL.get_budgetted_metric_names():
                    return SOCL.get_budget(metric)
                elif metric in SOCL.get_other_metric_names():
                    return SOCL.get_other_metrics_ideal_values(metric)

    # ------------------------------
    # Functionality:
    #       get all the tasks for the database
    # -------------------------------------------
    def get_tasks(self):
        return self.__tasks

    def get_task_by_name(self, name):
        for task in self.get_tasks():
            if task.name == name:
                return task
        return None

    def get_blocks(self):
        return self.__blocks

    def get_block_by_name(self, name):
        for block in self.get_blocks():
            if block.get_generic_instance_name() == name:
                return block
        return None

    # ------------------------------
    # Functionality:
    #       get the list of blocks that a task can map to.
    # Variables:
    #       task_name: name of the task to query.
    # -------------------------------------------
    def get_task_s_mappable_blocksL_by_task_s_name(self, task_name):
        return self.mappable_blocksL_to_tasks_s_name_dict[task_name]

    # ------------------------------
    # Functionality:
    #       get the task (task class) version of taskL, if one has already be generated.
    # Variables:
    #       taskL: taskL object to compare against.
    #       tasks: list of already generated tasks.
    # -------------------------------------------
    def get_tasks_from_taskL(self, taskL, tasks):
        for task in tasks:
            if taskL.task_name == task.name:
                return task
        return None

    # ------------------------------
    # Functionality:
    #       find all the blocks that input tasks can map to simultaneously. This means the all the tasks
    #       should be mappable to each one of the blocks in the output
    # Variables:
    #       block_type: filter based on block type
    #       tasks: list of tasks to find mappable blocks for
    # -------------------------------------------
    def find_all_mappable_blocksL_for_tasks(self, block_type, tasks=[]):
        # this function works by having two buckets (shared_so_far and so_far_temp
        # and swapping them
        if block_type in ["ic"]:
            return self.ic_block_list
        elif block_type == "mem":
            return self.mem_block_list
        elif block_type == "pe":
            shared_so_far_temp = self.get_task_s_mappable_blocksL_by_task_s_name(tasks[0].name)
            found_common_block = False
            for task in tasks[1:]:
                blockL_list = self.get_task_s_mappable_blocksL_by_task_s_name(task.name)
                shared_so_far = shared_so_far_temp  # swap
                shared_so_far_temp = []
                for blockL in blockL_list:
                    if blockL in shared_so_far:
                        shared_so_far_temp.append(blockL)
                        found_common_block = True

            if len(tasks) == 1 or found_common_block:
                return shared_so_far_temp
            else:
                raise Exception("no common block found")
        else:
            raise ValueError("block_type:" + block_type + " not supported")

    # ------------------------------
    # Functionality:
    #       get all the blocksL
    # Variables:
    #       block_type: filter based on block_type
    # -------------------------------------------
    def get_blocksL_by_type(self, block_type):
        return list(filter(lambda blockL: blockL.block_type == block_type, self.blocksL))

    def get_all_BlocksL(self):
        return self.blocksL

    # ------------------------------
    # Functionality:
    #      get the blocks work/energy (how much work done per energy consumed)
    # Variables:
    #      blockL: blockL under query.
    # -------------------------------------------
    def get_block_work_over_energy_distribution(self, blockL):
        blockL = list(filter(lambda block: block.block_instance_name == blockL.block_instance_name, self.blocksL))
        assert(len(blockL) == 1)
        return blockL[0].work_over_energy_distribution

    def get_block_one_over_area_distribution(self, blockL):
        blockL = list(filter(lambda block: block.block_instance_name == blockL.block_instance_name, self.blocksL))
        assert(len(blockL) == 1)
        return blockL[0].one_over_area_distribution

    # ------------------------------
    # Functionality:
    #      get the blocks leakage power
    # Variables:
    #      blockL: blockL under query.
    # -------------------------------------------
    def get_block_leakage_power(self, blockL):
        blockL = list(filter(lambda block: block.block_instance_name == blockL.block_instance_name, self.blocksL))
        assert(len(blockL) == 1)
        return blockL[0].leakage_power

    # ------------------------------
    # Functionality:
    #      get the blocks power knobs
    # Variables:
    #      blockL: blockL under query.
    # -------------------------------------------
    def get_block_power_knobs(self, blockL):
        return 0
        blockL = list(filter(lambda block: block.block_instance_name == blockL.block_instance_name, self.blocksL))
        assert(len(blockL) == 1)
        return blockL[0].power_knobs

    #def introduce_iteration(self, tasks):

    # ------------------------------
    # Functionality:
    #       parses taskL and generates Tasks objects (including their dependencies)
    # -------------------------------------------
    def parse_and_gen_tasks(self):
        tasks = [self.cast(taskL) for taskL in self.tasksL]
        for task in tasks:
            corresponding_taskL = self.get_taskL_from_task_name(task.name)
            if config.eval_mode == "statistical":
                task.add_task_work_distribution(corresponding_taskL.get_task_work_distribution())
            taskL_children = corresponding_taskL.get_children()
            for taskL_child in taskL_children:
                child_task =  self.get_tasks_from_taskL(taskL_child, tasks)
                taskL__ = self.get_taskL_from_task_name(task.name)
                task.add_child(child_task, taskL__.get_self_to_child_work(taskL_child), corresponding_taskL.get_child_nature(taskL_child))
                task.set_burst_size(taskL__.get_burst_size())
                if config.eval_mode == "statistical":
                    task.add_task_to_child_work_distribution(child_task, taskL__.get_task_to_child_task_work_distribution(taskL_child))

        return tasks

    # ------------------------------
    # Functionality:
    #       find all the compatible blocks for a specific task. This version is the fast version (using caching), however
    #       it's a bit less intuitive
    # Variables:
    #       block_type: filter based on the type (pe, mem, ic)
    #       tasks: all the tasks that the blocks need to compatible for.
    # -------------------------------------------
    def find_all_compatible_blocks_fast(self, block_type, tasks):
        mappable_blocksL = self.find_all_mappable_blocksL_for_tasks(block_type, tasks)
        assert(len(mappable_blocksL) > 0), "there should be at least one block for all the tasks to map to"
        result = []
        for mappable_blockL in mappable_blocksL:
            if mappable_blockL in self.cached_blockL_to_block.keys(): # to improve performance, we look into caches
                result.append(self.cached_blockL_to_block[mappable_blockL])
            else:
                casted = self.cast(mappable_blockL)
                self.cached_blockL_to_block[mappable_blockL] = casted
                self.cached_block_to_blockL[casted] = mappable_blockL
                result.append(casted)
        return result

    # ------------------------------
    # Functionality:
    #       find all the compatible blocks for a specific task
    # Variables:
    #       block_type: filter based on the type (pe, mem, ic)
    #       tasks: all the tasks that the blocks need to compatible for.
    # -------------------------------------------
    def find_all_compatible_blocks(self, block_type, tasks):
        mappable_blocksL = self.find_all_mappable_blocksL_for_tasks(block_type, tasks)
        assert(len(mappable_blocksL) > 0), "there should be at least one block for all the tasks to map to"
        return [self.cast(mappable_blockL) for mappable_blockL in mappable_blocksL]

    # get the block name (without type in name) generate a block object with it, and return it
    # Note that without type means that the name does not contains the type, and we add the type when we generate
    # BlockL so we can make certain deduction about the block for Platform architect purposes
    def gen_one_block_by_name_without_type(self, blk_name_without_type):
        # generate the block
        blockL = [el for el in self.blocksL if el.block_instance_name_without_type == blk_name_without_type][0]
        block = self.cast(blockL)
        # set the SOC type:
        # TODO: for now we just set it to the most superior SOC. Later, get an input for this
        ordered_SOCsL = sorted(self.SOCsL, key=lambda SOC: SOC.get_budget("latency"))
        block.set_SOC(ordered_SOCsL[0].type, self.SOC_id)
        return block

    # get the block name and generate a block object with it, and return it
    def gen_one_block_by_name(self, blk_name):
        blk_name_refined = '_'.join(
            blk_name.split("_")[:-1])  # fix the name by getting rid of the instance part of the name
        suffix = blk_name.split("_")[-1]
        # This is becuase BlockL do not have an instance name
        # generate the block
        blockL = [el for el in self.blocksL if el.block_instance_name_without_type == blk_name_refined][0]
        block = self.cast(blockL)
        block_name = block.instance_name + "_" + suffix
        block.set_instance_name(block_name)
        # set the SOC type:
        # TODO: for now we just set it to the most superior SOC. Later, get an input for this
        ordered_SOCsL = sorted(self.SOCsL, key=lambda SOC: SOC.get_budget("latency"))
        block.set_SOC(ordered_SOCsL[0].type, self.SOC_id)
        return block


    # ------------------------------
    # Functionality:
    #       get taskL from the task name
    # Variables:
    #       task_name: name of the task
    # -------------------------------------------
    def get_taskL_from_task_name(self, task_name):
        for taskL in self.tasksL:
            if taskL.task_name == task_name:
                return taskL
        raise Exception("taskL:" + task_name + " does not exist")

    # ------------------------------
    # Functionality:
    #       get pe for a taskL
    # Variables:
    #       taskL: taskL
    # -------------------------------------------
    def get_corresponding_pe_blocksL(self, taskL):
        task_to_pe_blocks_mapL_list = list(filter(lambda pe_map: pe_map.task_name == taskL.task_name, self.pe_mapsL))
        assert(len(task_to_pe_blocks_mapL_list) >= 1), taskL.task_name
        pe_blocksL_list = []
        for el in task_to_pe_blocks_mapL_list:
            blockL_list = list(filter(lambda blockL: blockL.block_instance_name == el.pe_block_instance_name, self.blocksL))
            assert(len(blockL_list) == 1)
            pe_blocksL_list.append(blockL_list[0])
        return pe_blocksL_list

    def get_task_starting_time(self, task):
        return list(filter(lambda pe_sch: pe_sch.task_name == task.name, self.pe_schedulesL))[0].starting_time

    def get_block_peak_work_rate_distribution(self, blockL):
        blockL = list(filter(lambda block: block.block_instance_name == blockL.block_instance_name, self.blocksL))
        assert(len(blockL) == 1)
        return blockL[0].peak_work_rate_distribution

    def get_block_work_over_area_distribution(self, blockL):
        blockL = list(filter(lambda block: block.block_instance_name == blockL.block_instance_name, self.blocksL))
        assert(len(blockL) == 1)
        return blockL[0].work_over_area_distribution

    def get_blocks_immediate_superior(self, block):
        return None

    # ----------------------------------
    # samplers
    # ----------------------------------
    # sample the database from a set of compatible blocks
    def sample_blocks(self, all_compatible_blocks, mode="random"):
        if (config.DEBUG_FIX):
            random.seed(0)
        else:
            time.sleep(.00001)
            random.seed(datetime.now().microsecond)

        if not (all_compatible_blocks):
            raise Exception("din't find any compatible blocks")

        if (mode == "random"):
            block = random.choice(all_compatible_blocks)
        elif (mode == "immediate_superior"):
            block = sorted(all_compatible_blocks)[0]
        else:
            print("mode: " + mode + " for block sampling is not defined")
            exit(0)
        return block

    def sample_all_blocks(self, tasks, block, mode="random"):
        all_compatible_blocks = self.find_all_compatible_blocks(block.type, tasks)
        new_block = self.sample_blocks(all_compatible_blocks, mode)
        return new_block

    def sample_similar_block(self, block):
        for blocksL in self.get_all_BlocksL():
            if blocksL.block_instance_name == block.get_generic_instance_name():
                return self.cast(blocksL)

        print("there should be at least one block that is similar ")
        exit(0)

    def sample_DMA(self):
       return list(filter(lambda blockL: blockL.block_type == block_type, self.blocksL))

    def get_tasksL(self):
        return self.tasksL

    # ------------------------------
    # Functionality:
    #      get all the PE's that the task can run on
    # ------------------------------
    def get_corresponding_pe_blocksL(self, taskL):
        task_to_pe_blocks_mapL_list = list(filter(lambda pe_map: pe_map.task_name == taskL.task_name, self.pe_mapsL))
        assert(len(task_to_pe_blocks_mapL_list) >= 1), taskL.task_name
        pe_blocksL_list = []
        for el in task_to_pe_blocks_mapL_list:
            blockL_list = list(filter(lambda blockL: blockL.block_instance_name == el.pe_block_instance_name, self.blocksL))
            if not(len(blockL_list) ==1):
                print("weird")
            assert(len(blockL_list) == 1)
            pe_blocksL_list.append(blockL_list[0])
        return pe_blocksL_list

    def get_task_starting_time(self, task):
        return list(filter(lambda pe_sch: pe_sch.task_name == task.name, self.pe_schedulesL))[0].starting_time

    # ------------------------------
    # Functionality:
    #      get the block work ratio given the task and it's direction (read/write)
    # Variables:
    #      task: task under query
    #      pe_block: pe block to get work ratio for
    #      dir: direction for work ratio (read/write)
    # -------------------------------------------
    def get_block_work_ratio_by_task_dir(self, task, pe_block, dir):
        work_ratio = [pe_map_.get_work_ratio_new() for pe_map_ in self.pe_mapsL if pe_map_.task_name == task.name and
                                 pe_map_.pe_block_instance_name == pe_block.get_generic_instance_name()]
        if dir == 'read': relationship = "parent"
        elif dir == "write": relationship = "child"
        elif dir == "loop_back": relationship = "self"
        if "DMA" in task.name:
            if relationship == "parent": return {task.get_parents()[0].name:1}
            elif relationship == "child": return {task.get_children()[0].name:1}
            elif relationship == "self": return {task.name:1}
        else:
            return work_ratio[0][relationship]

    # task work is reported in number of instructions
    def get_task_work(self, taskL):
        taskL = list(filter(lambda taskL_: taskL_.task_name == taskL.task_name, self.tasksL))
        assert(len(taskL) == 1)
        return taskL[0].work

    def get_task_iteration(self, taskL):
        taskL = list(filter(lambda taskL_: taskL_.task_name == taskL.task_name, self.tasksL))
        assert(len(taskL) == 1)
        return taskL[0].iteration

    def get_task_throughput_info(self, taskL):
        taskL = list(filter(lambda taskL_: taskL_.task_name == taskL.task_name, self.tasksL))
        assert(len(taskL) == 1)
        return taskL[0].get_throughput_info()

    def get_task_type(self, taskL):
        taskL = list(filter(lambda taskL_: taskL_.task_name == taskL.task_name, self.tasksL))
        assert(len(taskL) == 1)
        return taskL[0].get_type()






    """
    # find the block that is better than the current block that the tasks are running on
    # superior means more power, area efficient or better latency
    def get_blocks_immediate_superior_for_tasks(self, block, tasks):
        mappable_blocksL = self.find_mappable_blocks_among_tasks(block.block_type, tasks)
        superior_blockL = self.get_blocks_immediate_superior(block.block_subtype)
        while(superior_blockL):
            if superior_blockL in mappable_blocksL:
                return self.cast(mappable_blocksL)
        return block
    """

    # -------------------
    # generators (finder and convert)
    # ------------------
    def populate_mappable_blocksL_to_tasks_s_name_dict(self):
        for taskL in self.tasksL:
            mappable_blocksL_to_task_list = self.get_corresponding_pe_blocksL(taskL)
            self.mappable_blocksL_to_tasks_s_name_dict[taskL.task_name] = mappable_blocksL_to_task_list

    # get all the tasks to blocks mappings
    def get_mappable_blocksL_to_tasks(self):
        return self.mappable_blocksL_to_tasks_s_name_dict

    def sample_DMA_blocks(self):
        DMA_blocks = [blockL for blockL in self.get_blocksL_by_type("pe") if "DMA" in blockL.block_instance_name]
        random_blockL = random.choice(DMA_blocks)
        return self.cast(random_blockL)

    # block type chosend from ["pe", "mem", "ic"]
    def sample_all_blocks_by_type(self, mode="random", tasks=[], block_type="pe"):
        all_compatible_blocks = self.find_all_compatible_blocks(block_type, tasks)
        return self.sample_blocks(all_compatible_blocks, mode)

    def sample_most_inferior_blocks_by_type(self, mode="random", tasks=[], block_type="pe"):
        all_compatible_blocks = self.find_all_compatible_blocks(block_type, tasks)
        return sorted(all_compatible_blocks)[0]

    def sample_most_inferior_blocks_before_unrolling_by_type(self, mode="random", tasks=[], block_type="pe", block=""):
        if not block.subtype == "ip":
            return self.sample_similar_block(block)
        else:
            all_compatible_blocks = self.find_all_compatible_blocks(block_type, tasks)
            sorted_blocks = sorted(all_compatible_blocks)
            for block_ in  sorted_blocks:
                if block_.subtype == "ip" and block_.get_block_freq() == block.get_block_freq():
                    return block_
            return sorted(all_compatible_blocks)[0]


    # superior = better performant wise
    # Variables:
    #       cur_blck: current block to find a superior for
    #       all_comtible_blcks: list of blocks to pick from
    def find_superior_blocks(self, cur_blck, all_comptble_blcks:List[Block]):
        srtd_comptble_blcks = sorted(all_comptble_blcks)
        cur_blck_idx = 0
        for blck in srtd_comptble_blcks:
            if cur_blck.get_generic_instance_name() == blck.get_generic_instance_name():
                break
            cur_blck_idx +=1
        if (cur_blck_idx == len(srtd_comptble_blcks)-1):
            # TODO: not good coding, should really fold this in the previous hierarchy
            return [srtd_comptble_blcks[-1]]
        else:
            return srtd_comptble_blcks[cur_blck_idx+1:]

    # ------------------------------
    # Functionality:
    #   find the better SOC for the block under query
    # Variables:
    #   metric_name: the metric name to pick a better SOC based off of.
    # ------------------------------
    def find_superior_SOC(self, block, metric_name):
        cur_SOC_type = block.SOC_type
        ordered_SOCsL = sorted(self.SOCsL, key=lambda SOC: SOC.get_budget(metric_name))
        cur_SOC_idx = ordered_SOCsL.index(cur_SOC_type)
        if cur_SOC_idx == len(ordered_SOCsL): return ordered_SOCsL[cur_SOC_idx]
        else: return ordered_SOCsL[cur_SOC_idx +1]

    # ------------------------------
    # Functionality:
    #       find a superior (based on the metric_name) SOC.
    # Variables:
    #       block: block to find a better SOC for.
    #       metric_name: name of the metric to choose a better SOC based off of.
    # ------------------------------
    def up_sample_SOC(self, block, metric_name):
        superior_SOC_type = self.find_superior_SOC(block, metric_name)
        block.set_SOC(superior_SOC_type, self.SOC_id)
        return block

    def copy_SOC(self, block_copy_to, block_copy_from):
        block_copy_to.SOC_type = block_copy_from.SOC_type
        block_copy_to.SOC_id = block_copy_from.SOC_id
        return block_copy_to

    # ------------------------------
    # Functionality:
    #       get the worse SOC (based on the metric name)
    # ------------------------------
    def sample_most_inferior_SOC(self, block, metric_name):
        ordered_SOCsL = sorted(self.SOCsL, key=lambda SOCL: SOCL.get_budget(metric_name))
        block.set_SOC(ordered_SOCsL[0].type, self.SOC_id)
        return block

    # ------------------------------
    # Functionality:
    #       find a more superior block  compatible with all the input tasks
    # ------------------------------
    def up_sample_blocks(self, block,  mode="random",  tasks=[]):
        all_compatible_blocks = self.find_all_compatible_blocks(block.type, tasks)
        superior_blocks = self.find_superior_blocks(block, all_compatible_blocks)
        return self.sample_blocks(superior_blocks, mode)

    # ------------------------------
    # Functionality:
    # check if a block is superior comparing to another block
    # ------------------------------
    def check_superiority(self, block_1,  block_2,  tasks=[]):
        all_compatible_blocks = self.find_all_compatible_blocks(block_1.type, tasks)
        superior_blocks_names = [block.get_generic_instance_name() for block in self.find_superior_blocks(block_1, all_compatible_blocks)]
        if block_2.get_generic_instance_name() in superior_blocks_names:
            return True
        else:
            return False


    # ------------------------------
    # Functionality:
    #  find a block that is superior (from a specific metric perspective) comparing to the current block. This version
    # is the fast version, where we use caching, however, it's a bit less intuitive
    # Variables:
    #   blck_to_imprv: block to improve upon
    #   metric: metric to consider while choosing a block superiority. Chosen from power, area, performance.
    # ------------------------------
    def up_sample_down_sample_block_fast(self, blck_to_imprv, metric, sampling_dir, tasks=[]):
        all_compatible_blocks = self.find_all_compatible_blocks_fast(blck_to_imprv.type, tasks)
        if metric == "latency":
            metric_to_sort = 'peak_work_rate'
        elif metric == "power":
            #metric_to_sort = 'work_over_energy'
            metric_to_sort = 'one_over_power'
        elif metric == "area":
            metric_to_sort = 'one_over_area'
        else:
            print("metric: " + metric + " is not defined")

        if sampling_dir > 0:
            reversed = True
        else:
            reversed = False
        srtd_comptble_blcks = sorted(all_compatible_blocks, key=attrgetter(metric_to_sort), reverse=reversed)  #
        idx = 0

        # find the block
        results = []
        for blck in srtd_comptble_blcks:
            #if (getattr(blck, metric_to_sort) == getattr(blck_to_imprv, metric_to_sort)):
            if sampling_dir < 0:  # need to reduce
                if (getattr(blck, metric_to_sort) > getattr(blck_to_imprv, metric_to_sort)):
                    results.append(blck)
            elif sampling_dir > 0:  # need to reduce
                if (getattr(blck, metric_to_sort) < getattr(blck_to_imprv, metric_to_sort)):
                    results.append(blck)

        if len(results) == 0:
            for el in srtd_comptble_blcks:
                if el.get_generic_instance_name() == blck_to_imprv.get_generic_instance_name():
                    results = [el]
                    break

        return results

    # ------------------------------
    # Functionality:
    #  find a block that is superior (from a specific metric perspective) comparing to the current block
    # Variables:
    #   blck_to_imprv: block to improve upon
    #   metric: metric to consider while choosing a block superiority. Chosen from power, area, performance.
    # ------------------------------
    def up_sample_down_sample_block(self, blck_to_imprv, metric, sampling_dir, tasks=[]):
        all_compatible_blocks = self.find_all_compatible_blocks(blck_to_imprv.type, tasks)
        if metric == "latency":
            metric_to_sort = 'peak_work_rate'
        elif metric == "power":
            #metric_to_sort = 'work_over_energy'
            metric_to_sort = 'one_over_power'
        elif metric == "area":
            metric_to_sort = 'one_over_area'
        else:
            print("metric: " + metric + " is not defined")

        if sampling_dir > 0:
            reversed = True
        else:
            reversed = False
        srtd_comptble_blcks = sorted(all_compatible_blocks, key=attrgetter(metric_to_sort), reverse=reversed)  #
        idx = 0

        # find the block
        results = []
        for blck in srtd_comptble_blcks:
            #if (getattr(blck, metric_to_sort) == getattr(blck_to_imprv, metric_to_sort)):
            if sampling_dir < 0:  # need to reduce
                if (getattr(blck, metric_to_sort) > getattr(blck_to_imprv, metric_to_sort)):
                    results.append(blck)
            elif sampling_dir > 0:  # need to reduce
                if (getattr(blck, metric_to_sort) < getattr(blck_to_imprv, metric_to_sort)):
                    results.append(blck)

        if len(results) == 0:
            results = [blck_to_imprv]
        return results

    def equal_sample_up_sample_down_sample_block(self, blck_to_imprv, metric, sampling_dir, tasks=[]):
        all_compatible_blocks = self.find_all_compatible_blocks(blck_to_imprv.type, tasks)
        if metric == "latency":
            metric_to_sort = 'peak_work_rate'
        elif metric == "power":
            # metric_to_sort = 'work_over_energy'
            metric_to_sort = 'one_over_power'
        elif metric == "area":
            metric_to_sort = 'one_over_area'
        else:
            print("metric: " + metric + " is not defined")

        if sampling_dir > 0:
            reversed = True
        else:
            reversed = False
        srtd_comptble_blcks = sorted(all_compatible_blocks, key=attrgetter(metric_to_sort), reverse=reversed)  #
        idx = 0

        # find the block
        results = []
        for blck in srtd_comptble_blcks:
            if sampling_dir < 0:  # need to reduce
                if (getattr(blck, metric_to_sort) >= getattr(blck_to_imprv, metric_to_sort)):
                    results.append(blck)
            elif sampling_dir > 0:  # need to reduce
                if (getattr(blck, metric_to_sort) <= getattr(blck_to_imprv, metric_to_sort)):
                    results.append(blck)

        return results

    def up_sample_down_sample_block_multi_metric_fast(self, blck_to_imprv, sorted_metric_dir, tasks=[]):
        all_compatible_blocks = self.find_all_compatible_blocks_fast(blck_to_imprv.type, tasks)

        metrics_to_sort_reversed = []
        for metric,dir in sorted_metric_dir.items():
            if metric == "latency":
                metric_to_sort = 'peak_work_rate'
            elif metric == "power":
                #metric_to_sort = 'work_over_energy'
                metric_to_sort = 'one_over_power'
            elif metric == "area":
                metric_to_sort = 'one_over_area'
            else:
                print("metric: " + metric + " is not defined")
            metrics_to_sort_reversed.append((metric_to_sort, -1*dir))

        most_important_metric = list(sorted_metric_dir.keys())[-1]
        sampling_dir = sorted_metric_dir[most_important_metric]

        #srtd_comptble_blcks = sorted(all_compatible_blocks, key=attrgetter(metric_to_sort), reverse=reversed)  #
        srtd_comptble_blcks = sorted(all_compatible_blocks, key=lambda blk: (metrics_to_sort_reversed[2][1]*getattr(blk, metrics_to_sort_reversed[2][0]),
                                                                             metrics_to_sort_reversed[1][1]*getattr(blk, metrics_to_sort_reversed[1][0]),
                                                                             metrics_to_sort_reversed[0][1]*getattr(blk, metrics_to_sort_reversed[0][0])))
        idx = 0

        # find the block
        results = []
        """
        # first make sure it can meet across all metrics
        for blck in srtd_comptble_blcks:
            if metrics_to_sort_reversed[2][1]*getattr(blck, metrics_to_sort_reversed[2][0]) > \
                    metrics_to_sort_reversed[2][1]*getattr(blck_to_imprv, metrics_to_sort_reversed[2][0]):
                if metrics_to_sort_reversed[1][1] * getattr(blck, metrics_to_sort_reversed[1][0]) >= \
                        metrics_to_sort_reversed[1][1] * getattr(blck_to_imprv,metrics_to_sort_reversed[1][0]):
                    if metrics_to_sort_reversed[0][1]*getattr(blck, metrics_to_sort_reversed[0][0]) >= \
                            metrics_to_sort_reversed[0][1] * getattr(blck_to_imprv, metrics_to_sort_reversed[0][0]):
                        results.append(blck)

        # meet across two metrics
        if len(results) == 0:
            for blck in srtd_comptble_blcks:
                if metrics_to_sort_reversed[2][1] * getattr(blck, metrics_to_sort_reversed[2][0]) > \
                        metrics_to_sort_reversed[2][1] * getattr(blck_to_imprv, metrics_to_sort_reversed[2][0]):
                    if metrics_to_sort_reversed[1][1] * getattr(blck, metrics_to_sort_reversed[1][0]) >= \
                            metrics_to_sort_reversed[1][1] * getattr(blck_to_imprv, metrics_to_sort_reversed[1][0]):
                        results.append(blck)
        """
        # meet across at least one meteric
        if len(results) == 0:
            for blck in srtd_comptble_blcks:
                if metrics_to_sort_reversed[2][1] * getattr(blck, metrics_to_sort_reversed[2][0]) > \
                        metrics_to_sort_reversed[2][1] * getattr(blck_to_imprv, metrics_to_sort_reversed[2][0]):
                    results.append(blck)

       # we need pareto front calculation here, but we are doing something simple at the moment instead
        if len(results) > 1:
            first_el = results[0]
            second_el = results[1]
            if metrics_to_sort_reversed[1][1] * getattr(first_el, metrics_to_sort_reversed[1][0]) >= \
                    metrics_to_sort_reversed[1][1] * getattr(second_el, metrics_to_sort_reversed[1][0]):
                results = [results[0]]
            else:
                results = [results[1]]

#        if len(results)  > 0:
#            self.check_weird_nests(results, blck_to_imprv, metrics_to_sort_reversed, srtd_comptble_blcks)
        if len(results) == 0:
            for el in srtd_comptble_blcks:
                if el.get_generic_instance_name() == blck_to_imprv.get_generic_instance_name():
                    results = [el]
                    break
        #if len(results) == 0:
        #    results = [srtd_comptble_blcks[-1]]

        return results

    def equal_sample_up_sample_down_sample_block_fast(self, blck_to_imprv, metric, sampling_dir, tasks=[]):
        all_compatible_blocks = self.find_all_compatible_blocks_fast(blck_to_imprv.type, tasks)
        if metric == "latency":
            metric_to_sort = 'peak_work_rate'
        elif metric == "power":
            #metric_to_sort = 'work_over_energy'
            metric_to_sort = 'one_over_power'
        elif metric == "area":
            metric_to_sort = 'one_over_area'
        else:
            print("metric: " + metric + " is not defined")

        if sampling_dir > 0:
            reversed = True
        else:
            reversed = False

        srtd_comptble_blcks = sorted(all_compatible_blocks, key=attrgetter(metric_to_sort), reverse=reversed)  #
        #srtd_comptble_blcks = sorted(all_compatible_blocks, key=lambda blk: (getattr(blk, metrics_to_sort[0]), getattr(blk, metrics_to_sort[1]), getattr(blk, metrics_to_sort[2])), reverse=reversed)  #
        idx = 0

        # find the block
        results = []
        for blck in srtd_comptble_blcks:
            #if (getattr(blck, metric_to_sort) == getattr(blck_to_imprv, metric_to_sort)):
            if sampling_dir < 0:  # need to reduce
                if (getattr(blck, metric_to_sort) >= getattr(blck_to_imprv, metric_to_sort)):
                    results.append(blck)
            elif sampling_dir > 0:  # need to reduce
                if (getattr(blck, metric_to_sort) <= getattr(blck_to_imprv, metric_to_sort)):
                    results.append(blck)

        return results
