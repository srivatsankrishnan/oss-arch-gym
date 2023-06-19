#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import json
import os
from settings import config
from typing import Dict
import warnings
from design_utils.components.hardware import *
from design_utils.components.workload import *
from typing import List


# This task maps a task to a processing block
class TaskToBlocksMap:  #which block(s) the task is  mapped to. a task is usually mapped to a PE, interconnect and memory
    def __init__(self, task:Task, blocks_dir_work_ratio:Dict[Tuple[Block, str], float]={}):
        self.task = task  # list of tasks
        self.block_dir_workRatio_dict: Dict[Tuple[Block, str], float] = blocks_dir_work_ratio   #  block work ratio with its direction (read/write),
                                                                                                #  direction specifies whether a bock is used for reading or writing for this task
                                                                                                 #  work_ratio specifies the task's computation load for the block.
    # find a task (object) by its name
    def find_task_by_name(self, task_name):
        for task in self.tasks:
            if task.name == task_name:
                return task

    # return the task's work. Work can be specified in terms of bytes or instructions
    # We use instructions as the reference (instead of bytes)
    def get_ref_task_work(self):
        return self.task.get_task_work_distribution()

    # return tuples containing tasks and their direction for all tasks that the block hosts.
    # note that direction can be write/read (for tasks on memory/bus) and loop (for PE).
    def get_block_family_members_allocated(self, block_name):
        result = []
        for block_dir, work_ratio_ in self.block_dir_workRatio_dict.items():
            if block_name == block_dir[0].instance_name:
                for task_name in work_ratio_.keys():
                    result.append((task_name, block_dir[1]))

        return result

    # ------------------------------
    # Functionality
    #       get work ratio associated of the task and its direction (read/write)
    # Variables:
    #       block_name: block name of interest
    #       dir_: direction of interest, i.e., read or write
    # ------------------------------
    def get_workRatio_by_block_name_and_dir(self, block_name, dir_):
        blocks = [block_dir[0] for block_dir in self.block_dir_workRatio_dict.keys()]
        work_ratio = 0
        for block_dir, work_ratio_ in self.block_dir_workRatio_dict.items():
            if block_name == block_dir[0].instance_name and block_dir[1] == dir_:
                for family_member, work_ratio_value in work_ratio_.items():
                    if not (family_member in self.task.get_fake_children_name()):
                        work_ratio += work_ratio_value
        return work_ratio

    # ------------------------------
    # Functionality
    #       get work ratio associated of the task and its channel name.
    #       channel can be read/write (for buses) or "same" as in only one channel
    #       for memory and ips.
    #       fake means that there are tasks that the current
    #       task doesn't need to recalculate the result for, rather the result
    #       has been already calculated for another task, and only need to be copied
    # Variables:
    #       block_name: block name of interest
    # ------------------------------
    def get_workRatio_by_block_name_and_channel_eliminating_fake(self, block_name, channel_name):
        if channel_name == "same":
            return self.get_workRatio_by_block_name(block_name)
        else:
            return self.get_workRatio_by_block_name_and_dir_eliminating_fake(block_name, channel_name)


    def get_workRatio_by_block_name_and_family_member_names_and_channel_eliminating_fake(self, block_name, family_members, channel_name):
        if channel_name == "same":
            return self.get_workRatio_by_block_name_and_family_member_names(block_name, [el[0] for el in family_members])
        else:
            return self.get_workRatio_by_block_name_and_family_member_names_and_dir_eliminating_fake(block_name,
                                                                                                     [el[0] for el in family_members], channel_name)

    # given a block, how much work (measured in terms of instructions or bytes depending on the block type)
    # does it need to deliver
    # channel_name: read/write for memory and buses and "same" for PE
    # block_name: name of the block of interest.
    # Note that if we have already decided that read/write channels are the same for buses
    # and memory, then same will be used
    def get_task_total_work_for_block(self, block_name, channel_name, task_name):
        work_ratio = None
        if channel_name == "same":
            work_ratio = self.get_workRatio_by_block_name_and_family_member_names(block_name, [task_name])
        else:
            work_ratio = self.get_workRatio_by_block_name_and_family_member_names_and_dir_eliminating_fake(block_name,
                                                                                                     [task_name], channel_name)

        ref_task_work = self.get_ref_task_work()[0][0]
        return work_ratio*ref_task_work

    # get all the tasks that a block hosts
    def get_tasks_of_block(self, block_name):
        tasks = []
        for block_dir, work_ratio_ in self.block_dir_workRatio_dict.items():
            if block_name == block_dir[0].instance_name:
                for family_member in work_ratio_.keys():
                    if not (family_member in self.task.get_fake_children_name()):
                        tasks.append(family_member)

        return tasks

    # get all the tasks that the block hosts with a certain direction
    # dir: read/write for memory and buses and loop for PE
    def get_tasks_of_block_dir(self, block_name, dir):
        tasks = []
        for block_dir, work_ratio_ in self.block_dir_workRatio_dict.items():
            if block_name == block_dir[0].instance_name and dir == block_dir[1]:
                for family_member in work_ratio_.keys():
                    if not (family_member in self.task.get_fake_children_name()):
                        tasks.append(family_member)

        return tasks

    def get_tasks_of_block_with_src_dest(self, block_name):
        print("this is not supported at the moment")
        exit(0)
        pass
        tasks = []
        for block_dir, work_ratio_ in self.block_dir_workRatio_dict.items():
            if block_name == block_dir[0].instance_name:
                for family_member in work_ratio_.keys():
                    if not (family_member in self.task.get_fake_children_name()):
                        tasks.append(family_member)

        return tasks

    # get tasks of a block with a certain channel
    #   channel can be read/write (for buses) or "same" as in only one channel
    def get_tasks_of_block_channel(self, block_name, channel_name):
        if channel_name == "same":
            return self.get_tasks_of_block_with_src_dest(block_name)
        else:
            return self.get_tasks_of_block_dir(block_name, channel_name)

    # ------------------------------
    # Functionality
    #       get work ratio associated of the task and its dir_ (read/write) or loop (for PEs.
    #       fake means that there are tasks that the current
    #       task doesn't need to recalculate the resul for, rather the result
    #       has been already calculated for another task, and only need to be copied
    # Variables:
    #       block_name: block name of interest
    #          dir_: read/write loop
    # ------------------------------
    def get_workRatio_by_block_name_and_dir_eliminating_fake(self, block_name, dir_):
        blocks = [block_dir[0] for block_dir in self.block_dir_workRatio_dict.keys()]
        work_ratio = 0
        for block_dir, work_ratio_ in self.block_dir_workRatio_dict.items():
            if block_name == block_dir[0].instance_name and block_dir[1] == dir_:
                for family_member, work_ratio_value in work_ratio_.items():
                    if not (family_member in self.task.get_fake_family_name()):
                        work_ratio += work_ratio_value
        return work_ratio

    # ------------------------------
    # Functionality
    #       get work ratio associated of the task and its direction (read/write) using the block name
    # Variables:
    #       block_name: block name of interest
    # ------------------------------
    def get_workRatio_by_block_name(self, block_name):
        work_ratio = 0
        for block_dir, work_ratio_ in self.block_dir_workRatio_dict.items():
            if block_name == block_dir[0].instance_name:
                for family_member, work_ratio_value in work_ratio_.items():
                    if not (family_member in self.task.get_fake_children_name()):
                        work_ratio += work_ratio_value
        if work_ratio == 0:
            if config.WARN:
                warnings.warn("workratio for block" + str(block_name) + "is zero")
        return work_ratio

    # only return work ratio for certain family members
    def get_workRatio_by_block_name_and_family_member_names(self, block_name, family_member_names):
        work_ratio = 0
        for block_dir, work_ratio_ in self.block_dir_workRatio_dict.items():
            if block_name == block_dir[0].instance_name:
                for family_member, work_ratio_value in work_ratio_.items():
                    if not (family_member in self.task.get_fake_children_name()) and (family_member in family_member_names):
                        work_ratio += work_ratio_value
        if work_ratio == 0:
            if config.WARN:
                warnings.warn("workratio for block" + str(block_name) + "is zero")
        return work_ratio

    def get_workRatio_by_block_name_and_family_member_names_and_dir_eliminating_fake(self, block_name, family_member_names, dir_):
        blocks = [block_dir[0] for block_dir in self.block_dir_workRatio_dict.keys()]
        work_ratio = 0
        for block_dir, work_ratio_ in self.block_dir_workRatio_dict.items():
            if block_name == block_dir[0].instance_name and block_dir[1] == dir_:
                for family_member, work_ratio_value in work_ratio_.items():
                    if not (family_member in self.task.get_fake_family_name()) and (family_member in family_member_names):
                        work_ratio += work_ratio_value
        return work_ratio

    # get direction for a block
    def get_dir_of_block(self, block):
        for block_dir, work_ratio_ in self.block_dir_workRatio_dict.items():
            if block.instance_name == block_dir[0].instance_name:
                return block_dir[1]

    # get all the blocks that a task is mapped to .
    def get_blocks(self):
        return list(set([block_dir[0] for block_dir in (self.block_dir_workRatio_dict.keys())]))

    def channel_blocks(self):
        self.blocks_with_channels = []

    def get_blocks_with_channel(self):
        return self.blocks_with_channels


    def get_blocks_with_channel(self):
        exit(0)
        pass
        results = []
        for block_dir in (self.block_dir_workRatio_dict.keys()):
            block = block_dir[0]
            dir_ = block_dir[1]
            if block_dir[0].type == "ic":
                if (block, dir_) not in results:
                    results.append((block, dir_))
            elif block_dir[0].type == "mem":
                if (block, dir_) not in results:
                    results.append((block, dir_))
            elif block_dir[0].type == "pe":
                if (block, dir_) not in results:
                    results.append((block, dir_))
                #if (block, "same") not in results:
                #    results.append((block, "same"))

        return results

    # ------------------------------
    # Functionality
    #       get the PE (processing element, i.e., ip or general purpose processor or dsp) associated with the task
    # ------------------------------
    def get_pe_name(self):
        for block_dir in self.block_dir_workRatio_dict.keys():
            if self.block_dir_workRatio_dict[block_dir] == 1:
                return block_dir[0].block_instance_name

    # ------------------------------
    # Functionality
    #       print the task name and its corresponding blocks. Used for debugging purposes.
    # ------------------------------
    def print(self):
        print(self.task.name + str(list(map(lambda block: block.name, self.block_workRatio_dict.keys()))))


# This task maps a all the tasks within the workload to a processing blocks
class WorkloadToHardwareMap:
    def __init__(self, input_file="scheduling.json", mode=""):
        self.input_file = input_file  # input file to read the mapping from. TODO: this is not supported yet.

        # vars keeping status
        self.tasks_to_blocks_map_list:List[TaskToBlocksMap] = []  # list of all the tasks and the blocks they have mapped to.

        if mode == "from_json":  # TODO: this is not supported yet.
            self.populate_tasks_to_blocks_map_list_from_json()

    # ------------------------------
    # Functionality
    #       map the tasks to the blocks using information from an input json file. TODO: not supported yet.
    # ------------------------------
    def populate_tasks_to_blocks_map_list_from_json(self):
        raise Exception("deprecated function")
        with open(os.path.join(config.design_input_folder, self.input_file), 'r') as json_file:
            data = json.load(json_file)

        for task_data in data["workload_hardware_mapping"]:
            task_to_blocks_map = TaskToBlocksMap(task_data["task_name"], task_data["task_fraction"], \
                                                 task_data["task_fraction_index"]) #single task to blocks map
            for block in task_data["blocks"]:
                task_to_blocks_map.block_workRatio_dict[block] = block["work_ratio"]
            self.tasks_to_blocks_map_list.append(task_to_blocks_map)

    # ------------------------------
    # Functionality
    #       get a task within the workload.
    # Variables:
    #       task: task of interest.
    # ------------------------------
    def get_by_task(self, task: Task):
        for el in self.tasks_to_blocks_map_list:
            if el.task == task:
                return el
        return None

    # ------------------------------
    # Functionality
    #       get a task within the workload by its name.
    # Variables:
    #       task: task of interest.
    # ------------------------------
    def get_by_task_name(self, task_name: str): # we go with the task name for task name
        for el in self.tasks_to_blocks_map_list:
            if el.task.name == task_name:
                return el
        raise Exception("can have multiple or none task_to_blocks_maps for a single task")

    # ------------------------------
    # Functionality
    #       find out whether a task is already mapped.
    # Variables:
    #       task: task of interest.
    # ------------------------------
    def find_task(self, task:Task):
        for task_mapped in self.tasks_to_blocks_map_list:
            if task == task_mapped.task:
                return task

        raise Exception("couldn't find a task with the name " + str(task.name))

    # ------------------------------
    # Functionality
    #       get blocks associated with a task.
    # Variables:
    #       task: task of interest.
    # ------------------------------
    def get_blocks_associated_with_task(self, task:Task):
        return task.get_blocks()

    # ------------------------------
    # Functionality
    #       get all the tasks within the workload.
    # ------------------------------
    def get_tasks(self):
        tasks = []
        for el in self.tasks_to_blocks_map_list:
            tasks.append(el.task)
        return tasks

    # ------------------------------
    # Functionality
    #       get all the blocks used in mapping given their type
    # Variables:
    #   type: "type of the block (pe, mem, ic)"
    # ------------------------------
    def get_blocks_by_type(self, type):
        blocks = self.get_blocks()
        return list(filter(lambda x: x.type == type, blocks))

    # ------------------------------
    # Functionality
    #       get all the blocks used in mapping.
    # ------------------------------
    def get_blocks(self):
        blocks = []
        for task_to_blocks_map in self.tasks_to_blocks_map_list:
            task_to_blocks_map_blocks = task_to_blocks_map.get_blocks()
            for block in task_to_blocks_map_blocks:
                if block not in blocks:
                    blocks.append(block)
        return blocks

    # ------------------------------
    # Functionality
    #       get all tasks that are mapped to a block
    # Variables:
    #       block: block of interest.
    # ------------------------------
    def get_tasks_associated_with_block(self, block:Block):
        tasks = []
        for task_mapped in self.tasks_to_blocks_map_list:
            if block in task_mapped.get_blocks():
                tasks.append(task_mapped.task)
        return tasks

    # ------------------------------
    # Functionality
    #       get all the information in the mapping class. Used mainly for debugging purposes.
    # ------------------------------
    def print(self):
        for el in self.tasks_to_blocks_map_list:
            el.print()