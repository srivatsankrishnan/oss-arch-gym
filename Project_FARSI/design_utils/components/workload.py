#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import json
import os
from settings import config
from typing import Dict, List
import collections
import random
from datetime import datetime
import numpy as np
import time
import math


# This class is to model a task, that is the smallest software execution unit.
class Task:
    task_id_for_debugging_static = 0
    def __init__(self, name, work, iteration_ctr =1, type="latency_based", throughput_info = {}):
        self.iteration_ctr = iteration_ctr
        self.name = name
        self.progress = 0  # progress percentage (how much of the task has been finished)
        self.task_fraction = 1  # TODO: right now always set to 1
        self.task_fraction_index = 0  # TODO: always shunned to zero for now
        self.__children = []  # task children, i.e., the tasks that read from it
        self.__parents = []  # task parents, i.e., the tasks that write to the task
        self.__siblings = []  # tasks siblings, i.e., tasks with the same parents.
        self.PA_prop_dict = collections.OrderedDict()  # all the props used PA
        self.PA_prop_auto_tuning_list  = []  # list of auto tuned variables
        self.__task_to_family_task_work  = {}  # task to parent work-ratio
        self.__task_to_family_task_work[self] = work  # task self work
        self.task_id_for_debugging = self.task_id_for_debugging_static
        self.updated_task_work_for_debug = False
        self.__task_work_distribution = []  # Todo: double check and deprecate later
        self.__task_to_child_work_distribution = {}  #  double check and deprecate later
        self.__fake_children = []  # from the parent perspective, there is no new data transferred to children
                                   # but the reuse of the old generate data. So this is not used in
                                   # work ratio calculation

        self.__fake_parents = []  # from the parent perspective, there is no new data transferred to children
                                   # but the reuse of the old generate data. So this is not used in
                                   # work ratio calculation

        self.__task_to_family_task_work_unit = {}  # task to family task unit of work. For example
                                                   # work unit from bus and memory perspective is the burst size
                                                   # (in bytes)
        self.burst_size = config.default_burst_size

        self.type = type
        self.throughput_info = throughput_info

    def get_throughput_info(self):
        return self.throughput_info

    def get_iteration_cnt(self):
        return self.iteration_ctr

    def set_burst_size(self, burst_size):
        self.burst_size = burst_size

    def get_burst_size(self):
        return self.burst_size

    def get_name(self):
        return self.name
    # ---------------
    # Functionality:
    #       get a task's family tasks (tasks that it reads from).
    # Variables:
    #       task_name: name of the task.
    # ---------------
    def get_family_task_by_name(self, task_name):
        for task in self.__parents + self.__siblings + self.__children + [self]:
            if task.name == task_name:
                return task

    # ---------------
    # Functionality:
    #       resetting Platform Architect (PA) props. Used in PA design generation.
    #       task_name: name of the task.
    # ---------------
    def reset_PA_props(self):
        self.PA_prop_dict = collections.OrderedDict()

    # ---------------
    # Functionality:
    #       update Platform Architect (PA) props.
    # Variables:
    #       PA_prop_dict:  dictionary containing all the PA props
    # ---------------
    def update_PA_props(self, PA_prop_dict):
        self.PA_prop_dict.update(PA_prop_dict)

    def update_PA_auto_tunning_knob_list(self, prop_auto_tuning_list):
        self.PA_prop_auto_tuning_list = prop_auto_tuning_list

    # ---------------
    # Functionality:
    #       pick one off the children at random.
    # ---------------
    def sample_child(self):
        random.seed(datetime.now().microsecond)
        return random.choice(self.get_children())

    # ---------------
    # Functionality:
    #       sample the task distribution work. Used for jitter modeling/incorporation.
    # ---------------
    def sample_self_task_work(self):
        time.sleep(.00005)
        np.random.seed(datetime.now().microsecond)
        task_work = [task_work for task_work, work_prob in self.get_task_work_distribution()]
        work_prob = [work_prob for task_work, work_prob in self.get_task_work_distribution()]
        return np.random.choice(task_work, p=work_prob)

    # ---------------
    # Functionality:
    #       sample the task to child  work (how much data gets writen into the child task) distribution.
    #       Used for jitter modeling/incorporation.
    # Variables:
    #       child: task's child
    # ---------------
    def sample_self_to_child_task_work(self, child):
        np.random.seed(datetime.now().microsecond)
        task_to_child_work = [task_work for task_work, work_prob in self.get_task_to_child_work_distribution(child)]
        work_prob = [work_prob for task_work, work_prob in self.get_task_to_child_work_distribution(child)]
        return np.random.choice(task_to_child_work, p=work_prob)

    # ---------------
    # Functionality:
    #       update the task work (used in jitter modeling after a new work is assigned to the task).
    # Variables:
    #       self_work: work to assign to the task.
    # ---------------
    def update_task_work(self, self_work):
        delete_later_ =  self.get_self_task_work()
        self.updated_task_work_for_debug = True
        self.__task_to_family_task_work[self] = self_work
        delete_later = self.get_self_task_work()
        a = delete_later

    # ---------------
    # Functionality:
    #       update the task to child work (used in jitter modeling after a new work is assigned to the task).
    # Variables:
    #       child:  tasks's child.
    #       child_work: tasks's child work.
    # ---------------
    def update_task_to_child_work(self, child, child_work):
        self.__task_to_family_task_work[child] = child_work
        child.__task_to_family_task_work[self] = child_work
        if self.updated_task_work_for_debug:
            self.update_task_work_for_debug = False
            self.task_id_for_debugging +=1

    # ---------------
    # Functionality:
    #      add task to child work to the distribution of tasks. Used for jitter modeling.
    # Variables:
    #       child:  tasks's child.
    #       work: work to add to the distribution.
    # ---------------
    def add_task_to_child_work_distribution(self, child, work):
        self.__task_to_child_work_distribution[child] = work

    # ---------------
    # Functionality:
    #       add a parent(a task that it reads data from) for the task.
    # Variables:
    #       child:  tasks's child.
    #       work: works of the child.

    # ---------------
    def add_parent(self, parent,child_nature = "real"):
        self.__parents.append(parent)
        if child_nature == "fake":
            self.__fake_parents.append(parent)

    # add a child (a task that it writes to).
    # nature determines whether the work is real or fake. real means real generation of the data (which needs to be pass
    # along to the child) whereas fake means that the data has already been generated and just needs to be passed along.
    def add_child(self, child, work, child_nature="real"):
        self.__task_to_family_task_work[child] = work
        for other_child in self.__children:
            other_child.add_sibling(child)
            child.add_sibling(other_child)
        self.__children.append(child)
        child.add_parent(self, child_nature)
        child.__task_to_family_task_work[self] = work
        assert(child_nature in ["fake", "real"]), "child nature can only be fake or real but " + child_nature + " was given"
        if child_nature == "fake":
            self.__fake_children.append(child)

    # ---------------
    # Functionality:
    #       fake children are children that we  need to pass  data to, but we don't need to generate the data
    #       since it is already generate. This situation happens when two children use the same exact data
    # ---------------
    def get_fake_children(self):
        return self.__fake_children

    def get_fake_children_name(self):
        return [task.name for task in self.__fake_children]

    # ---------------
    # Functionality:
    #       fake parents are parents that pass data to, but don't need to generate the data
    #       since it is already generate.
    # ---------------
    def get_fake_parent_name(self):
        return [task.name for task in self.__fake_parents]

    def get_fake_family_name(self):
        return self.get_fake_children_name() + self.get_fake_parent_name()

    # ---------------
    # Functionality:
    #       remove a child (a task that it writes data to) for the task.
    # Variables:
    #       child:  tasks's child.
    # ---------------
    def remove_child(self, child):
        for other_child in self.__children:
            other_child.remove_sibling(child)
            child.remove_sibling(other_child)
        self.__children.remove(child)
        del self.__task_to_family_task_work[child]
        child.__parents.remove(self)
        del child.__task_to_family_task_work[self]


    def remove_parent(self, parent):
        self.__parents.remove(parent)
        del self.__task_to_family_task_work[parent]
        parent.__children.remove(self)
        del parent.__task_to_family_task_work[self]


    # ---------------
    # Functionality:
    #       add sibling (a task with the same parent) for the task.
    # Variables:
    #       task:  sibling task.
    # ---------------
    def add_sibling(self, task):
        if task not in self.__siblings:
            self.__siblings.append(task)

    # ---------------
    # Functionality:
    #       removing sibling (a task with the same parent) for the task.
    # Variables:
    #       task:  sibling task.
    # ---------------
    def remove_sibling(self, task):
        if task in self.__siblings:
            self.__siblings.remove(task)

    # ---------------
    # Functionality:
    #       get the relationship of the task with the input task.
    # Variables:
    #       task_:  the task to find the relationship for.
    # ---------------
    def get_relationship(self, task_):
        if any([task__.name == task_.name for task__ in self.__children]):
            return "child"
        elif any([task__.name == task_.name for task__ in self.__parents]):
            return "parent"
        elif any([task__.name == task_.name for task__ in self.__siblings]):
            return "sibling"
        elif task_.name == self.name:
            return "self"
        else:
            return "no relationship"

    # ---------------
    # Functionality:
    #       get tasks's work
    # ---------------
    def get_self_task_work(self):
        return self.__task_to_family_task_work[self]

    # ---------------
    # Functionality:
    #       get self to task family work (how much data is passed from/to the family task).
    # Variables:
    #       family_task: family task.
    # ---------------
    def get_self_to_family_task_work(self, family_task):
        if family_task in self.get_children():
            return self.__task_to_family_task_work[family_task]
        elif family_task in self.get_parents():
            return family_task.get_self_to_family_task_work(self)
        elif family_task == self:
            return self.get_self_task_work()
        else:
            print(family_task.name + " is not a family task of " + self.name)
            exit(0)

    def get_type(self):
        return self.type

    def get_self_total_work(self, mode):
        total_work = 0
        if mode == "execute":
            total_work = self.__task_to_family_task_work[self]
        if mode == "read":
            for family_task in self.get_parents():
                total_work += family_task.get_self_to_family_task_work(self)
        if mode == "write":
            for family_task in self.get_children():
                total_work += self.__task_to_family_task_work[family_task]
        return total_work


    # return self to family task unit of work. For example
    # work unit from bus and memory perspective is the burst size
    # (in bytes)
    def get_self_to_family_task_work_unit(self, family_task):
        return self.__task_to_family_task_work_unit[family_task]

    # determines what the dicing "grain" should be such that that
    # work unit (e.g., burst size) is respected.
    # Note that we ensure that smallest "read" (just as a convention) will respect the
    # burst-size. Everything else is adjusted accordingly
    def set_dice_factor(self, block_size):
        smallest_read = self.get_smallest_task_work_by_dir("read")
        smallest_write = self.get_smallest_task_work_by_dir("write")
        smallest_instructions = self.get_smallest_task_work_by_dir("loop")

        dice_factor = math.floor(smallest_read/block_size)  # use read,# this is just decided. Doesn't have to be this. Just had to pick something

        if dice_factor == 0:
            dice_factor = 1
        else:
            smallest_read_scaled = math.floor(smallest_read/dice_factor)
            smallest_write_scaled = math.floor(smallest_write/dice_factor)
            task_instructions_scaled = math.floor(smallest_instructions/dice_factor)

            if smallest_write_scaled == 0 or task_instructions_scaled == 0:
                dice_factor = 1
        return dice_factor

    # based on the some reference work unit (same as block_size) determine the rest of the
    # work units.
    def calc_work_unit(self):
        dice_factor = self.set_dice_factor(self.burst_size)
        for family in self.get_family():
            self.__task_to_family_task_work_unit[family] = int(self.get_self_to_family_task_work(family)/dice_factor)
            assert(self.get_self_to_family_task_work(family)/dice_factor > .1)

    def get_smallest_task_work_by_dir(self, dir):
        if dir == "write":
            family_tasks = self.get_children()
        elif dir == "read":
            family_tasks = self.get_parents()
        elif dir == "loop":
            family_tasks = [self]

        if "souurce" in self.name:
            return 0
        if "siink" in self.name:
            return 0

        if len(family_tasks) == 0:
            print("what")

        return min([self.get_self_to_family_task_work(task_) for task_ in family_tasks])

    def get_biggest_task_work_by_dir(self, dir):
        if dir == "write":
            family_tasks = self.get_children()
        elif dir == "read":
            family_tasks = self.get_parents()
        elif dir == "loop":
            family_tasks = [self]

        if "souurce" in self.name:
            return 0
        if "siink" in self.name:
            return 0
        return max([self.get_self_to_family_task_work(task_) for task_ in family_tasks])



    def get_smallest_task_work_unit_by_dir(self, dir):
        if dir == "write":
            family_tasks = self.get_children()
        elif dir == "read":
            family_tasks = self.get_parents()
        elif dir == "loop":
            family_tasks = [self]

        if "souurce" in self.name:
            return 0
        if "siink" in self.name:
            return 0
        return min([self.get_self_to_family_task_work_unit(task_) for task_ in family_tasks])



    # ---------------
    # Functionality:
    #       add task's work to the distribution work.
    # Variables:
    #       work: new work to add to the distribution.
    # ---------------
    def add_task_work_distribution(self, work):
        self.task_work_distribution = work

    # ---------------
    # Functionality:
    #       get task's work distribution.
    # ---------------
    def get_task_work_distribution(self):
        return self.task_work_distribution

    # ---------------
    # Functionality:
    #       get task's to child work distribution.
    # ---------------
    def get_task_to_child_work_distribution(self, child):
        return self.__task_to_child_work_distribution[child]

    # ---------------
    # Functionality:
    #       get the work ratio (how much data is written to/ read from) for the family task.
    # Variables:
    #       family_task_name: name of the family (parent/child) task.
    # ---------------
    def get_work_ratio_by_family_task_name(self, family_task_name):
        family_task = self.get_family_task_by_name(family_task_name)
        return self.get_work_ratio(family_task)

    # ---------------
    # Functionality:
    #       get the work ratio (how much data is written to/ read from) for the family task.
    # Variables:
    #       family_task: name of the family (parent/child) task.
    # ---------------
    def get_work_ratio(self, family_task):
        """
        if not (self.task_id_for_debugging == self.task_id_for_debugging_static):
            print("debugging not matching")
            exit(0)
        """
        if self.get_self_task_work() == 0: # dummy tasks
            return 1
        return self.get_self_to_family_task_work(family_task)/self.get_self_task_work()

    # ---------------
    # Functionality:
    #      getters
    # ---------------
    def get_children(self):
        return self.__children

    def get_parents(self):
        return self.__parents

    def get_siblings(self):
        return self.__siblings

    def get_family(self):
        return self.__parents + self.__children

    def is_task_dummy(self):
        return "souurce" in self.name or "siink" in self.name or "dummy_last" in self.name


# Task Graph for the workload.
class TaskGraph:
    def __init__(self, tasks):
        self.__tasks = tasks
        _ = [task_.calc_work_unit() for task_ in self.__tasks]

    # -----------
    # Functionality:
    #       get the root of the task graph.
    # -----------
    def get_root(self):
        roots = []
        for task in self.__tasks:
            if not task.get_parents():
                roots.append(task)
        if not(len(roots)== 1):
            print("weird")
        assert(len(roots) == 1), "must have only one task at the top of the dep graph. added a dummy otherwise to do this"
        return roots[0]

    def get_all_tasks(self):
        return self.__tasks


    def get_task_by_name(self, name):
        for tsk in self.__tasks:
            if tsk.name == name:
                return tsk

        print("task with the name " +  name  + " does not exist")
        exit(0)

    # -----------
    # Functionality:
    #       get task's parents (task that it reads from)
    # -----------
    def get_task_s_parents(self, task):
        return task.get_parents()

    # -----------
    # Functionality:
    #       get task's parents name
    # Variables:
    #       task_name: name of the task
    # -----------
    def get_task_s_parent_by_name(self, task_name):
        for task in self.__tasks:
            if task.name == task_name:
                return task.get_parents()
        raise Exception("task:" + task.name + " not in the task graph")

    # -----------
    # Functionality:
    #       get task's children
    # Variables:
    #       task_name: name of the task
    # -----------
    def get_task_s_children(self, task):
        return task.get_children()

    # -----------
    # Functionality:
    #       get task's children  by task name
    # Variables:
    #       task_name: name of the task
    # -----------
    def get_task_s_children_by_task_name(self, task_name):
        for task in self.__tasks:
            if task.name == task_name:
                return task.get_children()
        raise Exception("task:" + task.name + " not in the task graph")

    #  determine whether two tasks can run in parallel or not
    def task_can_run_in_parallel_helper(self, task_1, task_2, task_to_look_at, root):
        if task_to_look_at == task_2:
            return False
        elif task_to_look_at == root:
            return True

        result  = []
        for task in task_to_look_at.get_parents():
            result.append(self.task_can_run_in_parallel_helper(task_1, task_2, task, root))
            if not all(result):
                return False
        return True

    # ---------------
    # Functionality:
    #       establish if tasks can run in parallel or not .
    # ---------------
    def tasks_can_run_in_parallel(self, task_1, task_2):
        root = self.get_root()
        return (self.task_can_run_in_parallel_helper(task_1, task_2, task_1, root) and
                self.task_can_run_in_parallel_helper(task_2, task_1, task_2, root))


# This class emulates the software workload containing the task set.
class Workload:
    def __init__(self, input_file="workload.json", mode=""):
        self.tasks = []  # set of tasks.
        self.input_file = input_file  # if task reads from a file to be populated. Not supported yet.
        if mode == "from_json":
            self.populate_tasks()

    # -----------
    # Functionality:
    #      populate the tasks from a file. To be finished in the next round.
    # -----------
    def populate_tasks(self):
        raise Exception('not supported any more')
        with open(os.path.join(config.design_input_folder, self.input_file), 'r') as json_file:
            data = json.load(json_file)

        for datum in data["workload"]:
            self.tasks.append(Task(datum["task_name"], datum["work"]))

    # -----------
    # Functionality:
    #       get a task that has the similar name as the input task name. Used when we duplicate a design.
    # -----------
    def get_task_by_name(self, task):
        for task_ in self.tasks:
            if task.name == task_.name:
                return task_
        raise Exception("task instance with name:" + str(task.name) + " doesn't exist in this workload")