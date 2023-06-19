#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import json
import os
from settings import config
from design_utils.components.workload import *


# This class sets schedules for task.
class TaskToPEBlockSchedule:  #information about one task's schedule
    def __init__(self, task, starting_time):
        self.task = task
        self.starting_time = starting_time

    # ------------------------------
    # Functionality
    #       set the starting time of the task
    # Variables:
    #       starting_time: starting time of the task
    # ------------------------------
    def reschedule(self, starting_time):
        self.starting_time = starting_time


# this class sets schedules for the whole workload
class WorkloadToPEBlockSchedule:
    def __init__(self, input_file="scheduling.json", mode=""):
        self.input_file = input_file  # input file to read the schedules from
        self.task_to_pe_block_schedule_list_sorted = []  # list of tasks sorted based on their starting time
        if mode == "from_json": # TODO: this is not supported yet
            self.populate_task_to_pe_block_schedule_list()

    # ------------------------------
    # Functionality
    #       set the  schedule for the whole workload.
    # ------------------------------
    def populate_task_to_pe_block_schedule_list(self):
        raise Exception("not supporting this anymore")
        with open(os.path.join(config.design_input_folder, self.input_file), 'r') as json_file:
            data = json.load(json_file)

        task_to_pe_block_schedule_list_unsorted = []  # containing the unsorted scheduling units

    # ------------------------------
    # Functionality
    #       schedule out a task and schedule in another task on to a PE (processing element)
    # Variables:
    #       old_task_to_pe_block_schedule: task to schedule out
    #       new_task_to_pe_block_schedule: task to schedule in
    # ------------------------------
    def swap_task_to_pe_block_schedule(self, old_task_to_pe_block_schedule, new_task_to_pe_block_schedule):
        self.task_to_pe_block_schedule_list_sorted.remove(old_task_to_pe_block_schedule)
        self.task_to_pe_block_schedule_list_sorted.append(new_task_to_pe_block_schedule)
        self.task_to_pe_block_schedule_list_sorted = sorted(self.task_to_pe_block_schedule_list_sorted,
                                                      key=lambda schedule: schedule.starting_time)

    # ------------------------------
    # Functionality
    #       get a task from the list of tasks
    # Variables:
    #       task: task of interest
    # ------------------------------
    def get_by_task(self, task: Task): # we go with the task name for task name
        for el in self.task_to_pe_block_schedule_list_sorted:
            if task == el.task:
                return el
        raise Exception("too many or none tasks scheduled with " + task.name + "name")

    # ------------------------------
    # Functionality
    #       get a task from the list of tasks by its name
    # Variables:
    #       task: task of interest
    # ------------------------------
    def get_by_task_name(self, task: Task):  # we go with the task name for task name
        for el in self.task_to_pe_block_schedule_list_sorted:
            if task.name == el.task.name:
                return el
        raise Exception("too many or none tasks scheduled with " + task.name + "name")