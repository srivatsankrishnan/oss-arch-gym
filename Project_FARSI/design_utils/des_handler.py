#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

from design_utils.design import  *
from settings import config
from specs.data_base import *
from specs.LW_cl import *
import time
import random
from datetime import datetime
from typing import Dict, List, Tuple
from design_utils.components.hardware import *
from error_handling.custom_error import  *
import importlib
from DSE_utils.exhaustive_DSE import *
from visualization_utils.vis_hardware import *
import _pickle as cPickle

# This class allows us to modify the design. Each design is applied
# a move to get transformed to another.
# Move at the moment has 4 different parts (metric, kernel, block, transformation) that needs to be
# set
class move:
    def __init__(self, transformation_name, transformation_sub_name, batch_mode, dir, metric, blck, krnel, krnl_prob_dict_sorted):
        self.workload = ""
        self.transformation_name = transformation_name
        self.parallelism_type = []
        self.locality_type = []
        self.transformation_sub_name = transformation_sub_name
        self.customization_type = ""
        self.metric = metric
        self.batch_mode = batch_mode
        self.dir = dir
        self.blck = blck
        self.krnel = krnel
        self.tasks = "_"
        self.dest_block = "_"
        self.valid = True
        self.breadth = 0
        self.depth = 0
        self.mini_breadth = 0
        self.sorted_kernels = []
        self.sorted_blocks = []
        self.kernel_rnk_to_consider = 0
        self.sorted_metrics = []
        self.ref_des_dist_to_goal_all = 0
        self.ref_des_dist_to_goal_non_cost = 0
        self.cost = 0
        self.sorted_metric_dir = {}
        self.pre_move_ex = None
        self.moved_ex = None
        self.validity_meta_data = ""
        self.krnel_prob_dict_sorted = krnl_prob_dict_sorted
        self.generation_time = 0
        self.system_improvement_dict = {}
        #self.populate_system_improvement_log()

        self.transformation_selection_time = 0
        self.block_selection_time = 0
        self.kernel_selection_time = 0
        self.pickling_time = 0

        self.design_space_size = {"pe_mapping": 0 , "ic_mapping":0, "mem_mapping":0,
                                  "pe_allocation": 0, "ic_allocation": 0, "mem_allocation": 0,
                                  "transfer":0, "routing":0, "hardening":0, "softening":0,
                                    "bus_width_modulation":0, "mem_frequency_modulation":0, "ic_frequency_modulation":0, "pe_frequency_modulation":0, "identity":0,
                                  "loop_iteration_modulation":0
                                  }


    def get_design_space_size(self):
        return self.design_space_size

    def get_system_improvement_log(self):
        return self.system_improvement_dict

    def populate_system_improvement_log(self):
        # communication vs computation
        if self.get_block_ref().type in ["ic","mem"]:
            comm_comp = "comm"
        else:
            comm_comp = "comp"

        # which exact optimization targgeted: topology/mapping/tunning
        if self.get_transformation_name() in ["swap"]:
            exact_optimization = self.get_customization_type(self.get_block_ref(), self.get_des_block())
        elif self.get_transformation_name() in ["split_swap"]:
            if self.get_block_ref() == "pe":
                exact_optimization = self.get_customization_type(self.get_block_ref(), self.get_des_block()) +";" +self.get_block_ref().type +"_"+"allocation"
            else:
                exact_optimization = self.get_customization_type(self.get_block_ref(), self.get_des_block())
        elif self.get_transformation_name() in ["migrate"]:
            exact_optimization = self.get_block_ref().type +"_"+"mapping"
        elif self.get_transformation_name() in ["split_swap", "split", "transfer", "routing"]:
            exact_optimization = self.get_block_ref().type +"_"+"allocation"
        elif self.get_transformation_name() in ["transfer", "routing"]:
            exact_optimization = self.get_block_ref().type +"_"+self.get_transformation_name()
        elif self.get_transformation_name() in ["cost"]:
            exact_optimization = "cost"
        elif self.get_transformation_name() in ["dram_fix"]:
            exact_optimization = "dram_fix"
        elif self.get_transformation_name() in ["identity"]:
            exact_optimization = "identity"
        else:
            print(self.get_transformation_name() + " high level optimization is not specified")
            exit(0)


        # which high level optimization targgeted: topology/mapping/tunning
        if self.get_transformation_name() in ["swap", "split_swap"]:
            high_level_optimization = "hardware_tunning"
        elif self.get_transformation_name() in ["split_swap"]:
            high_level_optimization = "hardware_tunning;topology"
        elif self.get_transformation_name() in ["migrate"]:
            high_level_optimization = "mapping"
        elif self.get_transformation_name() in ["split_swap", "split", "transfer","routing"]:
            high_level_optimization = "topology"
        elif self.get_transformation_name() in ["cost"]:
            high_level_optimization = "cost"
        elif self.get_transformation_name() in ["dram_fix"]:
            high_level_optimization = "dram_fix"
        elif self.get_transformation_name() in ["identity"]:
            high_level_optimization = "identity"
        else:
            print(self.get_transformation_name() + " high level optimization is not specified")
            exit(0)

        # which architectural variable targgeted: topology/mapping/tunning
        if self.get_transformation_name() in ["split"]:
            architectural_principle = ""
            for el in self.parallelism_type:
                architectural_principle +=  el+";"
            architectural_principle = architectural_principle[:-1]
        elif self.get_transformation_name() in ["migrate"]:
            architectural_principle = ""
            for el in self.parallelism_type:
                architectural_principle +=  el+";"
            for el in self.locality_type:
                architectural_principle +=  el+";"
            architectural_principle = architectural_principle[:-1]
        elif self.get_transformation_name() in ["split_swap", "swap"]:
            if "loop_iteration_modulation" in  exact_optimization:
                architectural_principle = "loop_level_parallelism"
            else:
                architectural_principle = "customization"
        elif self.get_transformation_name() in ["transfer","routing"]:
            architectural_principle = "spatial_locality"
        elif self.get_transformation_name() in ["cost"]:
            architectural_principle = "cost"
        elif self.get_transformation_name() in ["dram_fix"]:
            architectural_principle = "dram_fix"
        elif self.get_transformation_name() in ["identity"]:
            architectural_principle = "identity"
        else:
            print(self.get_transformation_name() + " high level optimization is not specified")
            exit(0)

        self.system_improvement_dict["comm_comp"] = comm_comp
        self.system_improvement_dict["high_level_optimization"] = high_level_optimization
        self.system_improvement_dict["exact_optimization"] = exact_optimization
        self.system_improvement_dict["architectural_principle"] = architectural_principle


    def set_parallelism_type(self, parallelism_type):
        self.parallelism_type = parallelism_type

    def set_locality_type(self, locality_type):
        self.locality_type = locality_type


    def get_transformation_sub_name(self):
        return self.transformation_sub_name

    def set_krnel_ref(self, krnel):
        self.krnel = krnel

    # how long did it take to come up with the move
    def set_generation_time(self, generation_time):
        self.generation_time = generation_time

    def get_generation_time(self):
        return self.generation_time

    def set_logs(self, data, type_):
        if type_ == "cost":
            self.cost = data
        if type_ == "workload":
            self.workload = data
        if type_ == "pickling_time":
            self.pickling_time = data
        if type_ == "metric_selection_time":
            self.metric_selection_time = data
        if type_ == "dir_selection_time":
            self.dir_selection_time = data
        if type_ == "kernel_selection_time":
            self.kernel_selection_time = data
        if type_ == "block_selection_time":
            self.block_selection_time = data
        if type_ == "transformation_selection_time":
            self.transformation_selection_time = data
        if type_ == "kernels":
            self.sorted_kernels = data
        if type_ == "blocks":
            self.sorted_blocks = data
        if type_ == "metrics":
            self.sorted_metrics = data
        if type_ == "kernel_rnk_to_consider":
            self.kernel_rnk_to_consider = data
        if type_ == "ref_des_dist_to_goal_all":
            self.ref_des_dist_to_goal_all = data
        if type_ == "ref_des_dist_to_goal_non_cost":
            self.ref_des_dist_to_goal_non_cost = data

    def get_transformation_batch(self):
        return self.batch_mode

    def get_logs(self, type_):
        if type_ == "workload":
            return self.workload
        if type_ == "cost":
            return self.cost
        if type_ == "kernels":
            return self.sorted_kernels
        if type_ == "blocks":
            return self.sorted_blocks
        if type_ == "metrics":
            return self.sorted_metrics
        if type_ == "kernel_rnk_to_consider":
            return self.kernel_rnk_to_consider
        if type_ == "ref_des_dist_to_goal_all":
            return self.ref_des_dist_to_goal_all
        if type_ == "ref_des_dist_to_goal_non_cost":
            return self.ref_des_dist_to_goal_non_cost
        if type_ == "pickling_time":
            return self.pickling_time
        if type_ == "metric_selection_time":
            return self.metric_selection_time
        if type_ == "dir_selection_time":
            return self.dir_selection_time
        if type_ == "kernel_selection_time":
            return self.kernel_selection_time
        if type_ == "block_selection_time":
            return self.block_selection_time
        if type_ == "transformation_selection_time":
            return self.transformation_selection_time


    # depth and breadth determine how many designs to generate around (breadth) and
    # chain from (breadth) from the current design
    def set_breadth_depth(self, breadth, depth, mini_breadth):
        self.breadth = breadth
        self.depth = depth
        self.mini_breadth = mini_breadth

    def get_depth(self):
        return self.depth

    def get_mini_breadth(self):
        return self.mini_breadth

    def get_breadth(self):
        return self.breadth

    def set_metric(self, metric):
        self.metric = metric

    # the block to target in the move
    def set_ref_block(self, blck):
        self.blck = blck

    # the task to target in the move
    def set_tasks(self, tasks_):
        self.tasks = tasks_

    # the transformation to target in the move
    def set_transformation(self, trans):
        self.transformation_name = trans

    # the transformation to target in the move
    def set_transformation_sub_name(self, trans_sub_name):
        self.transformation_sub_name = trans_sub_name

    # set this to specify that the move's parameters are fully set
    # and hence the move is ready to be applied
    def set_validity(self, validity, reason=""):
        self.valid = validity
        self.validity_meta_data = reason

    def is_valid(self):
        return self.valid

    # set the block that we will change the ref_block to.
    def set_dest_block(self, block_):
        self.dest_block = block_


    def get_customization_type(self, ref_block, imm_block):
        return self.customization_type

    # set the block that we will change the ref_block to.
    def set_customization_type(self, ref_block, imm_block):
        if not ref_block.subtype  == imm_block.subtype:
            # type difference
            if ref_block.type == "pe":
                if ref_block.subtype == "gpp" and imm_block.subtype =="ip":
                    self.customization_type = "hardening"
                elif ref_block.subtype == "ip" and imm_block.subtype =="gpp":
                    self.customization_type = "softening"
                else:
                    self.customization_type = "unknown"
                    #print("we should have coverred all the customizations. what is missing then (1)")
                    #exit(0)
            elif ref_block.type == "mem":
                #self.customization_type = "memory_cell_ref_block.subtype +"_to_"  + imm_block.subtype
                self.customization_type = "mem_allocation"
            else:
                self.customization_type = "unknown"
                # print("we should have coverred all the customizations. what is missing then (1)")
                # exit(0)
        else:
            if not ref_block.get_loop_itr_cnt() == imm_block.get_loop_itr_cnt():
                self.customization_type = "loop_iteration_modulation"
            elif not ref_block.get_block_freq() == imm_block.get_block_freq():
                self.customization_type = ref_block.type+"_"+"frequency_modulation"
            elif not ref_block.get_block_bus_width() == imm_block.get_block_bus_width():
                self.customization_type = "bus_width_modulation"
            else:
                self.customization_type = "unknown"
                # print("we should have coverred all the customizations. what is missing then (1)")
                # exit(0)


    def get_block_attr(self, selected_metric):
        if selected_metric == "latency":
            selected_metric_to_sort = 'peak_work_rate'
        elif selected_metric == "power":
            #selected_metric_to_sort = 'work_over_energy'
            selected_metric_to_sort = 'one_over_power'
        elif selected_metric == "area":
            selected_metric_to_sort = 'one_over_area'
        else:
            print("selected_selected_metric: " + selected_metric + " is not defined")
        return selected_metric_to_sort


    # --------------------------------------
    # getters.
    # PS: look in to the equivalent setters to understand the parameters
    # --------------------------------------
    def get_tasks(self):
        return self.tasks

    def get_des_block(self):
        return self.dest_block

    def get_des_block_name(self):
        if self.dest_block == "_":
            return "_"
        else:
           return self.dest_block.instance_name

    def get_dir(self):
        assert(not(self.dir == "deadbeef")), "dir is not initialized"
        return self.dir

    def get_metric(self):
        assert(not(self.metric == "deadbeef")), "metric is not initialized"
        return self.metric

    def set_sorted_metric_dir(self, sorted_metric_dir):
        self.sorted_metric_dir = sorted_metric_dir

    def get_sorted_metric_dir(self):
        return self.sorted_metric_dir

    def get_transformation_name(self):
        assert(not(self.transformation_name == "deadbeef")), "name is not initialized"
        return self.transformation_name

    def get_block_ref(self):
        assert(not(self.blck=="deadbeef")), "block is not initialized"
        return self.blck

    def get_kernel_ref(self):
        assert (not (self.krnel == "deadbeef")), "block is not initialized"
        return self.krnel

    def print_info(self, mode="all"):
        if mode ==  "all":
            print("info:" + " tp::" + self.get_transformation_name() + ", mtrc::" + self.get_metric() + ", blck_ref::" +
                  self.get_block_ref().instance_name + ", block_des:" + self.get_des_block_name()+
                  ", tsk:" + self.get_kernel_ref().get_task().name)
        else:
            print("mode:" + mode + " is not supported for move printing")

    # see if you can apply the move
    # this is different than sanity check which checks whether the applied move messed up the design
    def safety_check(self, ex):
        return True


    # Check the validity of the move.
    def validity_check(self):
        if not self.is_valid():
            if self.validity_meta_data == "NoMigrantException":
                raise NoMigrantException
            elif self.validity_meta_data == "ICMigrationException":
                raise ICMigrationException
            elif self.validity_meta_data == "CostPairingException":
                raise CostPairingException
            elif self.validity_meta_data == "NoAbsorbee(er)Exception":
                raise NoAbException
            elif self.validity_meta_data == "TransferException":
                raise TransferException
            elif self.validity_meta_data == "RoutingException":
                raise RoutingException
            elif self.validity_meta_data == "IPSplitException":
                raise IPSplitException
            elif self.validity_meta_data == "NoValidTransformationException":
                raise NoValidTransformationException
            elif self.validity_meta_data == "NoParallelTaskException":
                raise NoParallelTaskException
            else:
                print("this invalidity reason is not supported" + self.validity_meta_data)
                exit(0)

    # Log the design before and after for post-processing
    def set_before_after_designs(self, pre_moved_ex, moved_ex):
        self.pre_move_ex = pre_moved_ex
        self.moved_ex = moved_ex

    # note that since we make a copy of hte design (and call it prev_des)
    # the instance_names are not gonna be exactly the same
    # Variables:
    #       pre_moved_ex: example design pre moving (transformation)
    #       moved_ex: example design after moving (transformation)
    #       mode = {"pre_application", "after_application"} # pre means before applying the move
    def sanity_check(self):
        pre_moved_ex = self.pre_move_ex
        moved_ex = self.moved_ex
        insanity_list = []

        # ---------------------
        # pre/post design are not specified. This is an indicator that the move was not really applied
        # which is caused by the errors when apply the move
        # ---------------------
        if moved_ex is None or pre_moved_ex is None:
            insanity = Insanity("_", "_", "no_design_provided")
            print(insanity.gen_msg())
            self.set_validity(False)
            raise MoveNoDesignException
            #return False

        # ---------------------
        # number of fronts sanity check
        # ---------------------
        pre_mvd_fronts_1 = sum([len(block.get_fronts("task_name_dir")) for block in pre_moved_ex.get_blocks()])
        pre_mvd_fronts_2 = sum([len(block.get_fronts("task_dir_work_ratio")) for block in pre_moved_ex.get_blocks()])
        if not pre_mvd_fronts_1 == pre_mvd_fronts_2:
            pre_mvd_fronts_1 = [block.get_fronts("task_name_dir") for block in pre_moved_ex.get_blocks()]
            pre_mvd_fronts_2 = [block.get_fronts("task_dir_work_ratio") for block in pre_moved_ex.get_blocks()]
            raise UnEqualFrontsError

        mvd_fronts_1 = sum([len(block.get_fronts("task_name_dir")) for block in moved_ex.get_blocks()])
        mvd_fronts_2 = sum([len(block.get_fronts("task_dir_work_ratio")) for block in moved_ex.get_blocks()])
        if not mvd_fronts_1 == mvd_fronts_2:
            mvd_fronts_1 = [block.get_fronts("task_name_dir") for block in moved_ex.get_blocks()]
            mvd_fronts_2 = [block.get_fronts("task_dir_work_ratio") for block in moved_ex.get_blocks()]
            raise UnEqualFrontsError

        # ---------------------
        # block count sanity checks
        # ---------------------
        if self.get_transformation_name() == "split":
            if self.get_block_ref().type == "ic":
                if not (len(moved_ex.get_blocks()) in [len(pre_moved_ex.get_blocks()),
                                                       # when can't succesfully split
                                                       len(pre_moved_ex.get_blocks()) + 1,
                                                       len(pre_moved_ex.get_blocks()) + 2,
                                                       len(pre_moved_ex.get_blocks()) + 3]):
                    insanity = Insanity("_", "_", "block_count_deviation")
                    insanity_list.append(insanity)
                    print("previous block count:" + str(
                        len(pre_moved_ex.get_blocks())) + " moved_ex block count:" + str(
                        len(moved_ex.get_blocks())))
                    print(insanity.gen_msg())
                    self.set_validity(False)
                    raise BlockCountDeviationError
            else:
                if not (len(moved_ex.get_blocks()) in [len(pre_moved_ex.get_blocks()),
                                                       # when can't successfully split
                                                       len(pre_moved_ex.get_blocks()) + 1]):
                    insanity = Insanity("_", "_", "block_count_deviation")
                    insanity_list.append(insanity)
                    print("previous block count:" + str(
                        len(pre_moved_ex.get_blocks())) + " moved_ex block count:" + str(
                        len(moved_ex.get_blocks())))
                    print(insanity.gen_msg())
                    self.set_validity(False)
                    raise BlockCountDeviationError
        elif self.get_transformation_name() in ["migrate"]:
            if not (len(moved_ex.get_blocks()) in [len(pre_moved_ex.get_blocks()),
                                                   len(pre_moved_ex.get_blocks()) - 1]):
                insanity = Insanity("_", "_", "block_count_deviation")
                insanity_list.append(insanity)
                print(
                    "previous block count:" + str(len(pre_moved_ex.get_blocks())) + " moved_ex block count:" + str(
                        len(moved_ex.get_blocks())))
                print(insanity.gen_msg())
                self.set_validity(False)
                raise BlockCountDeviationError
        elif self.get_transformation_name() in ["swap"]:
            if not (len(pre_moved_ex.get_blocks()) == len(moved_ex.get_blocks())):
                insanity = Insanity("_", "_", "block_count_deviation")
                insanity_list.append(insanity)
                print(
                    "previous block count:" + str(len(pre_moved_ex.get_blocks())) + " moved_ex block count:" + str(
                        len(moved_ex.get_blocks())))
                print(insanity.gen_msg())
                self.set_validity(False)
                raise BlockCountDeviationError

        # ---------------------
        # disconnection check
        # ---------------------
        if self.get_transformation_name() == "swap":
            if len(self.get_block_ref().neighs) > 0:
                insanity = Insanity("_", "_", "incomplete_swap")
                insanity_list.append(insanity)
                print("block" + move.get_block_ref().instance_name + " wasn't completely disconnected:")
                print(insanity.gen_msg())
                self.set_validity(False)
                raise IncompleteSwapError

        return


# This class takes care of instantiating, sampling, modifying the design objects.
# Used within by the design exploration framework to generate neighbouring design points.
class DesignHandler:
    def __init__(self, database):
        # instantiate a database object (this object contains all the information in the data)
        self.database = database  # hardware/software database used for design selection.
        self.__tasks = database.get_tasks()  # software tasks to include in the design.
        self.pruned_one = True     # used for moves that include adding a NOC which requires reconnecting
                                   # memory and processing elements (reconnect = prune and connnect)
        self.boost_SOC = False  # for multie SOC designs. Not activated yet.
        self.DMA_task_ctr = 0  # number of DMA engines used
        if config.FARSI_performance == "fast":
            self.get_immediate_block = self.get_immediate_block_fast
            self.get_immediate_block_multi_metric = self.get_immediate_block_multi_metric_fast
            self.get_equal_immediate_blocks_present = self.get_equal_immediate_blocks_present_fast
        else:
            self.get_immediate_block = self.get_immediate_block_slow
            self.get_equal_immediate_block_present = self.get_equal_immediate_block_present_slow

    # -------------------------------------------
    # Functionality:
    #       loading (Mapping) and unloading tasks to the blocks.
    # Variables:
    #       pe: processing element to load.
    #       mem: Memory element to load.
    #       tasks: tasks set (within the workload) to load PE and MEM with.
    # -----------------------------------------
    # only have one pe and write mem, so easy. TODO: how about multiple ones
    def load_tasks_to_pe_and_write_mem(self, pe, mem, tasks):  # is used only for initializing
        get_work_ratio = self.database.get_block_work_ratio_by_task_dir
        _ = [pe.load_improved(task, task) for task in tasks]  # load PEs with all the tasks
        for task in tasks:
            for task_child in task.get_children():
                mem.load_improved(task, task_child)  # load memory with tasks

    # only have one pe and write mem, so easy. TODO: how about multiple ones
    def load_tasks_to_pe_and_write_mem_for_hops(self, pes, mems, tasks, num_of_hops):  # is used only for initializing
        get_work_ratio = self.database.get_block_work_ratio_by_task_dir

        # souurce/siink
        task = tasks[0]
        pes[0].load_improved(task, task)
        for task_child in task.get_children():
            mems[0].load_improved(task, task_child)  # load memory with tasks
        task = tasks[-1]
        pes[-1].load_improved(task, task)
        for task_child in task.get_children():
            mems[-1].load_improved(task, task_child)  # load memory with tasks


        last_task = 1
        for i in range(1, len(tasks) - (num_of_hops-2) -1):
            task = tasks[i]
            if (i % 2) == 0:
                pe_mem_idx =0
            else:
                pe_mem_idx = len(pes) -1

            pes[pe_mem_idx].load_improved(task, task)
            for task_child in task.get_children():
                mems[pe_mem_idx].load_improved(task, task_child)  # load memory with tasks
            last_task +=1

        # dummy tasks in the middle
        idx = 1
        for i in range(last_task, len(tasks)-1):
            task = tasks[i]
            pes[idx].load_improved(task, task)
            for task_child in task.get_children():
                mems[idx].load_improved(task, task_child)  # load memory with tasks
            idx +=1


    # ------------------------------
    # Functionality:
    #       loading (Mapping) and unloading tasks to reading memory/ICs blocks.
    #       Assigning read mem means to find the parent task's writing mem
    #       routing means assigning (loading a bus with a task) (the already existing) bus to a task such that
    #       for it's read rout its the fastest rout from pe to read_mem and
    #       for it's write rout its the fastest rout from pe to write_mem.
    #       note that these two conditions are independently considered
    # Variables:
    #       ex_dp: example design.
    # ------------------------------
    def load_tasks_to_read_mem_and_ic(self, ex_dp):
        # assign buses (for both read and write) and mem for read
        self.load_read_mem_and_ic_recursive(ex_dp, [], ex_dp.get_hardware_graph().get_task_graph().get_root(), [], None)
        # prune whatever ic connection where there is no traffic on it
        self.disconnect_ics_with_no_shared_task(ex_dp)


    def disconnect_ics_with_no_shared_task(self, ex_dp):
        design_ics = [blck for blck in ex_dp.get_blocks() if blck.type == "ic"]
        for ic in design_ics:
            ic_tasks = [el.get_name() for el in ic.get_tasks_of_block()]
            ic_neighs = [neigh for neigh in ic.get_neighs() if neigh.type=="ic"]
            for neigh in ic_neighs:
                neigh_tasks = [el.get_name() for el in neigh.get_tasks_of_block()]
                no_task_shared = len(ic_tasks) == len(list(set(ic_tasks) - set(neigh_tasks)))
                if no_task_shared:
                    neigh.disconnect(ic)
                    print("=========== ==========")
                    print("=========== disconnected a path==========")
                    print("=========== ==========")



    # check if there is another task (on the block that can run in parallel with the task of interest
    def find_parallel_tasks_of_task_in_block(self, ex_dp, sim_dp, task, block):
        task_synced = [task__ for task__ in block.get_tasks_of_block() if task__.name == task.name][0]
        parallel_tasks = []
        tasks_of_block = [task_ for task_ in block.get_tasks_of_block() if (not ("souurce" in task_.name) or not ("siink" in task_.name))]

        if config.parallelism_analysis == "static":
            for task_ in tasks_of_block:
                if ex_dp.get_hardware_graph().get_task_graph().tasks_can_run_in_parallel(task_, task_synced):
                    parallel_tasks.append(task_)
        elif config.parallelism_analysis == "dynamic":
            parallel_tasks_names_ = sim_dp.get_dp_rep().get_tasks_parallel_task_dynamically(task)
            tasks_using_diff_pipe_cluster = sim_dp.get_dp_rep().get_tasks_using_the_different_pipe_cluster(task, block)

            parallel_tasks_names= list(set(parallel_tasks_names_) - set(tasks_using_diff_pipe_cluster))
            for task_ in tasks_of_block:
                if task_.get_name() in  parallel_tasks_names:
                    parallel_tasks.append(task_)

        cluster_one = parallel_tasks
        cluster_two = list(set(tasks_of_block) - set(cluster_one))
        return [cluster_one, cluster_two]

    # can any other task on the block run in parallel with that ref task
    def can_any_task_on_block_run_in_parallel(self, ex_dp, sim_dp, ref_task, block):
        clusters_run_in_parallel = self.find_parallel_tasks_of_task_in_block(ex_dp, sim_dp, ref_task, block)
        if len(clusters_run_in_parallel[0]) > 0:
            return True
        else:
            return False

    # ------------------------------
    # Functionality:
    #      load a single memory with a task
    # Variables:
    #      ex_dp: example design to use.
    #      mem: memory to load.
    #      task: task that will occupy memory
    #      dir_: direction (read/write) that the task will use memory
    #      family_task: parent/children task to read/write from to.
    # ------------------------------
    def load_single_mem(self, ex_dp, mem, task, dir_, family_task):
        #get_work_ratio = self.database.get_block_work_ratio_by_task_dir
        if (len(ex_dp.get_blocks_of_task_by_block_type(task, "pe")) ==0):
            print("This should not happen. Something wen't wrong")
            raise NoPEError
        pe = ex_dp.get_blocks_of_task_by_block_type(task, "pe")[0]  # get pe blocks associated with the task
        mem.load_improved(task, family_task)  # load the the memory with the task. Use family ask for work ratio.

    # ------------------------------
    # Functionality:
    #      load a single bus with a task
    # Variables:
    #      ex_dp: example design to use.
    #      mem: memory connected to the bust.
    #      task: task that will occupy bus.
    #      dir_: direction (read/write) that the task will use the bus for
    #      father_task: parent task to read from.
    # ------------------------------
    def load_single_bus(self, ex_dp, mem, task, dir_, family_task):
        #get_work_ratio = self.database.get_block_work_ratio_by_task_dir
        pe_list = ex_dp.get_blocks_of_task_by_block_type(task, "pe")
        if len(pe_list) == 0:
            print("This should not happen. Something went wrong")
            raise NoPEError
        pe = pe_list[0]
        get_work_ratio = self.database.get_block_work_ratio_by_task_dir
        buses = ex_dp.hardware_graph.get_path_between_two_vertecies(pe, mem)[1:-1]
        for bus in buses:
            #bus.load((task, dir_), get_work_ratio(task, pe, dir_), father_task)
            bus.load_improved(task, family_task)

    # ------------------------------
    # Functionality:
    #      load a read buses and memories recursively. Iterate through the hardware graph
    #      and fill out all the memory and buses for read. Note that we populate the read
    #      elements after write. You need to one, and then the other will get filled accordingly.
    #      ex_dp: example design to use.
    #      read_mem: memory to read from.
    #      task: task that will occupy bus.
    #      tasks_seen: a task list to prevent consider a task twice.
    #      father_task: parent task to read from.
    # ------------------------------
    def load_read_mem_and_ic_recursive(self, ex_dp, read_mem, task, tasks_seen, father_task):
        mem_blocks = ex_dp.get_blocks_of_task_by_block_type(task, "mem")  # all the memory blocks of the task
        if "souurce" in task.name:
            write_mem = ex_dp.get_blocks_of_task_by_block_type(task, "mem")
        elif "siink" in task.name:
            write_mem= None
        else:
            write_mem = ex_dp.get_blocks_of_task_by_block_type_and_task_dir(task, "mem", "write")

        # get the tasks that this task writes to and the memory within which the transaction happens
        if write_mem:
            write_mem_tasks = [(mem, ex_dp.get_write_mem_tasks(task,mem)) for mem in write_mem]
        else:
            write_mem_tasks = None

        # Add read Memory, buses
        if read_mem:
            self.load_single_mem(ex_dp, read_mem, task, "read", father_task)
            self.load_single_bus(ex_dp, read_mem, task, "read", father_task)

        if task in tasks_seen: return # terminate
        else: tasks_seen.append(task)


        if not(len(mem_blocks) == 0) and write_mem:
            for mem, child_tasks in write_mem_tasks:
                for child in child_tasks:
                    self.load_single_bus(ex_dp, mem, task, "write", child)  # we make an assumption that the task

        """
        # Add write buses
        if not(len(mem_blocks) == 0) and write_mem:
            if (len(task.get_children()) == 0):
                    print("what")
                    raise TaskNoChildrenError
            for child in task.get_children():
                self.load_single_bus(ex_dp, write_mem, task, "write", child) # we make an assumption that the task
                #self.load_single_bus(ex_dp, write_mem, task, "write", task.get_children()[0]) # we make an assumption that the task
                                                                                          # even if having multiple children, it will be
                                                                                          # writing its results in the same memory
        """
        if write_mem_tasks:
            for mem, child_tasks in write_mem_tasks:
                for child in child_tasks:
                    self.load_read_mem_and_ic_recursive(ex_dp, mem, child, tasks_seen, task)
        """
        # recurse down
        for task_ in task.get_children():
            if len(mem_blocks) == 0:
                mem_blocks_ = ex_dp.get_blocks_of_task_by_block_type(task, "mem")  # all the memory blocks of the task
                print("what")
            self.load_read_mem_and_ic_recursive(ex_dp, write_mem, task_, tasks_seen, task)
        """
    # ------------------------------
    # Functionality:
    #      unload read buses. Need to do this first, to prepare the design for the next iteration.
    # Variables:
    #       ex_dp: example_design
    # ------------------------------
    def unload_read_buses(self, ex_dp):
        busses = ex_dp.get_blocks_by_type("ic")
        _ = [bus.unload_read() for bus in busses]

    # ------------------------------
    # Functionality:
    #      unload all buses. Need to do this first, to prepare the design for the next iteration.
    # Variables:
    #       ex_dp: example_design
    # ------------------------------
    def unload_buses(self, ex_dp):
        busses = ex_dp.get_blocks_by_type("ic")
        _ = [bus.unload_all() for bus in busses]

    # ------------------------------
    # Functionality:
    #      unload read memories. Need to do this first, to prepare the design for the next iteration.
    # Variables:
    #       ex_dp: example_design
    # ------------------------------
    def unload_read_mem(self, ex_dp):
        mems = ex_dp.get_blocks_by_type("mem")
        _ = [mem.unload_read() for mem in mems]

    # ------------------------------
    # Functionality:
    #      unload read memories and buses. Need to do this first, to prepare the design for the next iteration.
    # Variables:
    #       ex_dp: example_design
    # ------------------------------
    def unload_read_mem_and_ic(self, ex_dp):
        self.unload_buses(ex_dp)
        self.unload_read_mem(ex_dp)

    # ------------------------------
    # Functionality:
    #      find out whether a task needs a DMA (read and write memory are different)
    # Variables:
    #       ex_dp: example_design
    #       task: task to consider
    # ------------------------------
    def find_task_s_DMA_needs(self, ex_dp, task):
        mem_blocks = ex_dp.get_blocks_of_task_by_block_type(task, "mem")  # get the  memory blocks of task
        dir_mem_dict = {}
        dir_mem_dict["read"] =[]
        dir_mem_dict["write"] =[]

        # iterate through memory blocks and get their directions (read/write)
        for mem_block in mem_blocks:
            task_dir_list = mem_block.get_task_dir_by_task_name(task)
            for task, dir in task_dir_list:
                dir_mem_dict[dir].append(mem_block)

        # find out whether read/write memories are different
        if len(dir_mem_dict["write"]) > 1:
            raise Exception("a tas can only write to one memory")
        # calc diff to see if write and read mems are diff
        set_diff = set(dir_mem_dict["read"]) - set(dir_mem_dict["write"])
        DMA_src_dest_list = []
        for mem in list(set_diff):
            DMA_src_dest_list.append((mem, dir_mem_dict["write"][0]))
        return DMA_src_dest_list

    # ------------------------------
    # Functionality:
    #     Add a DMA block for the task. TODO: add code.
    # ------------------------------
    def inject_DMA_blocks(self):
        return 0

    # possible implementations of DMA injection
    # at the moment, it's comment out.
    # TODO: uncomment and ensure it's correctness
    """ 
        def inject_DMA_task_for_a_single_task(self, ex_dp:ExDesignPoint, task):
            if config.VIS_GR_PER_GEN: vis_hardware(ex_dp)
            task_s_DMA_s_src_dest_list = self.find_task_s_DMA_needs(ex_dp, task)
            task_s_pe = ex_dp.get_blocks_of_task_by_block_type("pe")
            bus_task_unloaded_list  = []  #to keep track of what you already unloaded to avoid unloading again
            for task_s_mem_src, task_s_mem_dest in task_s_DMA_s_src_dest_list:
                if self.pe_hanging_off_of_src_block(task_s_pe, task_s_mem_src):
                    ic_neigh_of_src = [neigh for neigh in task_s_mem_src.neighs if neigh.type == "ic"][0]
                    DMA_block = self.database.sample_DMA_blocks()
                    DMA_block = self.database.copy_SOC(DMA_block, task_s_mem_src)
                    DMA_block.connect(ic_neigh_of_src)
                    self.copy_task(task)


                    src.load((task,"write"))
                    src.unload((task, "read"))
                    ic_neigh_of_src.unload((task, "read"))

                ic_neigh_of_src = [neigh for neigh in src.neighs if neigh.type == "ic"][0]
                ic_neigh_of_des = [neigh for neigh in dest.neighs if neigh.type == "ic"][0]
                if



                reads_work_ratio = src.get_task_s_work_ratio_by_task_and_dir(task, "read")
                # unload task from the src memory and the immediate bus connected to it

                buses = ex_dp.hardware_graph.get_path_between_two_vertecies(src, dest)[1:-1]
                for bus in buses:
                    if (bus, task) not in bus_task_unloaded_list:
                        bus.unload((task, "write"))
                        bus_task_unloaded_list.append((bus,task))

                DMA_block = self.database.sample_DMA_blocks()
                DMA_block = self.database.copy_SOC(DMA_block, src)
                DMA_block.connect(ic_neigh_of_src)

                for task_to_read_from, work_ratio_value in reads_work_ratio.items():
                    task_s_bytes_to_transfer = task.work * work_ratio_value
                    DMA_task = Task("DMA_from_" + task_to_read_from + "_to_" + task.name , task_s_bytes_to_transfer)
                    # load DMA task to the appropriate mem and buses
                    DMA_block.load((DMA_task,"loop_back"), {DMA_task.name:1})
                    src.load((DMA_task, "read"), {task_to_read_from:1})
                    ic_neigh_of_src.load((DMA_task, "read"), {task_to_read_from:1})
                    for bus in buses: bus.load((DMA_task, "write"), {task.name:1})
                    dest.load((DMA_task, "write"), {task.name: 1})
                    # load destination and it's immediate bus with the task
                    dest.load((task, "read"), {DMA_task.name: work_ratio_value})
                    ic_neigh_of_des.load((task, "read"), {DMA_task.name: work_ratio_value})

                    DMA_task.add_child(task)
                    parent = ex_dp.get_task_by_name(task_to_read_from)
                    parent.remove_child(task)
                    parent.add_child(DMA_task)
                    self.DMA_task_ctr += 1
                    ex_dp.hardware_graph.update_graph(block_to_prime_with=ex_dp.get_blocks()[0])
                    if config.VIS_GR_PER_GEN: vis_hardware(ex_dp)
    #            self.unload_read_mem_and_ic(ex_dp)
    #            self.load_tasks_to_read_mem_and_ic(ex_dp)

                if config.VIS_GR_PER_GEN: vis_hardware(ex_dp)
        """
    """
    def inject_DMA_tasks_for_all_tasks(self, ex_dp:ExDesignPoint):
        tasks = [task for task in ex_dp.get_tasks() if not self.task_dummy(task)]
        for task in tasks:
            self.inject_DMA_task_for_a_single_task(ex_dp, task)
        task_s_DMA_s_src_dest_dict = {}
        for task in tasks:
            task_s_DMA_s_src_dest_dict[task] = self.find_task_s_DMA_if_necessary(ex_dp, task)
        self.inject_DMA_task(task_s_DMA_s_src_dest_dict)
        self.inject_DMA_blocks(task_s_DMA_s_src_dest_dict)
    """

    # ------------------------------
    # Functionality:
    #     find all whether pe is connected to another block
    # Varaibles:
    #       pe: processing element to query
    #       src: the src block to see if pe is connected to
    # ------------------------------
    def pe_hanging_of_of_src_block(self, pe, src):
        return pe in src.neighs

    # ------------------------------
    # Functionality:
    #    add DMA for all the task that need it (their read/write memory is not the same)
    # Varaibles:
    #       ex_dp: example design
    # ------------------------------
    def inject_DMA_tasks_for_all_tasks(self, ex_dp: ExDesignPoint):
        tasks = [task for task in ex_dp.get_tasks() if not task.is_task_dummy()]
        for task in tasks:
            self.inject_DMA_task_for_a_single_task(ex_dp, task)
        self.inject_DMA_blocks()

    # generate and populate the queues that connect different hardware block, e.g., bus and memory, pe and bus, ...
    def pipe_design(self, ex_dp, workload_to_blocks_map):
        ex_dp.get_hardware_graph().pipe_design()

    # -------------------------------------------
    # Functionality:
    #       converts the example design point (ex_dp) to a simulatable design point (sim_dp)
    #       ex_dp and sim_dp are kept separate since sim_dp contains information about the simulation such
    #       as latency/power/area ...
    # Variables:
    #       ex_dp: example design
    # ------------------------------------------
    def convert_to_sim_des_point(self, ex_dp):
        # generate simulation semantics such as task to block map (workload_to_blocks_map(),
        # and workload to block schedule (tasks schedule)

        # generate task to hardware mapping
        workload_to_blocks_map = self.gen_workload_to_blocks_from_blocks(ex_dp.get_blocks())
        self.pipe_design(ex_dp, workload_to_blocks_map)  # populate the queues

        # generate task schedules
        workload_to_pe_block_schedule = WorkloadToPEBlockSchedule()
        for task in ex_dp.get_tasks():
            workload_to_pe_block_schedule.task_to_pe_block_schedule_list_sorted.append(TaskToPEBlockSchedule(task, 0))

        # convert to teh sim design
        return SimDesignPoint(ex_dp.get_hardware_graph(), workload_to_blocks_map, workload_to_pe_block_schedule)

    # -------------------------------------------
    # Functionality:
    #       generate task to block mapping (use by the simulator)
    # Variables:
    #       blocks: hardware blocks to consider for mapping
    # -------------------------------------------
    def gen_workload_to_blocks_from_blocks(self, blocks):
        workload_to_blocks_map = WorkloadToHardwareMap()
        # make task_to_block out of blocks
        for block in blocks:
            for task_dir, work_ratio in block.get_tasks_dir_work_ratio().items():   # get the task to task work ratio (gables)
                task, dir = task_dir
                task_to_blocks = workload_to_blocks_map.get_by_task(task)
                if task_to_blocks:
                    task_to_blocks.block_dir_workRatio_dict[(block, dir)] = work_ratio
                else:
                    task_to_blocks = TaskToBlocksMap(task, {(block, dir): work_ratio})
                    workload_to_blocks_map.tasks_to_blocks_map_list.append(task_to_blocks)
        return workload_to_blocks_map

    # generate light systems (i.e., systems that are not FARSI compatible, but have all the characterizations necessary
    #                        to generate FARSI systems from). This steps allow us to quickly generate all the systems
    #                        without worrying about generating FARSI systems
    def light_system_gen_exhaustively(self, system_workers, database):
        all_systems = exhaustive_system_generation(system_workers, database.db_input.gen_config)  # not farsi compatible
        return all_systems

    # generate FARSI systems from the light systems inputted. This is used for exhaustive search
    # comparison
    def FARSI_system_gen_exhaustively(self, light_systems, system_workers):
        mapping_process_cnt = system_workers[0]
        mapping_process_id = system_workers[1]
        FARSI_gen_process_cnt = system_workers[2]
        FARSI_gen_process_id = system_workers[3]

        FARSI_systems = []  # generating FARSI compatible systems from all systems

        # upper_bound and lower_bound used for allowing for parallel execution
        #light_systems_lower_bound = 0
        #light_systems_upper_bound = len(light_systems)
        light_systems_lower_bound = int(FARSI_gen_process_id*(len(light_systems) /FARSI_gen_process_cnt))
        light_systems_upper_bound = int(min((FARSI_gen_process_id + 1) * (len(light_systems) / FARSI_gen_process_cnt), len(light_systems)))

        num_of_sys_to_gen = light_systems_upper_bound - light_systems_lower_bound

        # adding and loading the pes
        for idx, system in enumerate(light_systems[light_systems_lower_bound:light_systems_upper_bound]):

            if (idx % max(int(num_of_sys_to_gen/10.0),1) == 0): # debugging
                print("---------" + str(idx/num_of_sys_to_gen) + "% of systems generated for process " +
                      str(mapping_process_id) + "_" + str(FARSI_gen_process_id))

            buses_name = system.get_BUS_list()
            pe_primer = None
            pe_idx_global = 0
            mem_idx_global = 0
            for bus_idx, bus_name in enumerate(buses_name):
                prev_bus_block = None
                if not (bus_idx == 0):
                    prev_bus_block = bus_block
                bus_block = self.database.sample_similar_block(self.database.get_block_by_name(bus_name))
                if not (prev_bus_block is None):
                    bus_block.connect(prev_bus_block)

                pe_neigh_names = system.get_bus_s_pe_neighbours(bus_idx)
                for pe_idx, pe_name in enumerate(pe_neigh_names):
                    pe_block = self.database.sample_similar_block(self.database.get_block_by_name(pe_name))
                    pe_primer = pe_block # can be any pe block. used to generate the hardware graph
                    pe_block.connect(bus_block)
                    tasks = [self.database.get_task_by_name(task_) for task_ in system.get_pe_task_set()[pe_idx_global]]
                    for task in tasks:
                        pe_block.load_improved(task, task)
                    pe_idx_global += 1

                """
                # adding and loading the memories
                for bus_idx, bus_name in enumerate(buses_name):
                    prev_bus_block = None
                    if not (bus_idx == 0):
                        prev_bus_block = bus_block
                    bus_block = self.database.sample_similar_block(self.database.get_block_by_name(bus_name))
                    if not (prev_bus_block is None):
                        bus_block.connect(prev_bus_block)
                    """
                mem_neigh_names = system.get_bus_s_mem_neighbours(bus_idx)
                for mem_idx, mem_name in enumerate(mem_neigh_names):
                    mem_block = self.database.sample_similar_block(self.database.get_block_by_name(mem_name)) # must sample similar block otherwise, we always get the same exact block
                    mem_block.connect(bus_block)
                    #if mem_idx_global >= len(system.get_mem_task_set()):
                    #    print("what")
                    tasks = [self.database.get_task_by_name(task_) for task_ in system.get_mem_task_set()[mem_idx_global]]
                    for task in tasks:
                        for child in task.get_children():
                            mem_block.load_improved(task, child)
                    mem_idx_global += 1

            # generate a hardware graph and load read mem and ic
            hardware_graph = HardwareGraph(pe_primer)
            ex_dp = ExDesignPoint(hardware_graph)
            self.load_tasks_to_read_mem_and_ic(ex_dp)
            ex_dp.sanity_check()
            FARSI_systems.append(ex_dp)

        #for farsi_system in FARSI_systems:

        print("----- all system generated for process: " + str(mapping_process_id) + "_" + str(FARSI_gen_process_id))
        return FARSI_systems

    # ------------------------------
    # Functionality:
    #       generate initial design from the specified parsed design
    # ------------------------------
    def gen_specific_parsed_ex_dp(self, database):
        # the hw_g parsed from the csv has double connections, i.e,. mem to pe and pe to me. However, we only
        # need to specify  it one way (since connecting pe to mem, means automatically connecting mem to pe as well).
        # This function get rids of these duplications. This is not optional.
        def filter_duplicates(hw_g):
            seen_connections = []  # keep track of connections already seen to avoid double counting
            for blk, neighs in hw_g.items():
                for  neigh in neighs:
                    if not (blk, neigh) in seen_connections:
                        seen_connections.append((neigh, blk))

            for el in seen_connections:
                key = el[0]
                value = el[1]
                del hw_g[key][value]

        # get the (light, i.e, in sting format) hardware_graph and task_to_hw_mappings  from csv
        hw_g = database.db_input.get_parsed_hardware_graph()
        # get rid of mutual connection, as we already connect both ways with "connect" APIs
        filter_duplicates(hw_g)
        task_to_hw_mapping = database.db_input.get_parsed_task_to_hw_mapping()

        # use the hw_g to generate the topology (connect blocks together)
        block_seen = {}  # dictionary with key:blk_name, value of block object
        for blk_name, children_names in hw_g.items():
            if blk_name not in block_seen.keys():  # generate and memoize
                blk = self.database.gen_one_block_by_name(blk_name)
                block_seen[blk_name] = blk
            else:
                blk = block_seen[blk_name]
            for child_name in children_names:
                if child_name not in block_seen.keys():  # generate and memoize
                    child = self.database.gen_one_block_by_name(child_name)
                    block_seen[child_name] = child
                else:
                    child = block_seen[child_name]
                blk.connect(child)
            # get a block to prime with
            if blk.type == "pe":
                pe_primer = blk

        # load the blocks with tasks
        for blk_name, task_names in task_to_hw_mapping.items():
            blk = block_seen[blk_name]
            for task in task_names:
                task_parent_name = task[0]
                task_child_name = task[1]
                task_parent = self.database.get_task_by_name(task_parent_name)
                task_child = self.database.get_task_by_name(task_child_name)
                blk.load_improved(task_parent, task_child)

        # generate a hardware graph and load read mem and ic
        hardware_graph = HardwareGraph(pe_primer, "user_generated")  # user_generated will deactivate certain checks
                                                                     # noBUSerror that the tool can/must not generate
                                                                     # but are ok if inputed by user
        ex_dp = ExDesignPoint(hardware_graph)
        self.load_tasks_to_read_mem_and_ic(ex_dp)
        ex_dp.sanity_check()
        ex_dp.hardware_graph.update_graph_without_prunning()
        ex_dp.hardware_graph.pipe_design()
        return ex_dp

    # ------------------------------
    # Functionality:
    #       generate initial design from the hardcoded specified design
    # ------------------------------
    def gen_specific_hardcoded_ex_dp(self, database):
        lib_relative_addr = config.database_data_dir.replace(config.home_dir, "")
        lib_relative_addr_pythony_fied = lib_relative_addr.replace("/", ".")
        # only supporting SLAM at the moment -> Iulian: starting to support any hardcoded design
        workload_name = list(database.db_input.sw_hw_database_population["workloads"])[0] #supporting only one hardcoded workload
        files_to_import = [lib_relative_addr_pythony_fied + ".hardcoded." + workload + ".input" for workload in [workload_name]]
        imported_databases = [importlib.import_module(el) for el in files_to_import][0]
        ex_dp = imported_databases.gen_hardcoded_design(database)
        self.load_tasks_to_read_mem_and_ic(ex_dp)
        return ex_dp

    def get_most_inferior_block(self, block, tasks):
        return self.database.sample_most_inferior_blocks_by_type(block_type=block.type, tasks=self.__tasks)


    def get_most_inferior_block_before_unrolling(self, block, tasks):
        return self.database.sample_most_inferior_blocks_before_unrolling_by_type(block_type=block.type, tasks=block.get_tasks_of_block(), block=block)


    def gen_specific_design_with_a_star_noc(self, database):

        num_of_hops = database.db_input.sw_hw_database_population["misc_knobs"]["num_of_hops"]  # supporting only one hardcoded workload
        #if database.db_input.parallel_task_names == {}:
        if len(list(database.db_input.parallel_task_names.values())) == 0:
            max_parallelism = 0
        else:
            max_parallelism = max([len(el) for el in database.db_input.parallel_task_names.values()])
        pes = []
        mems = []
        ics = []
        parallel_task_names =database.db_input.parallel_task_names
        for el in range (0, max(max_parallelism,1)):
            pe = self.database.sample_most_inferior_blocks_by_type(block_type="pe", tasks=self.__tasks)
            mem = self.database.sample_most_inferior_blocks_by_type(block_type="mem", tasks=self.__tasks)
            if  el == 0:
                ic = self.database.sample_most_inferior_blocks_by_type(block_type="ic", tasks=self.__tasks)
            if  el == 0:
                ic = self.database.sample_most_inferior_SOC(ic, "power")
                ics.append(ic)

            pe = self.database.sample_most_inferior_SOC(pe, config.sorting_SOC_metric)
            mem = self.database.sample_most_inferior_SOC(mem, config.sorting_SOC_metric)
            pes.append(pe)
            mems.append(mem)

        for pe,mem in zip(pes, mems):
            pe.connect(ics[0])
            ics[0].connect(mem)

        serial_task_names = set([el.name for el in self.__tasks])
        for parallel_task_names in database.db_input.parallel_task_names.values():
            parallel_tasks = [database.get_task_by_name(tsk_name) for tsk_name in parallel_task_names]
            for idx, task in enumerate(parallel_tasks):
                serial_task_names = serial_task_names - set([task.name])
                pes[idx].load_improved(task, task)
                for task_child in task.get_children():
                    mems[idx].load_improved(task, task_child)  # load memory with tasks

        serial_tasks = [database.get_task_by_name(tsk_name) for tsk_name in serial_task_names]
        for task in serial_tasks:
            pes[0].load_improved(task, task)
            for task_child in task.get_children():
                mems[0].load_improved(task, task_child)  # load memory with tasks

        # generate a hardware graph and load read mem and ic
        """
        for pe in pes:
            for tsk in pe.get_tasks_of_block():
                if "souurce" in tsk.name:
                    pe_ = pe
        """

        hardware_graph = HardwareGraph(pes[0])
        ex_dp = ExDesignPoint(hardware_graph)
        self.load_tasks_to_read_mem_and_ic(ex_dp)
        ex_dp.hardware_graph.update_graph()
        ex_dp.hardware_graph.pipe_design()
        return ex_dp


    def gen_specific_design_with_hops_and_stars(self, database):
        num_of_hops = database.db_input.sw_hw_database_population["misc_knobs"]["num_of_hops"]  # supporting only one hardcoded workload
        num_of_NoCs = database.db_input.sw_hw_database_population["misc_knobs"]["num_of_NoCs"]  # supporting only one hardcoded workload
        if num_of_hops > num_of_NoCs:
            print("number of hops can't be greater than number of NoCs")
            exit(0)

        #if database.db_input.parallel_task_names == {}:
        if len(list(database.db_input.parallel_task_names.values())) == 0:
            max_parallelism = 0
        else:
            max_parallelism = max([len(el) for el in database.db_input.parallel_task_names.values()])
        pes = []
        mems = []
        ics = []
        parallel_task_names =database.db_input.parallel_task_names
        for el in range (0, max(max_parallelism,1)):
            pe = self.database.sample_most_inferior_blocks_by_type(block_type="pe", tasks=self.__tasks)
            mem = self.database.sample_most_inferior_blocks_by_type(block_type="mem", tasks=self.__tasks)
            if  el == 0:
                ic = self.database.sample_most_inferior_blocks_by_type(block_type="ic", tasks=self.__tasks)
            if  el == 0:
                ic = self.database.sample_most_inferior_SOC(ic, "power")
                ics.append(ic)

            pe = self.database.sample_most_inferior_SOC(pe, config.sorting_SOC_metric)
            mem = self.database.sample_most_inferior_SOC(mem, config.sorting_SOC_metric)
            pes.append(pe)
            mems.append(mem)

        for pe,mem in zip(pes, mems):
            pe.connect(ics[0])
            ics[0].connect(mem)

        serial_task_names = set([el.name for el in self.__tasks])
        for parallel_task_names in database.db_input.parallel_task_names.values():
            parallel_tasks = [database.get_task_by_name(tsk_name) for tsk_name in parallel_task_names]
            for idx, task in enumerate(parallel_tasks):
                serial_task_names = serial_task_names - set([task.name])
                pes[idx].load_improved(task, task)
                for task_child in task.get_children():
                    mems[idx].load_improved(task, task_child)  # load memory with tasks

        listified_serial_task_names = list(serial_task_names)
        listified_serial_task_names.sort()
        serial_task_names = listified_serial_task_names

        added_tasks = max(0, num_of_hops-2)
        serial_tasks = [database.get_task_by_name(tsk_name) for tsk_name in serial_task_names]
        # dummy tasks in the middle
        hoppy_task_names =database.db_input.hoppy_task_names

        last_non_hoppy_task = list(set(serial_task_names) - set(hoppy_task_names) - set(["synthetic_siink"]) - set(["synthetic_souurce"]))
        last_non_hoppy_task.sort()
        last_non_hoppy_task = last_non_hoppy_task[-1]

        for el in hoppy_task_names:
            pe = self.database.sample_most_inferior_blocks_by_type(block_type="pe", tasks=self.__tasks)
            mem = self.database.sample_most_inferior_blocks_by_type(block_type="mem", tasks=self.__tasks)
            pe = self.database.sample_most_inferior_SOC(pe, config.sorting_SOC_metric)
            mem = self.database.sample_most_inferior_SOC(mem, config.sorting_SOC_metric)
            pes.append(pe)
            mems.append(mem)
            ic = self.database.sample_most_inferior_blocks_by_type(block_type="ic", tasks=self.__tasks)
            ic = self.database.sample_most_inferior_SOC(ic, "power")
            ics.append(ic)
            pe.connect(ic)
            ic.connect(mem)

        hoppy_tasks = [database.get_task_by_name(tsk_name) for tsk_name in hoppy_task_names]
        for idx, task in enumerate(hoppy_tasks):
            pes[-1*(1+idx)].load_improved(task, task)
            for task_child in task.get_children():
                mems[-1*(1+idx)].load_improved(task, task_child)  # load memory with tasks


        # serial tasks
        if num_of_hops > 1:
            pe = self.database.sample_most_inferior_blocks_by_type(block_type="pe", tasks=self.__tasks)
            mem = self.database.sample_most_inferior_blocks_by_type(block_type="mem", tasks=self.__tasks)
            pe = self.database.sample_most_inferior_SOC(pe, config.sorting_SOC_metric)
            mem = self.database.sample_most_inferior_SOC(mem, config.sorting_SOC_metric)
            pes.append(pe)
            mems.append(mem)
            ic = self.database.sample_most_inferior_blocks_by_type(block_type="ic", tasks=self.__tasks)
            ic = self.database.sample_most_inferior_SOC(ic, "power")
            ics.append(ic)
            pe.connect(ic)
            ic.connect(mem)

        for idx, task in enumerate(serial_tasks):
            if task.name in hoppy_task_names:
                continue
            if "souurce" in task.name:
                idx_ =0
            elif "synthetic_0" in task.name:
                idx_ = -1
            else:
               idx_ = -1*(idx%2 )
            pes[idx_].load_improved(task, task)
            for task_child in task.get_children():
                mems[idx_].load_improved(task, task_child)  # load memory with tasks

        for idx, ic in enumerate(ics):
            if idx == 0:
                continue
            ic.connect(ics[idx - 1])

        if num_of_hops < num_of_NoCs:
            if num_of_hops == 2:
                ics[-1].connect(ics[0])
            if num_of_hops == 3:
                ics[-2].connect(ics[0])
                #ics[1].connect(ics[-1])


        hardware_graph = HardwareGraph(pes[0])
        ex_dp = ExDesignPoint(hardware_graph)
        self.load_tasks_to_read_mem_and_ic(ex_dp)
        ex_dp.hardware_graph.update_graph()
        ex_dp.hardware_graph.pipe_design()
        return ex_dp


    def gen_specific_design_with_hops(self, database):
        num_of_hops = database.db_input.sw_hw_database_population["misc_knobs"]["num_of_hops"]  # supporting only one hardcoded workload
        pes = []
        mems = []
        ics = []

        for el in range (0, num_of_hops):
            pe = self.database.sample_most_inferior_blocks_by_type(block_type="pe", tasks=self.__tasks)
            mem = self.database.sample_most_inferior_blocks_by_type(block_type="mem", tasks=self.__tasks)
            ic = self.database.sample_most_inferior_blocks_by_type(block_type="ic", tasks=self.__tasks)
            pe = self.database.sample_most_inferior_SOC(pe, config.sorting_SOC_metric)
            mem = self.database.sample_most_inferior_SOC(mem, config.sorting_SOC_metric)
            ic = self.database.sample_most_inferior_SOC(ic, "power")
            pes.append(pe)
            mems.append(mem)
            ics.append(ic)

        for pe,mem, ic in zip(pes, mems, ics):
            pe.connect(ic)
            ic.connect(mem)

        for idx, ic in enumerate(ics):
            if idx == 0:
                continue
            ic.connect(ics[idx-1])


        self.load_tasks_to_pe_and_write_mem_for_hops(pes, mems, self.__tasks, num_of_hops)

        # generate a hardware graph and load read mem and ic
        hardware_graph = HardwareGraph(pe)
        ex_dp = ExDesignPoint(hardware_graph)
        self.load_tasks_to_read_mem_and_ic(ex_dp)
        ex_dp.hardware_graph.update_graph()
        ex_dp.hardware_graph.pipe_design()
        return ex_dp



        #print("ok")


    # ------------------------------
    # Functionality:
    #       generate initial design. Used to boot strap the exploration
    # ------------------------------
    def gen_init_des(self):
        pe = self.database.sample_most_inferior_blocks_by_type(block_type="pe", tasks=self.__tasks)
        mem = self.database.sample_most_inferior_blocks_by_type(block_type="mem", tasks=self.__tasks)
        ic = self.database.sample_most_inferior_blocks_by_type(block_type="ic", tasks=self.__tasks)
        pe = self.database.sample_most_inferior_SOC(pe, config.sorting_SOC_metric)
        mem = self.database.sample_most_inferior_SOC(mem, config.sorting_SOC_metric)
        ic = self.database.sample_most_inferior_SOC(ic, "power")

        # connect blocks together
        pe.connect(ic)
        ic.connect(mem)

        self.load_tasks_to_pe_and_write_mem(pe, mem, self.__tasks)

        # generate a hardware graph and load read mem and ic
        hardware_graph = HardwareGraph(pe)
        ex_dp = ExDesignPoint(hardware_graph)
        self.load_tasks_to_read_mem_and_ic(ex_dp)
        ex_dp.hardware_graph.update_graph()
        ex_dp.hardware_graph.pipe_design()
        return ex_dp

    # ------------------------------
    # Functionality:
    #      find the hot blocks' block bottleneck. This means we find the bottleneck associated
    #       with the hottest (longest latency) block.
    # Variables:
    #       ex_dp: example design
    #       hot_kernel_block_bottlneck: block bottlneck
    # ------------------------------
    def find_cores_hot_kernel_blck_bottlneck(self, ex_dp:ExDesignPoint, hot_kernel_blck_bottleneck:Block):
        # iterate through the blocks and compare agains the name
        for block in ex_dp.get_blocks():
            if block.instance_name == hot_kernel_blck_bottleneck.instance_name:
                return block
        raise Exception("did not find a corresponding block in the ex_dp with the name:" +
                        str(hot_kernel_blck_bottleneck.instance_name))

    # ------------------------------
    # Functionality:
    #     for debugging purposes. Making sure that each ic (noc) has at least one connected block.
    # Variables:
    #       ex_dp: example design
    # ------------------------------
    def sanity_check_ic(self, ex_dp:ExDesignPoint):
        ic_blocks = ex_dp.get_blocks_by_type("ic")
        for block in ic_blocks:
            pe_neighs = [block for block in block.neighs if block.type == "pe"]
            if len(pe_neighs) == 0 :
                ex_dp.hardware_graph.update_graph(block_to_prime_with=block)
                if config.VIS_GR_PER_GEN: vis_hardware(ex_dp)
                vis_hardware(ex_dp)
                raise Exception("this design is not valid")

    # Get the immediate supperior/inferior block according to the metric (and direction).
    # This version is the fast version (by the use of caching), however, it's unintuive
    # Variables:
    #   metric_dir: -1 (increase) and 1 (decrease)
    #   block: the block to improve/de-improve
    def get_immediate_block_fast(self, block, metric, metric_dir, tasks):
        imm_blck_non_unique = self.database.up_sample_down_sample_block_fast(block, metric, metric_dir, tasks)[0]  # get the first value
        blkL = self.database.cached_block_to_blockL[imm_blck_non_unique]
        imm_blck = self.database.cast(blkL)
        return self.database.copy_SOC(imm_blck, block)

    def get_immediate_block_multi_metric_fast(self, block, metric, sorted_metric_dir, tasks):
        imm_blck_non_unique = self.database.up_sample_down_sample_block_multi_metric_fast(block, sorted_metric_dir, tasks)[0]  # get the first value
        blkL = self.database.cached_block_to_blockL[imm_blck_non_unique]
        imm_blck = self.database.cast(blkL)
        return self.database.copy_SOC(imm_blck, block)



    # Get the immediate supperior/inferior block according to the metric (and direction)
    # Variables:
    #   metric_dir: -1 (increase) and 1 (decrease)
    #   block: the block to improve/de-improve
    def get_immediate_block_slow(self, block, metric, metric_dir, tasks):
        imm_blck = self.database.up_sample_down_sample_block(block, metric, metric_dir, tasks)[0]  # get the first value
        return self.database.copy_SOC(imm_blck, block)

    # Get the immediate superior/inferior block according to the metric (and direction) that already
    # exist in the design
    # Variables:
    #   metric_dir: -1 (increase) and 1 (decrease)
    #   block: the block to improve/de-improve
    def get_equal_immediate_block_present_slow(self, ex_dp, block, metric, metric_dir, tasks):
        imm_blcks = self.database.equal_sample_up_sample_down_sample_block(block, metric, metric_dir, tasks)  # get the first value
        des_blocks = ex_dp.get_blocks()
        for block_ in imm_blcks:
            for des_block in des_blocks:
                if block_.get_generic_instance_name() == des_block.get_generic_instance_name() and \
                        not (block.instance_name  == des_block.instance_name):
                    return des_block

        return block

    def get_all_compatible_blocks_of_certain_char(self, ex_dp, block, metric, metric_dir, tasks, mode):
        metric_dir = -1
        imm_blcks_non_unique_1 = [el for el in self.database.equal_sample_up_sample_down_sample_block_fast(block, metric, metric_dir,
                                                                                           tasks) if not isinstance(el,str)]  # get the first value


        metric_dir = 1
        imm_blcks_non_unique_2 = [el for el in self.database.equal_sample_up_sample_down_sample_block_fast(block, metric, metric_dir,
                                                                                           tasks) if not isinstance(el,str)]  # get the first value

        results = []
        if mode =="frequency_modulation":
            for blk in imm_blcks_non_unique_1 + imm_blcks_non_unique_2:
                if not blk.get_block_freq() == block.get_block_freq() and (blk.subtype == block.subtype) and (blk.get_block_bus_width() == block.get_block_bus_width()) and blk.get_loop_itr_cnt() == block.get_loop_itr_cnt():
                    results.append(blk)
        elif mode =="allocation":
            for blk in imm_blcks_non_unique_1 + imm_blcks_non_unique_2:
                if not blk.subtype == block.subtype and blk.get_block_freq() == block.get_block_freq() and (blk.get_block_bus_width() == block.get_block_bus_width()) and blk.get_loop_itr_cnt() == block.get_loop_itr_cnt():
                    results.append(blk)
        elif mode == "bus_width_modulation":
            for blk in imm_blcks_non_unique_1 + imm_blcks_non_unique_2:
                if not (blk.get_block_bus_width() == block.get_block_bus_width() and blk.subtype == block.subtype and blk.get_block_freq() == block.get_block_freq()  and blk.get_loop_itr_cnt() == block.get_loop_itr_cnt()):
                    results.append(blk)
        elif mode == "loop_iteration_modulation":
            for blk in imm_blcks_non_unique_1 + imm_blcks_non_unique_2:
                if not blk.get_loop_itr_cnt() == block.get_loop_itr_cnt() and blk.subtype == block.subtype and blk.get_block_freq() == block.get_block_freq() and (blk.get_block_bus_width() == block.get_block_bus_width()):
                    results.append(blk)
        else:
            print("this mode:" + mode + "not supported")
            exit(0)

        return results


    def get_equal_immediate_blocks_present_fast(self, ex_dp, block, metric, metric_dir, tasks):
        imm_blcks_non_unique = self.database.equal_sample_up_sample_down_sample_block_fast(block, metric, metric_dir,
                                                                           tasks)  # get the first value

        imm_blcks_names = [blck.get_generic_instance_name() for blck in imm_blcks_non_unique]
        #all_compatible_blocks = [blck.get_generic_instance_name() for blck in self.database.find_all_compatible_blocks_fast(block.type, tasks)]

        blocks_present = ex_dp.get_blocks()
        result_blocks = []
        for block_present in blocks_present:
            if not (block.instance_name == block_present.instance_name) and block_present.get_generic_instance_name() in imm_blcks_names:
               result_blocks.append(block_present)

        if len(result_blocks) == 0:
            result_blocks = [block]
        return result_blocks

    def get_equal_immediate_block_present_multi_metric_fast(self, ex_dp, block, metric, sorted_metric_dir, tasks):
        imm_blcks_non_unique = self.database.equal_sample_up_sample_down_sample_block_multi_metric_fast(block, metric, sorted_metric_dir,
                                                                           tasks)  # get the first value
        des_blocks = ex_dp.get_blocks()
        for block_ in imm_blcks_non_unique:
            for des_block in des_blocks:
                if block_.get_generic_instance_name() == des_block.get_generic_instance_name() and \
                        not (block.instance_name == des_block.instance_name):
                    return des_block

        return block


    # ------------------------------
    # Functionality:
    #  transforming the current design (by either applying a swap, for block improvement or a split, for reducing block contention)
    # Variables:
    #       move_name: type of the move to apply (currently only supporting swap or split)
    #       sup_block: (swaper block) block to swap with
    #       hot_bloc: block to be swapped
    #       des_tup: (design tuple) containing ex_dp, sim_dp
    #       mode: not used any more. TODO: get rid of this
    #       hot_kernel_pos: position of the hot kenrel. Used for finding the hot kernel.
    # ------------------------------
    def move_to(self,move_name , sup_block, hot_block, des_tup, mode, hot_kernel_pos):
            if move_name == "swap":
                if not hot_block.type == "ic":
                    self.unload_buses(des_tup[0])  # unload buses
                else:
                    self.unload_read_buses(des_tup[0]) # unload buses
                self.swap_block(hot_block, sup_block)  # swap
                self.mig_cur_tasks_of_src_to_dest(hot_block, sup_block)  # migrate tasks over
                des_tup[0].hardware_graph.update_graph(block_to_prime_with=sup_block)  # update the hardware graph
                self.unload_buses(des_tup[0]) # unload buses
                self.unload_read_mem(des_tup[0]) # unload memories
                if config.VIS_GR_PER_GEN: vis_hardware(des_tup[0])
            elif move_name == "split":
                self.unload_buses(des_tup[0]) # unload buss
                self.reduce_contention(des_tup, mode, hot_kernel_pos) # reduce contention by allocating an extra block
            else:
                raise Exception("move:" + move_name + " is not supported")


    """
    # we assume that DRAM can be hanging only from one router.
    # This check ensures that we only have dram present around one router
    def improve_locality(self, ex_dp, move_to_apply):
        # find all the drams and their ics
        src_block = move_to_apply.get_block_ref()  # memory to move
        src_block_ic = [el for el in src_block.get_neights() if el.subtype == "ic"]
        dest_block = move_to_apply.get_des_block()  # destination Ic
        src_block.disconnect(src_block_ic)
        dest_block.connect(src_block)
    """

    # we assume that DRAM can be hanging only from one router.
    # This check ensures that we only have dram present around one router
    def fix_dram(self, ex_dp):
        # find all the drams and their ics
        all_drams = [el for el in ex_dp.get_hardware_graph().get_blocks() if el.subtype == "dram"]
        dram_ics = []
        for dram in all_drams:
            for neigh in dram.get_neighs():
                if neigh.type == "ic":
                    dram_ics.append(neigh)

        for ic in dram_ics:
            neighs_has_no_pe = len([neigh for neigh in ic.get_neighs() if neigh.type == "pe"]) == 0
            if neighs_has_no_pe:
                system_ic = ic
                break

        # find or allocatd (if none exist) the system IC (which DRAM hangs from
        system_ic = None
        for ic in dram_ics:
            ic_has_no_pe = len([neigh for neigh in ic.get_neighs() if neigh.type =="pe"]) == 0
            if ic_has_no_pe:
                system_ic = ic
                break
        if system_ic == None:
            dram_ics_sorted = sorted(dram_ics, key=attrgetter("peak_work_rate"), reverse=True)  #
            system_ic_alike = dram_ics_sorted[0]
            system_ic = self.allocate_similar_block(system_ic_alike, [])

        # iterate through drams and connect them to he system ic
        for dram_ic in dram_ics:
            if dram_ic not in system_ic.get_neighs():
                dram_ic.connect(system_ic)
            hanging_drams = [el for el in dram_ic.get_neighs() if el.subtype == "dram"]
            for hanging_dram in hanging_drams:
                hanging_dram.disconnect(dram_ic)
                hanging_dram.connect(system_ic)





    # ------------------------------
    # Functionality:
    #   By applying the move, the initial design is transformed to a new design
    # Variables:
    #   des_tup is the design tup, concretely (design, simulated design)
    # ------------------------------
    def apply_move(self, des_tup, move_to_apply):
        ex_dp, sim_dp = des_tup
        blck_ref = move_to_apply.get_block_ref()
        #print("applying move  " +  move.name + " -----" )
        #pre_moved_ex = copy.deepcopy(ex_dp)  # this is just for move sanity checking
        gc.disable()
        pre_moved_ex = cPickle.loads(cPickle.dumps(ex_dp, -1))
        gc.enable()

        if move_to_apply.get_transformation_name() == "identity":
            return ex_dp, True
        if move_to_apply.get_transformation_name() == "swap":
            if not blck_ref.type == "ic": self.unload_buses(ex_dp)  # unload buses
            else: self.unload_read_buses(ex_dp)  # unload buses
            succeeded = self.swap_block(blck_ref, move_to_apply.get_des_block())
            #succeeded = self.mig_cur_tasks_of_src_to_dest(move_to_apply.get_block_ref(), move_to_apply.get_des_block())  # migrate tasks over
            succeeded = self.mig_tasks_of_src_to_dest(ex_dp, blck_ref,
                                                      move_to_apply.get_des_block(), move_to_apply.get_tasks())
            self.unload_buses(ex_dp)  # unload buses
            self.unload_read_mem(ex_dp)  # unload memories
            ex_dp.hardware_graph.update_graph(block_to_prime_with=move_to_apply.get_des_block())  # update the hardware graph
            if config.DEBUG_SANITY:ex_dp.sanity_check() # sanity check
        elif move_to_apply.get_transformation_name() == "split":
            self.unload_buses(ex_dp)  # unload buss
            if blck_ref.type == "ic":
                succeeded,_ = self.fork_bus(ex_dp, blck_ref, move_to_apply.get_tasks())
            else:
                succeeded,_ = self.fork_block(ex_dp, blck_ref, move_to_apply.get_tasks())
            if config.DEBUG_SANITY:ex_dp.sanity_check() # sanity check
        elif move_to_apply.get_transformation_name() == "split_swap":
            # first split
            previous_designs_blocks = ex_dp.get_blocks()
            self.unload_buses(ex_dp)  # unload buss
            if blck_ref.type == "ic":
                succeeded, new_block = self.fork_bus(ex_dp, blck_ref, move_to_apply.get_tasks())
            else:
                succeeded, new_block = self.fork_block(ex_dp, blck_ref, move_to_apply.get_tasks())
            ex_dp.hardware_graph.update_graph(block_to_prime_with=blck_ref)  # update the hardware graph
            if succeeded:
                """
                current_blocks = ex_dp.get_blocks()
                new_block = list(set(current_blocks) - set(previous_designs_blocks))[0]

                # we need to do this, because sometimes the migrant tasks are swap and hence the new block gets the
                # other migrant tasks
                if len(new_block.get_tasks_of_block()) == 1:
                    block_to_swap = new_block
                else:
                    block_to_swap =blck_ref
                """
                block_to_swap = new_block
                if config.DEBUG_SANITY:ex_dp.sanity_check() # sanity check
                succeeded = self.swap_block(block_to_swap, move_to_apply.get_des_block())
                succeeded = self.mig_tasks_of_src_to_dest(ex_dp, block_to_swap,
                                                          move_to_apply.get_des_block(), move_to_apply.get_tasks())
                self.unload_buses(ex_dp)  # unload buses
                self.unload_read_mem(ex_dp)  # unload memories
                ex_dp.hardware_graph.update_graph(block_to_prime_with=move_to_apply.get_des_block())  # update the hardware graph
            else:
                print("something went wrong with split swap")

            if config.DEBUG_SANITY:ex_dp.sanity_check() # sanity check
        elif  move_to_apply.get_transformation_name() == "cleanup":
            self.unload_buses(ex_dp)  # unload buses
            self.unload_read_mem(ex_dp)  # unload memories
            if move_to_apply.get_transformation_sub_name() == "absorb":
                ref_block_neighs = blck_ref.get_neighs()
                for block in ref_block_neighs:
                    block.disconnect(blck_ref)
                    block.connect(move_to_apply.get_des_block())
                ex_dp.hardware_graph.update_graph_without_prunning(block_to_prime_with=move_to_apply.get_des_block())  # update the hardware graph
                succeeded = True
            else:
                if not blck_ref.type == "ic":  # ic migration is not supported
                    succeeded = self.mig_tasks_of_src_to_dest(ex_dp, blck_ref, move_to_apply.get_des_block(), move_to_apply.get_tasks())

                    ex_dp.hardware_graph.update_graph(block_to_prime_with=move_to_apply.get_des_block())  # update the hardware graph
                else:
                    succeeded = False
            if config.DEBUG_SANITY:ex_dp.sanity_check() # sanity check
        elif move_to_apply.get_transformation_name() == "migrate" or move_to_apply.get_transformation_name() == "cleanup":
            self.unload_buses(ex_dp)  # unload buses
            self.unload_read_mem(ex_dp)  # unload memories
            if not blck_ref.type == "ic":  # ic migration is not supported
                succeeded = self.mig_tasks_of_src_to_dest(ex_dp, blck_ref, move_to_apply.get_des_block(), move_to_apply.get_tasks())

                ex_dp.hardware_graph.update_graph(block_to_prime_with=move_to_apply.get_des_block())  # update the hardware graph
            else:
                succeeded = False
            if config.DEBUG_SANITY:ex_dp.sanity_check() # sanity check
        elif move_to_apply.get_transformation_name() == "dram_fix":
            self.unload_buses(ex_dp)  # unload buses
            self.unload_read_mem(ex_dp)  # unload memories
            self.fix_dram(ex_dp)
            ex_dp.hardware_graph.update_graph_without_prunning()  # update the hardware graph
            succeeded = True
        elif move_to_apply.get_transformation_name() == "transfer":
            self.unload_buses(ex_dp)  # unload buses
            self.unload_read_mem(ex_dp)  # unload memories
            src_block = blck_ref # memory to move
            src_block_ic = [el for el in src_block.get_neighs() if el.subtype == "ic"][0]
            dest_block = move_to_apply.get_des_block()  # destination Ic
            src_block.disconnect(src_block_ic)
            dest_block_ic = [el for el in dest_block.get_neighs() if el.subtype == "ic"][0]
            dest_block_ic.connect(src_block)
            ex_dp.hardware_graph.update_graph(src_block)  # update the hardware graph
            succeeded = True
        elif move_to_apply.get_transformation_name() == "routing":
            self.unload_buses(ex_dp)  # unload buses
            self.unload_read_mem(ex_dp)  # unload memories
            src_block = blck_ref # memory to move
            dest_block = move_to_apply.get_des_block()  # destination Ic
            src_block.connect(dest_block)
            ex_dp.hardware_graph.update_graph(dest_block)  # update the hardware graph
            succeeded = True
        else:
            raise Exception("transformation :" + move_to_apply.get_transformation_name() + " is not supported")

        ex_dp.hardware_graph.pipe_design()
        return ex_dp, succeeded

    # ------------------------------
    # Functionality:
    #    Relax the bottleneck either by: 1.using an improved block (swap) from the DB 2.reducing contention (splitrr)
    # Variables:
    #       des_tup: design tuple containing (ex_dp, sim_dp)
    #       mode: whether to use hot_kernel or not for deciding the move
    #       hot_kernel_pos: position of the hot kernel
    # ------------------------------
    def relax_bottleneck(self, des_tup, mode, hot_kernel_pos = 0):
        split_coeff = 5  # determines the probability to pick split (directing the hill-climber to pick a move)
        swap_coeff = 1  # determines the probability to pick swap
        if mode == "hot_kernel":
            self.unload_read_mem(des_tup[0])

            # determine if swap is beneficial (if there is a better hardware block in the data base
            # that can improve the current designs performance)
            swap_beneficial, sup_block, hot_block = self.block_is_improvable(des_tup, hot_kernel_pos)
            if config.DEBUG_FIX: random.seed(0)
            else: time.sleep(.00001), random.seed(datetime.now().microsecond)

            # if swap is beneficial, (possibly) give swap a shot
            if swap_beneficial:
                print(" block to improve is of type" + hot_block.type)
                # favoring swap for non pe's and split for pes (saving financial cost)

                if not hot_block.type == "pe":
                    split_coeff = 1
                    swap_coeff = 5
                else:
                    # favoring swap over swap to avoid
                    # unnecessary customizations.
                    split_coeff = 5
                    swap_coeff = 1
                move_to_choose = random.choice(swap_coeff*["swap"]+ split_coeff*["split"])
            else: move_to_choose = "split"
            self.move_to(move_to_choose, sup_block, hot_block, des_tup, mode, hot_kernel_pos)
        else:
            raise Exception("mode:" + mode + " not defined" )

    # ------------------------------
    # Functionality:
    #    swaping a block with an improved one to improve the performance
    # Variables:
    #       swapee: the block to swap
    #       swaper: the block to swap with
    # ------------------------------
    def swap_block(self, swapee, swaper):
        # find and attache a similar block
        neighs = swapee.neighs[:]
        for neigh in neighs:
            neigh.disconnect(swapee)
        for neigh in neighs:
            neigh.connect(swaper)

    # ------------------------------
    # Functionality:
    #    determine wether there is block in the database (of the same kind) that is superior
    # Variables:
    #       des_tup: design tuple containing ex_dp, sim_dp
    #       hot_kernel_post: position of the hottest kernel. Helps locating the kernel.
    # ------------------------------
    def block_is_improvable(self, des_tup, hot_kernel_pos):
        ex_dp, sim_dp = des_tup
        hot_blck = self.sim_dp.get_dp_stats().get_hot_block_system_complex(des_tup, hot_kernel_pos)
        hot_blck_synced = self.find_cores_hot_kernel_blck_bottlneck(ex_dp, hot_blck)
        new_block = self.database.up_sample_blocks(hot_blck_synced, "immediate_superior", hot_blck.get_tasks_of_block()) # up_sample = finding a superior block
        if self.boost_SOC: new_block = self.database.up_sample_SOC(new_block, config.sorting_SOC_metric)
        else: block = self.database.copy_SOC(new_block, hot_blck_synced)
        if block.get_generic_instance_name() != hot_blck_synced.get_generic_instance_name():
            return True, new_block, hot_blck_synced
        return False, "_", "_"

    # ------------------------------
    # Functionality:
    #    clustering tasks bases on their dependencies, i.e, having the same child or parent (this is to improve locality)
    # Variables:
    #       task: task of interest to look at.
    #       num_of_tasks_to_migrate: how many task to migrate to a new block (if we decide to split)
    #       residing_tasks_on_pe: tasks that are currently occupied the processing element of interest
    #       clusters: clusters already formed (This is because the helper is called recursively)
    # ------------------------------
    def cluster_tasks_based_on_dependency_helper(self, task, num_of_tasks_to_migrate, residing_tasks_on_pe, residing_tasks_on_pe_copy, clusters):
        if num_of_tasks_to_migrate == 0:
            return 0
        else:
            # go through the children and if they are on the pe and not already selected in the cluster
            task_s_children_on_pe = []
            for child in task.get_children():
                if child in residing_tasks_on_pe and child not in clusters[0]:
                    task_s_children_on_pe.append(child)
            task_children_queue = []  # list to keep the childs for breadth first search

            # no children that satisfies the requirement (including no children at all)
            if not task_s_children_on_pe:
                task = random.choice(residing_tasks_on_pe_copy)
                clusters[0].append(task)
                residing_tasks_on_pe_copy.remove(task)
                num_of_tasks_to_migrate -= 1
                return self.cluster_tasks_based_on_dependency_helper(task, num_of_tasks_to_migrate, residing_tasks_on_pe, residing_tasks_on_pe_copy, clusters)

            # iterate through children and add them to teh cluster
            for child in task_s_children_on_pe:
                if num_of_tasks_to_migrate == 0:
                    return 0
                task_children_queue.append(child)
                clusters[0].append(child)
                residing_tasks_on_pe_copy.remove(child)
                num_of_tasks_to_migrate -= 1

            # generate tasks to migrate
            # and recursively call the helper
            for child in task_children_queue:
                num_of_tasks_to_migrate = self.cluster_tasks_based_on_dependency_helper(child, num_of_tasks_to_migrate, residing_tasks_on_pe,
                                                              residing_tasks_on_pe_copy, clusters)
                if num_of_tasks_to_migrate == 0:
                    return 0


    def cluster_tasks_based_on_tasks_parallelism(self, ex_dp, sim_dp, task, block):
        return self.find_parallel_tasks_of_task_in_block(ex_dp, sim_dp, task, block)

    def cluster_tasks_based_on_tasks_serialism(self, ex_dp, sim_dp, task, block):
        return self.find_serial_tasks_of_task_in_block(ex_dp, sim_dp, task, block)

    # ------------------------------
    # Functionality:
    #    clustering tasks bases on their dependencies, i.e, having the same child or parent (this is to improve locality)
    # Variables:
    #       residing_task_on_block: tasks that are already occupying the block (that we want to split
    #       num_clusters: how many clusters to generate for migration.
    # ------------------------------
    def cluster_tasks_based_on_tasks_dependencies(self, task_ref, residing_tasks_on_block, num_clusters):
        cluster_0 = []
        clusters_length = int(len(residing_tasks_on_block)/num_clusters)
        residing_tasks_copy = residing_tasks_on_block[:]
        for tsk in residing_tasks_copy:
            if tsk.name == task_ref.name:
                ref_task = tsk
                break

        #ref_task = random.choice(residing_tasks_on_block)

        cluster_0.append(ref_task)
        residing_tasks_copy.remove(ref_task)
        tasks_to_add_pool = [ref_task]  # list containing all the tasks that have been added so far.
                                # we sample from it to pick a ref block with some condition

        if config.DEBUG_FIX: random.seed(0)
        else: time.sleep(.00001); random.seed(datetime.now().microsecond)

        break_completely = False
        # generate clusters
        while len(cluster_0) < clusters_length:
            ref_tasks_family = ref_task.get_family()
            family_on_block = set(ref_tasks_family).intersection(set(residing_tasks_copy))
            family_on_block_not_already_in_cluster = family_on_block - set(cluster_0)
            # find tasks family to migrate (task family is used for clustering to improve locality)
            tasks_to_add = list(family_on_block_not_already_in_cluster)
            for task in tasks_to_add:
                if len(cluster_0) < clusters_length:
                    cluster_0.append(task)
                    residing_tasks_copy.remove(task)
                else:
                    break_completely = True
                    break
            if break_completely:
                break
            if tasks_to_add: ref_task = random.choice(tasks_to_add)
            else:
                if tasks_to_add_pool:
                    ref_task = random.choice(tasks_to_add_pool)
                    tasks_to_add_pool.remove(ref_task)
                else:
                    ref_task = random.choice(residing_tasks_copy)
                    tasks_to_add_pool.append(ref_task)
            tasks_to_add_pool = list(set(tasks_to_add_pool + tasks_to_add))

        return [cluster_0, residing_tasks_copy]

    # ------------------------------
    # Functionality:
    #    clustering tasks if they share input/outputs to ensure high locality.
    # Variables:
    #       residing_task_on_block: tasks that are already occupying the block (that we want to split
    #       num_clusters: how many clusters to generate for migration.
    # ------------------------------
    def cluster_tasks_based_on_data_sharing(self, task_ref, residing_tasks_on_block, num_clusters):
        if (config.tasks_clustering_data_sharing_method == "task_dep"):
            return self.cluster_tasks_based_on_tasks_dependencies(task_ref, residing_tasks_on_block, num_clusters)
        else:
            raise Exception("tasks_clustering_data_sharing_method:" + config.tasks_clustering_data_sharing_method + "not defined")

    # ------------------------------
    # Functionality:
    #       random clustering of the tasks (for split). To introduce stochasticity in the system.
    # Variables:
    #       residing_task_on_block: tasks that are already occupying the block (that we want to split
    #       num_clusters: how many clusters to generate for migration.
    # ------------------------------
    def cluster_tasks_randomly(self, residing_tasks_on_pe, num_clusters=2):
        clusters:List[List[Task]] = [[] for i in range(num_clusters)]
        residing_tasks_on_pe_copy = residing_tasks_on_pe[:]

        if (config.DEBUG_FIX):
            random.seed(0)
        else:
            time.sleep(.00001)
            random.seed(datetime.now().microsecond)

        # pick some random number of tasks to migrate
        num_of_tasks_to_migrate = random.choice(range(1, len(residing_tasks_on_pe_copy)))
        # assign them to the first cluster
        for _ in range(num_of_tasks_to_migrate):
            task = random.choice(residing_tasks_on_pe_copy)
            clusters[0].append(task)
            residing_tasks_on_pe_copy.remove(task)

        # assign the rest of tasks to cluster 2
        for task in residing_tasks_on_pe_copy:
            clusters[1].append(task)

        # get rid of the empty clusters  (happens if num of clusters is one less than the total number of tasks)
        return [cluster for cluster in clusters if cluster]

    # ------------------------------
    # Functionality:
    #       Migrate all the tasks, from the known src to known destination block
    # Variables:
    #       dp:  design
    #       dest_blck: destination block. Block to migrate the tasks to.
    #       src_blck: source block, where task currently lives in.
    #       tasks:  the tasks to migrate
    # ------------------------------
    def mig_tasks_of_src_to_dest(self, dp: ExDesignPoint, src_blck, dest_blck, tasks):

        # sanity check
        for task in tasks:
            matched_block = False # if we there is an equality between src_block and current_src_blck (these need to be the same ensuring that nothing has gone wrong)
            # check if task is only a read task on ic or memory. If so raise the exception
            # since we have already unloaded it.
            task_dirs = [task_dir[1] for task_dir in src_blck.get_tasks_dir_work_ratio()]
            if "read" in task_dirs and len(task_dirs) == 1:
                raise NoMigrantException  # this scenario is an exception, but non read scenarios are errors

            cur_src_blocks = dp.get_blocks_of_task(task)
            for cur_src_blck in cur_src_blocks:
                if dest_blck.type == cur_src_blck.type:  # only pay attention to the block of the type similar to the one that you migrate to
                    if cur_src_blck == src_blck:
                        matched_block = True
            if not matched_block:
                print("task does not exist int the block")
                raise NoMigrantError
        for task in tasks:
            self.mig_one_task(dest_blck, src_blck, task)

    # ------------------------------
    # Functionality:
    #       Migrate one task, from the known src to known destination block
    # Variables:
    #       dest_blck: destination block. Block to migrate the tasks to.
    #       src_blck: source block, where task currently lives in.
    #       task:  the task to migrate
    # ------------------------------
    def mig_one_task(self, dest_blck, src_blck, task):
        # prevent migrating to yourself
        # random bugs would pop up if we do so
        if src_blck.instance_name == dest_blck.instance_name:
            return
        #work_ratio = src_blck.get_task_s_work_ratio_by_task(task)

        if (src_blck.type == "pe"):
            #print("blah blah the taks to migrate is" + task.name + " from " + src_blck.instance_name +  "to" + dest_blck.instance_name)

            dest_blck.load_improved(task, task)
            dir_ = "loop_back"
        else:  # write
            family_tasks_on_block = src_blck.get_task_s_family_by_task_and_dir(task, "write")  # just the names
            #assert (len(work_ratio) == 1), "only can have write work ratio at this stage"
            #print("blah blah the taks to migrate is" + task.name + " from " + src_blck.instance_name +  "to" + dest_blck.instance_name)
            if len(task.get_children()) ==0:
                print("This should not be happening")
                raise TaskNoChildrenError

            for family in family_tasks_on_block:
                family_task = [child for child in task.get_children() if family== child.name][0]
                dest_blck.load_improved(task, family_task)
            dir_ = "write"
        src_blck.unload((task, dir_))

        # delete this later
        tasks_left = []
        for task in src_blck.get_tasks_of_block():
            tasks_left.append(task.name)
        #print("blah blah tasks left on src block is " + str(tasks_left))

        if len(src_blck.get_tasks_of_block()) == 0:  # prunning out the block
            src_blck.disconnect_all()

    # ------------------------------
    # Functionality:
    #       Migrate all of the tasks from a known src to a known destination
    # Variables:
    #       dest_blck: destination block. Block to migrate the tasks to.
    #       src_blck: source block, where task currently lives in.
    # ------------------------------
    def mig_cur_tasks_of_src_to_dest(self, src_blck, dest_blck):
        tasks = src_blck.get_tasks_of_block()
        _ = [self.mig_one_task(dest_blck, src_blck, task) for task in tasks]

    # used when forking buses.
    # a special procedure needs to be followed
    # just as deciding on disconnecting and connecting the neighbouring PEs/MEMS and buses
    def attach_alloc_block_to_bus(self, ex_dp, block_to_mimic, mimicee_block, migrant_tasks):
        pe_neighs = [neigh for neigh in block_to_mimic.neighs if neigh.type == "pe"]
        mem_neighs = [neigh for neigh in block_to_mimic.neighs if neigh.type == "mem"]
        ic_neighs = [neigh for neigh in block_to_mimic.neighs if neigh.type == "ic"]
        migrant_tasks_names = [el.get_name() for el in migrant_tasks]
        for neigh in  pe_neighs + mem_neighs:
            neigh_tasks = [el.get_name() for el in neigh.get_tasks_of_block()]
            # if no overlap skip
            if len(list(set(migrant_tasks_names) - set(neigh_tasks) )) == len(migrant_tasks_names):
                continue
            else:
                mimicee_block.connect(neigh)
                block_to_mimic.disconnect(neigh)

        # randomly connect the ics.
        # we can do better here
        ctr = 0
        for ic in ic_neighs:
            if (ctr %2) == 1:
                ic.disconnect(block_to_mimic)
                ic.connect(mimicee_block)
            ctr +=1
        mimicee_block.connect(block_to_mimic)




    # ------------------------------
    # attach through the shared bus
    # block_to_mimic: this is the block which we use as a template for connections
    # mimicee_block: this the block that needs to mimic the block_to_mimic block's behavior (by attaching to its blocks)
    # ------------------------------
    def attach_alloc_block(self, ex_dp, block_to_mimic, mimicee_block):
        if(block_to_mimic.type == "ic"): # connect to all the mems and pes of the block_to_mimic
            pe_neighs = [neigh for neigh in block_to_mimic.neighs if neigh.type == "pe"]
            mem_neighs = [neigh for neigh in block_to_mimic.neighs if neigh.type == "mem"]
            _ = [mimicee_block.connect(neigh) for neigh in pe_neighs]
            _ = [mimicee_block.connect(neigh) for neigh in mem_neighs]
            #connect = block_to_mimic.connect(mimicee_block)
        else:
            bus_neighs = [neigh for neigh in block_to_mimic.neighs if neigh.type == "ic"]
            assert(len(bus_neighs) == 1), "can only have one bus neighbour"
            #print("attaching block to " + bus_neighs[0].instance_name)
            mimicee_block.connect(bus_neighs[0])

        ex_dp.hardware_graph.update_graph(block_to_prime_with=block_to_mimic)
        #self.sanity_check_ic(ex_dp)

    # ------------------------------
    # Functionality:
    #       allocating similar block for split purposes. We find a block of the same type and also superior.
    # Variables:
    #       old_blck: block to improve on.
    #       tasks: tasks residing on the old block. We need this because the new block should support all the
    #       these tasks
    # ------------------------------
    def allocate_similar_block(self, old_block, tasks):
        new_block = self.database.sample_similar_block(old_block)
        new_block = self.database.copy_SOC(new_block, old_block)
        return new_block

    # ------------------------------
    # Functionality:
    #       split the tasks between two clusters, one cluster having only one
    #       task (found based on the selected kernel) and the other, the rest
    # Variables:
    #       tasks_of_block: tasks resident on the block to choose from
    #       selected_kernel: the selected kernel to separate
    # ------------------------------
    def separate_a_task(self, tasks_of_block, selected_kernel):
        clusters = [[],[]]
        for task in tasks_of_block:
            if selected_kernel.get_task().name == task.name:
                clusters[0].append(task)
            else:
                clusters[1].append(task)
        return clusters

    # ------------------------------
    # Functionality:
    #      cluster tasks. This decides which task to migrate together. Used in split.
    # Variables:
    #       block: block where tasks resides in.
    #       num_clusters: how many clusters do we want to have.
    # ------------------------------
    def cluster_tasks(self, ex_dp, sim_dp, block, selected_kernel, selection_mode):
        if selection_mode == "random":
            return self.cluster_tasks_randomly(block.get_tasks_of_block())
        elif selection_mode == "tasks_dependency":
            return self.cluster_tasks_based_on_data_sharing(selected_kernel.get_task(), block.get_tasks_of_block(), 2)
        elif selection_mode == "single":
            return self.separate_a_task(block.get_tasks_of_block(), selected_kernel)
        elif selection_mode == "single_serialism":
            return [[selected_kernel.get_task()]]
        #elif selection_mode == "batch":
        #    return self.cluster_tasks_based_on_data_sharing(selected_kernel.get_task(), block.get_tasks_of_block(), 2)
        elif selection_mode == "batch":
            return self.cluster_tasks_based_on_tasks_parallelism(ex_dp, sim_dp, selected_kernel.get_task(), block)
        #elif selection_mode == "batch_nonparallel":
        #    return self.cluster_tasks_based_on_tasks_serialism(ex_dp, sim_dp, selected_kernel.get_task(), block)
        else:
            raise Exception("migrant clustering policy:" + selection_mode + "not supported")

    # ------------------------------
    # Functionality:
    #      Which one of the clusters to migrate.
    # Variables:
    #       block: block where tasks resides in.
    # ------------------------------
    def migrant_selection(self, ex_dp, sim_dp, block_after_unload, block_before_unload, selected_kernel, selection_mode):
        if config.DEBUG_FIX: random.seed(0)
        else: time.sleep(.00001), random.seed(datetime.now().microsecond)
        try:
            clustered_tasks = self.cluster_tasks(ex_dp,sim_dp, block_after_unload, selected_kernel, selection_mode)
        except:
            print("migrant selection went wrong. This needs to be fixed. Most likely occurs with random (As opposed to arch-aware) transformation_selection_mode")
            return []
        return clustered_tasks[0]

    # ------------------------------
    # Functionality:
    #     see if we can split a block (i.e., there are more than one tasks on it)
    # Variables:
    #      ex_dp: example design
    #      block: block of interest
    #      mode: depracated. TODO: get rid of it.
    # ------------------------------
    def block_forkable(self, ex_dp, block):
        if len(block.get_tasks_of_block()) < config.num_clusters:
            return False
        else:
            return True

    def task_in_block(self, block, task_):
        return (task_.name in [task.name for task in block.get_tasks_of_block()])

    # ------------------------------
    # Functionality:
    #     finds another block similar to the input block and attaches itself similarly
    # Variables:
    #      ex_dp: example design
    #      block: block of interest
    #      mode: deprecated. TODO: get rid of it.
    # ------------------------------
    def fork_block(self, ex_dp, block, migrant_tasks_non_filtered):

        migrant_tasks = []  # filter the tasks that don' exist on the block. This usually happens because we might unload the bus/memory
        # transformation gaurds
        if len(block.get_tasks_of_block()) < config.num_clusters:
            return False,""
        else:
            for task__ in migrant_tasks_non_filtered:
                # if tasks to migrate does not exist on the src block
                if not(task__.name in [task.name for task in block.get_tasks_of_block()]):  # this only should occur for reads,
                                                                                            # since we unload the reads
                    continue
                else:
                    migrant_tasks.append(task__)

        if len(migrant_tasks) == 0:
            return False,""

        # find and attach a similar block
        alloc_block = self.allocate_similar_block(block, migrant_tasks)
        #print("allocate block name" + alloc_block.instance_name)
        self.attach_alloc_block(ex_dp, block, alloc_block)
        # migrate tasks
        #self.mig_tasks_of_diff_blocks(ex_dp, migrant_tasks, alloc_block)
        self.mig_tasks_of_src_to_dest(ex_dp, block, alloc_block, migrant_tasks)
        ex_dp.hardware_graph.update_graph(block_to_prime_with=alloc_block)
        if config.VIS_GR_PER_GEN: vis_hardware(ex_dp)

        ex_dp.check_mem_fronts_sanity()
        return True, alloc_block

    # ------------------------------
    # Functionality:
    #     find out how many tasks do two blocks share. Used for split move.
    # Variables:
    #      ex_dp: example design
    #      block: block of interest
    #      mode: depracated. TODO: get rid of it.
    # ------------------------------
    def calc_data_sharing_amount(self, pe, mem):
        pe_tasks_name = [task.name for task in pe.get_tasks_of_block()]
        mem_tasks_name = [task.name for task in mem.get_tasks_of_block()]
        sharing_ctr = 0

        for pe_task_name in pe_tasks_name:
            for mem_task_name in mem_tasks_name:
                if pe_task_name == mem_task_name:
                    sharing_ctr +=1
        return sharing_ctr

    # ------------------------------
    # Functionality:
    #       Cluster blocks based on the number of tasks they share. This is used when deciding which block to split.
    # Variables:
    #       ex_dp: example design
    #       block: block to prime the selection with.
    # ------------------------------
    def cluster_blocks_based_on_data_sharing(self, ex_dp, block):
        self.pruned_one = False
        ex_dp.hardware_graph.update_graph(block_to_prime_with=block)
        if config.VIS_GR_PER_GEN: vis_hardware(ex_dp)

        pe_neighs = [neigh for neigh in block.neighs if neigh.type == "pe"]
        mem_neighs = [neigh for neigh in block.neighs if neigh.type == "mem"]

        sharing_matrix = {}
        from collections import defaultdict
        sharing_dict = {}
        for pe in pe_neighs:
            for mem in mem_neighs:
                sharing_dict[(pe, mem)] = self.calc_data_sharing_amount(pe, mem)

        sorted_sharing_list = sorted(sharing_dict.items(), key=operator.itemgetter(1))
        return sorted_sharing_list

    # ------------------------------
    # Functionality:
    #       fork (split) a bus.
    # Variables:
    #       ex_dp: example design
    #       block: bus to split
    #       mode: how to split. Right now, only uses hot kernel.
    # ------------------------------
    def fork_bus(self, ex_dp, block, migrant_tasks):

        self.pruned_one = False
        ex_dp.hardware_graph.update_graph(block_to_prime_with=block)
        if config.VIS_GR_PER_GEN: vis_hardware(ex_dp)

        pe_neighs = [neigh for neigh in block.neighs if neigh.type == "pe"]   # find pe neighbours
        mem_neighs = [neigh for neigh in block.neighs if neigh.type == "mem"]  # find memory neighbours

        pe_forked = False
        pe_good_to_go = False
        mem_forked = False
        mem_good_to_go = False

        # first check forkability of the neighbouring pe and mem (before attempting to fork either)
        # if either (mem, pe) needs to be forked
        if len(pe_neighs) == 1 or len(mem_neighs) == 1:
            pe_forkability = True
            mem_forkability = True
            # see if you can fork
            if len(pe_neighs) == 1:
                for task_ in migrant_tasks:
                    if not (self.block_forkable(ex_dp, pe_neighs[0]) or self.task_in_block(block,
                                                                                       task_)):
                        pe_forkability = False
                        break
            # see if you can fork
            if len(mem_neighs) == 1:
                for task_ in migrant_tasks:
                    if not (self.block_forkable(ex_dp, mem_neighs[0]) or self.task_in_block(mem_neighs[0],
                                                                                        task_)):
                        mem_forkability = False

            if not (mem_forkability and pe_forkability):
                return False,""

        # now fork the neighbours if necessary
        if len(pe_neighs) == 1:
            pe_forked,_ = self.fork_block(ex_dp,  pe_neighs[0], migrant_tasks)
            ex_dp.hardware_graph.update_graph(block_to_prime_with=block)
        else:
            pe_good_to_go = True

        if len(mem_neighs) == 1:
            mem_forked,_= self.fork_block(ex_dp,  mem_neighs[0], migrant_tasks)
            ex_dp.hardware_graph.update_graph(block_to_prime_with=block)
        else:
            mem_good_to_go = True

        if not((pe_forked or pe_good_to_go) and (mem_forked or mem_good_to_go)):
            return False, ""

        # allocate and attach a similar bus
        alloc_block = self.allocate_similar_block(block, [])



        # at the moment, we don't support the smarter version of attaching the blocks for buses
        if config.RUN_VERIFICATION_PER_GEN or config.RUN_VERIFICATION_PER_IMPROVMENT or config.RUN_VERIFICATION_PER_NEW_CONFIG:
            self.attach_alloc_block(ex_dp, block, alloc_block)
            ex_dp.hardware_graph.update_graph(block_to_prime_with=block)
            #if config.VIS_GR_PER_GEN: vis_hardware(ex_dp)

            # prune blocks (memory and processing elements) from the previous pus
            self.prune(ex_dp, block, alloc_block)
            block.connect(alloc_block)
        else:
            self.attach_alloc_block_to_bus(ex_dp, block, alloc_block, migrant_tasks)


        ex_dp.hardware_graph.update_graph(block_to_prime_with=block)
        return True, alloc_block

    # ------------------------------
    # Functionality:
    #       reducing contention on the bottleneck block. Used specifically for split.
    # Variables:
    #       des_tup:  design tuple containing ex_dp, sim_dp
    #       mode: mode used for splitting. At the moment, we only use hottest kernel as the deciding factor.
    #       hot_kernel_pos: position of the hot kernel. Helps locating the hot kernel.
    # ------------------------------
    def reduce_contention(self, des_tup, mode, hot_kernel_pos):
        ex_dp, sim_dp = des_tup
        if mode == "hot_kernel":
            hot_block = self.sim_dp.get_dp_stats().get_hot_block_system_complex(des_tup, hot_kernel_pos)
            hot_blck_synced = self.find_cores_hot_kernel_blck_bottlneck(ex_dp, hot_blck)
            if hot_block.type == "ic":
                self.fork_bus(ex_dp, hot_blck_synced)
            else:
                self.fork_block(ex_dp, hot_blck_synced)

    # ------------------------------
    # Functionality:
    #       allocate a new block, connect it to the current design and migrate some of tasks to it.
    # Variables:
    #       dp:  design point
    #       blck_bottlenck: block bottleneck  (this block is forked)
    # ------------------------------
    def alloc_mig(self, dp, blck_bottleneck):

        # find tasks to migrate
        clustered_tasks = []
        if blck_bottleneck.type == "ic":
            pe_neighs = [neigh for neigh in blck_bottleneck.neighs if neigh.type == "pe"]
            mem_neighs = [neigh for neigh in blck_bottleneck.neighs if neigh.type == "mem"]

            if len(pe_neighs) == 1:
                self.alloc_mig(dp,  pe_neighs[0])

            if len(pe_neighs) == 1:
                self.alloc_mig(dp,  mem_neighs[0])
        else:
            clustered_tasks = self.naively_cluster_tasks(blck_bottleneck.get_tasks_of_block())
            if config.DEBUG_FIX:
                random.seed(0)
            else:
                time.sleep(.00001)
                random.seed(datetime.now().microsecond)
            tasks_to_migrate = clustered_tasks[random.choice(range(0, len(clustered_tasks)))]  # grab a random cluster

        alloc_block = self.allocate_similar_block(blck_bottleneck, clustered_tasks)
        self.attach_alloc_block(dp, blck_bottleneck, alloc_block)
        if not(blck_bottleneck == "ic"):
            self.mig_tasks(dp, tasks_to_migrate, alloc_block)

    # ------------------------------
    # Functionality:
    #      used in pruning an bus(ic) from the blocks that are connected (through the fork process)  to the new bus.
    # Variables:
    #       dp: current design point.
    #       original_ic: ic to prune
    #       new_ic: new ic that has inherited some of the old ic neighbours (pc, mem)
    # ------------------------------
    def prune(self, dp:ExDesignPoint, original_ic, new_ic):
        # recursively prone the blocks in the dp
        # start with any block and traverse (since blocks
        self.prune_smartly(original_ic, dp, 2, new_ic)
        dp.hardware_graph.update_graph(block_to_prime_with=new_ic)

    # ------------------------------
    # Functionality:
    #     find memories that share the most number of tasks with a pe. Used for pruning smartly.
    # Variables:
    #       mem_clusters: cluster of memories under question.
    #       pe: pe to measure closeness with.
    # ------------------------------
    def find_closest_mem_cluster(self, mem_clusters, pe):
        def calc_mem_similarity(pe, mem):
            pe_tasks_name = [task.name for task in pe.get_tasks_of_block()]
            mem_tasks_name = [task.name for task in mem.get_tasks_of_block()]
            sharing_ctr = 0

            for pe_task_name in pe_tasks_name:
                for mem_task_name in mem_tasks_name:
                    if pe_task_name == mem_task_name:
                        sharing_ctr += 1
            return sharing_ctr

        def calc_cluster_similarity(pe, mem_cluster):
            similarity = 0
            for mem in mem_cluster:
                similarity += calc_mem_similarity(pe, mem)
            return similarity

        cluster_similarity= []
        for mem_cluster in mem_clusters:
            cluster_similarity.append(calc_cluster_similarity(pe, mem_cluster))
        return cluster_similarity.index(max(cluster_similarity))

    # ------------------------------
    # Functionality:
    #      used in pruning an bus(ic) from the blocks that are connected (through the fork process)  to the new bus.
    # Variables:
    #       original_ic: ic to prune
    #       ex_dp:  example design (current design)
    #       num_clusters: number of clusters for task migration
    #       new_ic: new ic that has inherited some of the old ic neighbours (pc, mem)
    # ------------------------------
    def prune_smartly(self, original_ic, ex_dp, num_clusters, new_ic):

        # this is used for balancing the clusters (in case one
        # of the pe clusters is left out empty
        def reshuffle_clusters_if_necessary(mem_clusters, pe_clusters):
            if len(pe_clusters) > 2:
                raise Exception("more than two cluster not supported yet")
            # if any of the clusters are empty, reshuffle
            if any([True for pe_cluster in pe_clusters if len(pe_cluster) <= 0]):
                pes = pe_clusters[0] + pe_clusters[1]
                pe_clusters[0] = pes[0: int(len(pes)/2)]
                pe_clusters[1] = pes[int(len(pes)/2): len(pes)]

        pe_neighs = [neigh for neigh in original_ic.neighs if neigh.type == "pe"]
        mem_neighs = [neigh for neigh in original_ic.neighs if neigh.type == "mem"]
        ic_neighs = [neigh for neigh in original_ic.neighs if neigh.type == "ic"]

        # disconnect the original ic from all of its neighbours
        original_ic.disconnect_all()

        # connect back some of the neighbours
        for ic_neigh in ic_neighs:
            original_ic.connect(ic_neigh)

        # cluster memory and pe neighbours (each cluster would be assigned to one ic and then get connected to it)
        mem_clusters = [mem_neighs[0: int(len(mem_neighs)/2)], mem_neighs[int(len(mem_neighs)/2): len(mem_neighs)]]
        pe_clusters = [[] for _ in mem_clusters]
        for pe in pe_neighs:
            cluster_idx = self.find_closest_mem_cluster(mem_clusters, pe)
            pe_clusters[cluster_idx].append(pe)
        reshuffle_clusters_if_necessary(mem_clusters, pe_clusters)
        pe_mem_clusters = [pe_clusters[0]+mem_clusters[0], pe_clusters[1] + mem_clusters[1]]

        # connect/disconnect the clusters to the ics
        for block in pe_mem_clusters[0]:
            block.disconnect_all()
            original_ic.connect(block)