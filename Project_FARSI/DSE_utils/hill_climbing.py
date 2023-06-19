#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
from copy import *
from decimal import Decimal
import zipfile
import csv
import _pickle as cPickle
#import ujson
from design_utils.components.hardware import *
from design_utils.components.workload import *
from design_utils.components.mapping import *
from design_utils.components.scheduling import *
from SIM_utils.SIM import *
from design_utils.design import *
from design_utils.des_handler import *
from design_utils.components.krnel import *
from typing import Dict, Tuple, List
from settings import config
from visualization_utils import vis_hardware, vis_stats, plot
from visualization_utils import vis_sim
#from data_collection.FB_private.verification_utils.common import *
import dill
import pickle
import importlib
import gc
import difflib
#from pygmo import *
#from pygmo.util import *
import psutil


class Counters():
    def __init__(self):
        self.krnel_rnk_to_consider = 0
        self.krnel_stagnation_ctr = 0
        self.fitted_budget_ctr = 0
        self.des_stag_ctr = 0
        self.krnels_not_to_consider = []
        self.population_generation_cnt = 0
        self.found_any_improvement = False
        self.total_iteration_ctr = 0

    def reset(self):
        self.krnel_rnk_to_consider = 0
        self.krnel_stagnation_ctr = 0
        self.fitted_budget_ctr = 0
        self.des_stag_ctr = 0
        self.krnels_not_to_consider = []
        self.population_generation_cnt = 0
        self.found_any_improvement = False


    def update(self, krnel_rnk_to_consider, krnel_stagnation_ctr, fitted_budget_ctr, des_stag_ctr, krnels_not_to_consider, population_generation_cnt, found_any_improvement, total_iteration_ctr):
        self.krnel_rnk_to_consider = krnel_rnk_to_consider
        self.krnel_stagnation_ctr = krnel_stagnation_ctr
        self.fitted_budget_ctr = fitted_budget_ctr
        self.des_stag_ctr = des_stag_ctr
        self.krnels_not_to_consider = krnels_not_to_consider[:]
        self.population_generation_cnt = population_generation_cnt
        self.found_any_improvement = found_any_improvement
        self.total_iteration_ctr = total_iteration_ctr
    #def update_improvement(self, improvement):
    #    self.found_any_improvement = self.found_any_improvement or improvement


    # ------------------------------
# This class is responsible for design space exploration using our proprietary hill-climbing algorithm.
# Our Algorithm currently uses swap (improving the current design) and  duplicate (relaxing the contention on the
# current bottleneck) as two main exploration move.
# ------------------------------
class HillClimbing:
    def __init__(self, database, result_dir):

        # parameters (to configure)
        self.counters = Counters()
        self.found_any_improvement = False
        self.result_dir = result_dir
        self.fitted_budget_ctr = 0  # counting the number of times that we were able to find a design to fit the budget. Used to terminate the search
        self.name_ctr = 0
        self.DES_STAG_THRESHOLD = config.DES_STAG_THRESHOLD   # Acceptable iterations count without improvement before termination.
        self.TOTAL_RUN_THRESHOLD = config.TOTAL_RUN_THRESHOLD  # Total  number of iterations to terminate with.
        self.neigh_gen_mode = config.neigh_gen_mode   # Neighbouring design pts generation mode ("all" "random_one").
        self.num_neighs_to_try = config.num_neighs_to_try  # How many neighs to try around a current design point.
        self.neigh_sel_mode = config.neigh_sel_mode  # Neighbouring design selection mode (best, sometimes, best ...)
        self.dp_rank_obj = config.dp_rank_obj  # Design point ranking object function(best, sometimes, best ...)
        self.num_clusters = config.num_clusters  # How many clusters to create everytime we split.
        self.budget_coeff = config.max_budget_coeff
        self.move_profile = []
        # variables (to initialize)
        self.area_explored = []  # List containing area associated with the all explored designs areas.
        self.latency_explored = []  # List containing latency associated with all explored designs latency.
        self.power_explored = []    # List containing Power associated with all explored designs latency.
        self.design_itr = []  # Design iteration counter. Simply indexing (numbering) the designs explored.
        self.space_distance = 2
        self.database = database  # hw/sw database to use for exploration.
        self.dh = DesignHandler(self.database)  # design handler for design modification.
        self.so_far_best_sim_dp = None     # best design found so far through out all iterations.
                                           # For iteratively improvements.
        self.cur_best_ex_dp, self.cur_best_sim_dp = None, None   # current iteration's best design.
        self.last_des_trail = None # last design (trail)
        self.last_move = None # last move applied
        self.init_ex_dp = None  # Initial exploration design point. (Staring design point for the whole algorithm)
        self.coeff_slice_size = int(self.TOTAL_RUN_THRESHOLD/config.max_budget_coeff)
        #self.hot_krnl_pos = 0  # position of the kernel among the kernel list. Used to found and improve the
                               # corresponding occupying block.

        self.min_cost_to_consider = .000001
        # Counters: to determine which control path the exploration should take (e.g., terminate, pick another block instead
        # of hotblock, ...).
        self.des_stag_ctr = 0  # Iteration count since seen last design improvement.
        self.population_generation_cnt = 0  # Total iteration count (for termination purposes).

        self.vis_move_trail_ctr = 0
        # Sanity checks (preventing bad configuration setup)
        if self.neigh_gen_mode not in ["all", "some"]: raise ValueError()
        # TODO: sel_cri needs to be fixed to include combinations of the objective functions
        if self.dp_rank_obj not in ["all", "latency", "throughput", "power", "design_cost"]: raise ValueError()
        if self.neigh_sel_mode not in ["best", "best_sometime"]: raise ValueError()
        self.des_trail_list = []
        self.krnel_rnk_to_consider = 0   # this rank determines which kernel (among the sorted kernels to consider).
                                         # we use this counter to avoid getting stuck
        self.krnel_stagnation_ctr = 0    # if the same kernel is selected across iterations and no improvement observed,
                                         # count up

        self.recently_seen_design_ctr = 0
        self.recently_cached_designs = {}
        self.cleanup_ctr = 0  # use this to invoke the cleaner once we pass a threshold

        self.SA_current_breadth = -1  # which breadth is current move on
        self.SA_current_mini_breadth = 0  # which breadth is current move on
        self.SA_current_depth = -1  # which depth is current move on
        self.check_point_folder = config.check_point_folder

        self.seen_SOC_design_codes = []  # config code of all the designs seen so far (this is mainly for debugging, concretely
                                     # simulation validation

        self.cached_SOC_sim = {} # cache of designs simulated already. index is a unique code base on allocation and mapping

        self.move_s_krnel_selection = config.move_s_krnel_selection
        self.krnels_not_to_consider = []
        self.all_itr_ex_sim_dp_dict: Dict[ExDesignPoint: SimDesignPoint] = {}  # all the designs look at
        self.reason_to_terminate = ""
        self.log_data_list = []
        self.population_observed_ctr = 0  #
        self.neighbour_selection_time = 0
        self.total_iteration_cnt = 0
        self.total_iteration_ctr = 0
        self.moos_tree = moosTreeModel(config.budgetted_metrics)  # only used for moos heuristic
        self.ctr_l = 0

    def get_total_iteration_cnt(self):
        return self.total_iteration_cnt

    def set_check_point_folder(self, check_point_folder):
        self.check_point_folder = check_point_folder

    # retrieving the pickled check pointed file
    def get_pickeld_file(self, file_addr):
        if not os.path.exists(file_addr):
            file_name = os.path.basename(file_addr)
            file_name_modified = file_name.split(".")[0]+".zip"
            dir_name = os.path.dirname(file_addr)
            zip_file_addr = os.path.join(dir_name, file_name_modified)
            if os.path.exists(zip_file_addr):
                with zipfile.ZipFile(zip_file_addr) as thezip:
                    with thezip.open(file_name, mode='r') as f:
                        obj = pickle.load(f)
            else:
                print(file_addr +" does not exist for unpickling")
                exit(0)
        else:
            with open(file_addr, 'rb') as f:  # will close() when we leave this block
                obj = pickle.load(f)
        return obj

    def populate_counters(self, counters):
        self.counters = counters
        self.krnel_rnk_to_consider = counters.krnel_rnk_to_consider
        self.krnel_stagnation_ctr= counters.krnel_stagnation_ctr
        self.fitted_budget_ctr = counters.fitted_budget_ctr
        self.des_stag_ctr = counters.des_stag_ctr
        self.krnels_not_to_consider = counters.krnels_not_to_consider[:]
        self.population_generation_cnt = counters.population_generation_cnt
        self.found_any_improvement = counters.found_any_improvement
        self.total_iteration_ctr = counters.total_iteration_ctr

    # ------------------------------
    # Functionality
    #       generate initial design point to start the exploration from.
    #       If mode is from_scratch, the default behavior is to pick the cheapest design.
    #       If mode is check_pointed, we start from a previously check pointed design.
    #       If mode is hardcode, we pick a design that is hardcoded.
    # Variables
    #       init_des_point: initial design point
    #       mode: starting point mode (from scratch or from check point)
    # ------------------------------
    def gen_init_ex_dp(self, mode="generated_from_scratch", init_des=""):
        if mode == "generated_from_scratch":  # start from the simplest design possible
            self.init_ex_dp = self.dh.gen_init_des()
        elif mode == "generated_from_check_point":
            pickled_file_addr = self.check_point_folder + "/" + "ex_dp_pickled.txt"
            database_file_addr = self.check_point_folder + "/" + "database_pickled.txt"
            counters_file_addr = self.check_point_folder + "/" + "counters_pickled.txt"
            #sim_pickled_file_addr = self.check_point_folder + "/" + "sim_dp_pickled.txt"
            if "db" in config.check_point_list:
                self.database = self.get_pickeld_file(database_file_addr)
            if "counters" in config.check_point_list:
                self.counters = self.get_pickeld_file(counters_file_addr)
            self.init_ex_dp = self.get_pickeld_file(pickled_file_addr)
            self.populate_counters(self.counters)
        elif mode == "FARSI_des_passed_in":
            self.init_ex_dp = init_des
        elif mode == "hardcoded":
            self.init_ex_dp = self.dh.gen_specific_hardcoded_ex_dp(self.dh.database)
        elif mode == "parse":
            self.init_ex_dp = self.dh.gen_specific_parsed_ex_dp(self.dh.database)
        elif mode == "hop_mode":
            self.init_ex_dp = self.dh.gen_specific_design_with_hops_and_stars(self.dh.database)
        elif mode == "star_mode":
            self.init_ex_dp = self.dh.gen_specific_design_with_a_star_noc(self.dh.database)
        else: raise Exception("mode:" + mode + " is not supported")


    # ------------------------------
    # Functionality:
    #       Generate one neighbouring design based on the moves available.
    #       To do this, we first specify a move and then apply it.
    #       A move is specified by a metric, direction kernel, block, and transformation.
    #       look for move definition in the move class
    # Variables
    #       des_tup: design tuple. Contains a design tuple (ex_dp, sim_dp). ex_dp: design to find neighbours for.
    #                                                                       sim_dp: simulated ex_dp.
    # ------------------------------
    def gen_one_neigh(self, des_tup):
        ex_dp, sim_dp = des_tup

        # Copy to avoid modifying the current designs.
        #new_ex_dp_pre_mod = copy.deepcopy(ex_dp)  # getting a copy before modifying
        #new_sim_dp_pre_mod = copy.deepcopy(sim_dp) # getting a copy before modifying
        #new_ex_dp = copy.deepcopy(ex_dp)
        t1 = time.time()
        gc.disable()
        new_ex_dp = cPickle.loads(cPickle.dumps(ex_dp, -1))
        gc.enable()
        t2 = time.time()
        #new_sim_dp = copy.deepcopy(sim_dp)
        new_des_tup = (new_ex_dp, sim_dp)

        # ------------------------
        # select (generate) a move
        # ------------------------
        # It's important that we do analysis of move selection on the copy (and not the original) because
        # 1. we'd like to keep original for further modifications
        # 2. for block identification/comparison of the move and the copied design
        safety_chk_passed = False
        # iterate and continuously generate moves, until one passes some sanity check
        while not safety_chk_passed:
            move_to_try, total_transformation_cnt = self.sel_moves(new_des_tup, "dist_rank")
            safety_chk_passed = move_to_try.safety_check(new_ex_dp)
            move_to_try.populate_system_improvement_log()

        move_to_try.set_logs(t2-t1, "pickling_time")

        # ------------------------
        # apply the move
        # ------------------------
        # while conduction various validity/sanity checks
        try:
            self.dh.unload_read_mem(new_des_tup[0])    # unload read memories
            move_to_try.validity_check()  # call after unload rad mems, because we need to check the scenarios where
                                          # task is unloaded from the mem, but was decided to be migrated/swapped
            new_ex_dp_res, succeeded = self.dh.apply_move(new_des_tup, move_to_try)
            move_to_try.set_before_after_designs(new_des_tup[0], new_ex_dp_res)
            new_ex_dp_res.sanity_check()  # sanity check
            move_to_try.sanity_check()
            self.dh.load_tasks_to_read_mem_and_ic(new_ex_dp_res)  # loading the tasks on to memory and ic
            new_ex_dp_res.hardware_graph.pipe_design()
            new_ex_dp_res.sanity_check()
        except Exception as e:
            # if the error is already something that we are familiar with
            # react appropriately, otherwise, simply raise it.
            if e.__class__.__name__ in errors_names:
                print("Error: " + e.__class__.__name__)
                # TODOs
                # for now, just return the previous design, but this needs to be fixed immediately
                new_ex_dp_res = ex_dp
                #raise e
            elif e.__class__.__name__ in exception_names:
                print("Exception: " + e.__class__.__name__)
                new_ex_dp_res = ex_dp
                move_to_try.set_validity(False)
            else:
                raise e


        return new_ex_dp_res, move_to_try, total_transformation_cnt

    # ------------------------------
    # Functionality:
    #   Select a item given a probability distribution.
    #   The input provides a list of values and their probabilities/fitness,...
    #   and this function randomly picks a value based on the fitness/probability, ...
    #   Used for random but prioritized selections (of for example blocks, or kernels)
    # input: item_prob_dict {} (item, probability)
    # ------------------------------
    def pick_from_prob_dict(self, item_prob_dict):
        # now encode this priorities into a encoded histogram (for example with performance
        # encoded as 1 and power as 2 and ...) with frequencies
        if config.DEBUG_FIX: random.seed(0)
        else: time.sleep(.00001), random.seed(datetime.now().microsecond)

        item_encoding = np.arange(0, len(item_prob_dict.keys()))  # encoding the metrics from 1 to ... clusters
        rand_var_dis = list(item_prob_dict.values())  # distribution

        encoded_metric = np.random.choice(item_encoding, p=rand_var_dis)  # cluster (metric) selected
        selected_item = list(item_prob_dict.keys())[encoded_metric]
        return selected_item

    # ------------------------------
    # Functionality:
    #       return all the hardware blocks that have same hardware characteristics (e.g, same type and same work-rate and mappability)
    # ------------------------------
    def find_matching_blocks(self, blocks):
        matched_idx = []
        matching_blocks = []
        for idx, _ in enumerate(blocks):
            if idx in matched_idx:
                continue
            matched_idx.append(idx)
            for idx_2 in range(idx+1, len(blocks)):
                if blocks[idx].get_generic_instance_name() == blocks[idx_2].get_generic_instance_name():  # for PEs
                    matching_blocks.append((blocks[idx], blocks[idx_2]))
                    matched_idx.append(idx_2)
                elif blocks[idx].type in ["mem", "ic"] and blocks[idx_2].type in ["mem", "ic"]:  # for mem and ic
                    if blocks[idx].subtype == blocks[idx_2].subtype:
                        matching_blocks.append((blocks[idx], blocks[idx_2]))
                        matched_idx.append(idx_2)
        return matching_blocks

    def get_task_parallelism_type(self, sim_dp, task, parallel_task):
        workload_tasks = sim_dp.database.db_input.workload_tasks
        task_s_workload = sim_dp.database.db_input.task_workload[task]
        parallel_task_s_workload = sim_dp.database.db_input.task_workload[parallel_task]
        if task_s_workload == parallel_task_s_workload:
            return "task_level_parallelism"
        else:
            return "workload_level_parallelism"

    # check if there is another task (on the block that can run in parallel with the task of interest
    def check_if_task_can_run_with_any_other_task_in_parallel(self, sim_dp, task, block):
        parallelism_type = []
        if task.get_name() in ["souurce", "siink", "dummy_last"]:
            return False, parallelism_type
        if block.type == "pe":
            task_dir = "loop_back"
        else:
            task_dir = "write"
        tasks_of_block = [task_ for task_ in block.get_tasks_of_block_by_dir(task_dir) if
                          (not ("souurce" in task_.name) or not ("siink" in task_.name))]
        if config.parallelism_analysis == "static":
            for task_ in tasks_of_block:
                if sim_dp.get_dp_rep().get_hardware_graph().get_task_graph().tasks_can_run_in_parallel(task_, task):
                    return True, _
        elif config.parallelism_analysis == "dynamic":
            parallel_tasks_names_ = sim_dp.get_dp_rep().get_tasks_parallel_task_dynamically(task)
            tasks_using_diff_pipe_cluster = sim_dp.get_dp_rep().get_tasks_using_the_different_pipe_cluster(task, block)
            parallel_tasks_names= list(set(parallel_tasks_names_) - set(tasks_using_diff_pipe_cluster))
            for task_ in tasks_of_block:
                if task_.get_name() in  parallel_tasks_names:
                    parallelism_type.append(self.get_task_parallelism_type(sim_dp, task.get_name(), task_.get_name()))
            if len(parallelism_type)  > 0:
                return True, list(set(parallelism_type))
        return False, parallelism_type


    # ------------------------------
    # Functionality:
    #       check if there are any tasks across two blocks that can be run in parallel
    #       this is used for cleaning up, if there is not opportunities for parallelization
    # Variables:
    #        sim_dp: design
    # ------------------------------
    def check_if_any_tasks_on_two_blocks_parallel(self, sim_dp, block_1, block_2):
        tasks_of_block_1 = [task for task in block_1.get_tasks_of_block_by_dir("write") if not task.is_task_dummy()]
        tasks_of_block_2 = [task for task in block_2.get_tasks_of_block_by_dir("write") if not task.is_task_dummy()]

        for idx_1, _ in enumerate(tasks_of_block_1):
            tsk_1 = tasks_of_block_1[idx_1]
            parallel_tasks_names_ = sim_dp.get_dp_rep().get_tasks_parallel_task_dynamically(tsk_1)
            tasks_using_diff_pipe_cluster = sim_dp.get_dp_rep().get_tasks_using_the_different_pipe_cluster(tsk_1, block_1)
            parallel_tasks_names = list(set(parallel_tasks_names_) - set(tasks_using_diff_pipe_cluster))


            for idx_2, _ in enumerate(tasks_of_block_2):
                tsk_2  = tasks_of_block_2[idx_2]
                if tsk_1.get_name() == tsk_2.get_name():
                    continue
                if config.parallelism_analysis == "static":
                    if sim_dp.get_dp_rep().get_hardware_graph().get_task_graph().tasks_can_run_in_parallel(tsk_1, tsk_2):
                        return True
                elif config.parallelism_analysis == "dynamic":
                    if tsk_2.get_name() in parallel_tasks_names:
                        return True
        return False

    # ------------------------------
    # Functionality:
    #       Return all the blocks that are unnecessarily parallelized, i.e., there are no
    #       tasks across them that can run in parallel.
    #       this is used for cleaning up, if there is not opportunities for parallelization
    # Variables:
    #        sim_dp: design
    #        matching_blocks_list: list of hardware blocks with equivalent characteristics (e.g., two A53, or two
    #        identical acclerators)
    # ------------------------------
    def find_blocks_with_all_serial_tasks(self, sim_dp, matching_blocks_list):
        matching_blocks_list_filtered = []
        for matching_blocks in matching_blocks_list:
            if self.check_if_any_tasks_on_two_blocks_parallel(sim_dp, matching_blocks[0], matching_blocks[1]):
                continue
            matching_blocks_list_filtered.append((matching_blocks[0], matching_blocks[1]))

        return matching_blocks_list_filtered

    # ------------------------------
    # Functionality:
    #      search through all the blocks and return a pair of blocks that cleanup can  apply to
    # Variables:
    #        sim_dp: design
    # ------------------------------
    def pick_block_pair_to_clean_up(self, sim_dp, block_pairs):
        if len(block_pairs) == 0:
            return block_pairs

        cleanup_ease_list = []
        block_pairs_sorted = []  # sorting the pairs elements (within each pair) based on the number of tasks on each
        for blck_1, blck_2 in block_pairs:
            if blck_2.type == "ic":  # for now ignore ics
                continue
            elif blck_2.type == "mem":
                if self.database.check_superiority(blck_1, blck_2):
                    block_pairs_sorted.append((blck_1, blck_2))
                else:
                    block_pairs_sorted.append((blck_2, blck_1))
            else:
                if len(blck_1.get_tasks_of_block()) < len(blck_2.get_tasks_of_block()):
                    block_pairs_sorted.append((blck_1, blck_2))
                else:
                    block_pairs_sorted.append((blck_2, blck_1))

            distance = len(sim_dp.get_dp_rep().get_hardware_graph().get_path_between_two_vertecies(blck_1, blck_2))
            num_tasks_to_move = min(len(blck_1.get_tasks_of_block()), len(blck_2.get_tasks_of_block()))

            cleanup_ease_list.append(distance + num_tasks_to_move)

        # when we need to clean up the ics, ignore for now
        if len(cleanup_ease_list) == 0:
            return []

        picked_easiest = False
        min_ease = 100000000
        for idx, ease in enumerate(cleanup_ease_list):
            if ease < min_ease:
                picked_easiest = True
                easiest_pair = block_pairs_sorted[idx]
                min_ease = ease
        return easiest_pair

    # ------------------------------
    # Functionality:
    #      used to determine if two different task can use the same accelerators.
    # ------------------------------
    def are_same_ip_tasks(self, task_1, task_2):
        return (task_1.name, task_2.name) in self.database.db_input.misc_data["same_ip_tasks_list"] or (task_2.name, task_1.name) in self.database.db_input.misc_data["same_ip_tasks_list"]

    # ------------------------------
    # Functionality:
    #     find all the tasks that can run on the same ip (accelerator)
    # Variables:
    #      des_tup: (design, simulated design)
    # ------------------------------
    def find_task_with_similar_mappable_ips(self, des_tup):
        ex_dp, sim_dp = des_tup
        #krnls = sim_dp.get_dp_stats().get_kernels()
        blcks = ex_dp.get_blocks()
        pe_blocks = [blck for blck in blcks if blck.type=="pe"]
        tasks_sub_ip_type = [] # (task, sub_ip)
        matches = []
        for blck in pe_blocks:
            tasks_sub_ip_type.extend(zip(blck.get_tasks_of_block(), [blck.subtype]*len(blck.get_tasks_of_block())))

        for task, sub_ip_type in tasks_sub_ip_type:
            check_for_similarity = False
            if sub_ip_type == "ip":
                check_for_similarity = True
            if not check_for_similarity:
                continue

            for task_2, sub_ip_type_2 in tasks_sub_ip_type:
                if task_2.name == task.name :
                    continue
                if self.are_same_ip_tasks(task, task_2):
                    for blck in pe_blocks:
                        if task_2 in blck.get_tasks_of_block():
                            block_to_migrate_from = blck
                        if task in blck.get_tasks_of_block():
                            block_to_migrate_to = blck

                    if not (block_to_migrate_to == block_to_migrate_from):
                        matches.append((task_2, block_to_migrate_from, task, block_to_migrate_to))

        if len(matches) == 0:
            return None, None, None, None
        else:
           return random.choice(matches)

    # ------------------------------
    # Functionality:
    #     pick a block pair to apply cleaning to
    # Variables:
    #      des_tup: (design, simulated design)
    # ------------------------------
    def gen_block_match_cleanup_move(self, des_tup):
        ex_dp, sim_dp = des_tup
        krnls = sim_dp.get_dp_stats().get_kernels()
        blcks = ex_dp.get_blocks()

        # move tasks to already generated IPs
        # clean up the matching blocks
        matching_blocks_list = self.find_matching_blocks(blcks)
        matching_blocks_lists_filtered = self.find_blocks_with_all_serial_tasks(sim_dp, matching_blocks_list)
        return self.pick_block_pair_to_clean_up(sim_dp, matching_blocks_lists_filtered)

    # ------------------------------
    # Functionality:
    #     is current iteration a clean up iteration (ie., should be used for clean up)
    # ------------------------------
    def is_cleanup_iter(self):
        result = (self.cleanup_ctr % (config.cleaning_threshold)) >= (config.cleaning_threshold- config.cleaning_consecutive_iterations)
        return result

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

    def select_block_to_migrate_to(self, ex_dp, sim_dp, hot_blck_synced, selected_metric, sorted_metric_dir, selected_krnl):
        # get initial information
        locality_type = []
        parallelism_type =[]
        task = ex_dp.get_hardware_graph().get_task_graph().get_task_by_name(selected_krnl.get_task_name())
        selected_metric = list(sorted_metric_dir.keys())[-1]
        selected_dir = sorted_metric_dir[selected_metric]
        # find blocks equal or immeidately better
        equal_imm_blocks_present_for_migration = self.dh.get_equal_immediate_blocks_present(ex_dp, hot_blck_synced,
                                                                selected_metric, selected_dir, [task])


        # does parallelism exist in the current occupying block
        current_block_parallelism_exist, parallelism_type = self.check_if_task_can_run_with_any_other_task_in_parallel(sim_dp,
                                                                                                     task,
                                                                                                     hot_blck_synced)
        inequality_dir = selected_dir*-1
        results_block = [] # results

        task_s_blocks = ex_dp.get_hardware_graph().get_blocks_of_task(task)
        if len(task_s_blocks) == 0:
            print("a task must have at lease three blocks")
            exit(0)


        remove_list = []  # list of blocks to remove from equal_imm_blocks_present_for_migration
        # improve locality by only allowing migration to the PE/MEM close by
        if hot_blck_synced.type == "mem":
            # only keep memories that are connected to the IC neighbour of the task's pe
            # This is to make sure that we keep data local (to the router), instead of migrating to somewhere far
            task_s_pe = [blk for blk in task_s_blocks if blk.type == "pe"][0] # get task's pe
            tasks_s_ic = [el for el in task_s_pe.get_neighs() if el.type == "ic"][0] # get pe's ic
            potential_mems = [el for el in tasks_s_ic.get_neighs() if el.type == "mem"] # get ic's memories
            for el in equal_imm_blocks_present_for_migration:
                if el not in potential_mems:
                    remove_list.append(el)
                    locality_type = ["spatial_locality"]
            for el in remove_list:
                equal_imm_blocks_present_for_migration.remove(el)
        elif hot_blck_synced.type == "pe":
            # only keep memories that are connected to the IC neighbour of the task's pe
            # This is to make sure that we keep data local (to the router), instead of migrating to somewhere far
            task_s_mems = [blk for blk in task_s_blocks if blk.type == "mem"] # get task's pe
            potential_pes = []
            for task_s_mem in task_s_mems:
                tasks_s_ic = [el for el in task_s_mem.get_neighs() if el.type == "ic"][0]  # get pe's ic
                potential_pes.extend([el for el in tasks_s_ic.get_neighs() if el.type == "pe"])  # get ic's memories
            for el in equal_imm_blocks_present_for_migration:
                if el not in potential_pes:
                    remove_list.append(el)
                    locality_type = ["spatial_locality"]
            for el in remove_list:
                equal_imm_blocks_present_for_migration.remove(el)

        # iterate through the blocks and find the best one
        for block_to_migrate_to in equal_imm_blocks_present_for_migration:
            # skip yourself
            if block_to_migrate_to == hot_blck_synced:
                continue

            block_metric_attr = self.get_block_attr(selected_metric) # metric to pay attention to
            # iterate and found blocks that are at least as good as the current block
            if getattr(block_to_migrate_to, block_metric_attr) == getattr(hot_blck_synced, block_metric_attr):
                # blocks have similar attr value
                if (selected_metric == "power" and selected_dir == -1)  or \
                    (selected_metric == "latency" and selected_dir == 1) or (selected_metric == "area"):
                    # if we want to slow down (reduce latency, improve power), look for parallel task on the other block
                    block_to_mig_to_parallelism_exist, parallelism_type = self.check_if_task_can_run_with_any_other_task_in_parallel(sim_dp,
                                                                                                               task,
                                                                                                               block_to_migrate_to)
                    if (selected_metric == "area" and selected_dir == -1):
                        # no parallelism possibly allows for theo the other memory to shrink
                        if not block_to_mig_to_parallelism_exist:
                            results_block.append(block_to_migrate_to)
                            parallelism_type = ["serialism"]
                    else:
                        if block_to_mig_to_parallelism_exist:
                            results_block.append(block_to_migrate_to)
                            parallelism_type = ["serialism"]
                else:
                    # if we want to accelerate (improve latency, get more power), look for parallel task on the same block
                    if current_block_parallelism_exist:
                        results_block.append(block_to_migrate_to)
            elif inequality_dir*getattr(block_to_migrate_to, block_metric_attr) > inequality_dir*getattr(hot_blck_synced, block_metric_attr):
                results_block.append(block_to_migrate_to)
                break

        # if no block found, just load the results_block with current block
        if len(results_block) == 0:
            results_block = [hot_blck_synced]
            found_block_to_mig_to = False
        else:
            found_block_to_mig_to = True

        # pick at random to try random scenarios. At the moment, only equal and immeidately better blocks are considered
        random.seed(datetime.now().microsecond)
        result_block = random.choice(results_block)

        selection_mode = "batch"
        if found_block_to_mig_to:
            if getattr(result_block, block_metric_attr) == getattr(hot_blck_synced, block_metric_attr):
                selection_mode = "batch"
            else:
                selection_mode = "single"


        return result_block, found_block_to_mig_to, selection_mode, parallelism_type, locality_type


    def is_system_ic(self, ex_dp, sim_dp, blck):
        if not sim_dp.dp_stats.fits_budget(1):
            return False
        elif sim_dp.dp_stats.fits_budget(1) and not self.dram_feasibility_check_pass(ex_dp):
            return False
        else:
            for block in ex_dp.get_hardware_graph().get_blocks():
                neighs = block.get_neighs()
                if any(el for el in neighs if el.subtype == "dram"):
                    if block == blck:
                        return True
        return False

    def bus_has_pe_mem_topology_for_split(self, ex_dp, sim_dp, ref_task, block):
        if not block.type == "ic" or ref_task.is_task_dummy():
            return False
        found_pe_block = False
        found_mem_block = False

        migrant_tasks  = self.dh.find_parallel_tasks_of_task_in_block(ex_dp, sim_dp, ref_task, block)[0]
        migrant_tasks_names = [el.get_name() for el in migrant_tasks]
        mem_neighs = [el for el in block.get_neighs() if el.type == "mem"]
        pe_neighs = [el for el in block.get_neighs() if el.type == "pe"]

        for neigh in  pe_neighs:
            neigh_tasks = [el.get_name() for el in neigh.get_tasks_of_block_by_dir("loop_back")]
            # if no overlap skip
            if len(list(set(migrant_tasks_names) - set(neigh_tasks) )) == len(migrant_tasks_names):
                continue
            else:
                found_pe_block = True
                break

        for neigh in  mem_neighs:
            neigh_tasks = [el.get_name() for el in neigh.get_tasks_of_block_by_dir("write")]
            # if no overlap skip
            if len(list(set(migrant_tasks_names) - set(neigh_tasks) )) == len(migrant_tasks_names):
                continue
            else:
                found_mem_block = True
                break


        if found_pe_block and found_mem_block :
            return True
        else:
            return False

    def get_feasible_transformations(self, ex_dp, sim_dp, hot_blck_synced, selected_metric, selected_krnl, sorted_metric_dir):

        # if this knob is set, we randomly pick a transformation
        # THis is to illustrate the architectural awareness of FARSI a
        if config.transformation_selection_mode == "random":
            all_transformations = config.all_available_transformations
            return all_transformations

        # pick a transformation smartly
        imm_block = self.dh.get_immediate_block_multi_metric(hot_blck_synced, selected_metric, sorted_metric_dir,  hot_blck_synced.get_tasks_of_block())
        task = ex_dp.get_hardware_graph().get_task_graph().get_task_by_name(selected_krnl.get_task_name())
        feasible_transformations = set(config.metric_trans_dict[selected_metric])

        # find the block that is at least as good as the block (for migration)
        # if can't find any, we return the same block
        selected_metric = list(sorted_metric_dir.keys())[-1]
        selected_dir = sorted_metric_dir[selected_metric]

        equal_imm_block_present_for_migration, found_blck_to_mig_to, selection_mode, parallelism_type, locality_type = self.select_block_to_migrate_to(ex_dp, sim_dp, hot_blck_synced,
                                                                selected_metric, sorted_metric_dir, selected_krnl)

        hot_block_type = hot_blck_synced.type
        hot_block_subtype = hot_blck_synced.subtype

        parallelism_exist, parallelism_type = self.check_if_task_can_run_with_any_other_task_in_parallel(sim_dp, task, hot_blck_synced)
        other_block_parallelism_exist = False
        all_transformations = config.metric_trans_dict[selected_metric]
        can_improve_locality = self.can_improve_locality(ex_dp, hot_blck_synced, task)
        can_improve_routing = self.can_improve_routing(ex_dp, sim_dp, hot_blck_synced, task)

        bus_has_pe_mem_topology_for_split = self.bus_has_pe_mem_topology_for_split(ex_dp, sim_dp, task,hot_blck_synced)
        # ------------------------
        # based on parallelism, generate feasible transformations
        # ------------------------
        if parallelism_exist:
           if selected_metric == "latency":
               if selected_dir == -1:
                    if hot_block_type == "pe":
                        feasible_transformations = ["migrate", "split"]  # only for PE since we wont to be low cost, for IC/MEM cost does not increase if you customize
                    else:
                        if hot_block_type == "ic":
                            mem_neighs = [el for el in hot_blck_synced.get_neighs() if el.type == "mem"]
                            pe_neighs = [el for el in hot_blck_synced.get_neighs() if el.type == "pe"]
                            if len(mem_neighs) <= 1 or len(pe_neighs)  <= 1 or not bus_has_pe_mem_topology_for_split:
                                feasible_transformations = ["swap", "split_swap"]  # ", "swap", "split_swap"]
                            else:
                                feasible_transformations = ["migrate", "split"]  # ", "swap", "split_swap"]
                        else:
                            feasible_transformations = ["migrate", "split"] #", "swap", "split_swap"]
               else:
                    # we can do better by comparing the advantage disadvantage of migrating
                    # (Advantage: slowing down by serialization, and disadvantage: accelerating by parallelization)
                   feasible_transformations = ["swap"]
           if selected_metric == "power":
               if selected_dir == -1:
                   # we can do better by comparing the advantage disadvantage of migrating
                   # (Advantage: slowing down by serialization, and disadvantage: accelerating by parallelization)
                   feasible_transformations = ["swap", "split_swap"]
               else:
                   feasible_transformations = all_transformations
           if selected_metric == "area":
               if selected_dir == -1:
                   if hot_block_subtype == "pe":
                       feasible_transformations = ["migrate", "swap"]
                   else:
                       feasible_transformations = ["migrate", "swap", "split_swap"]
               else:
                   feasible_transformations = all_transformations
        elif not parallelism_exist:
           if selected_metric == "latency":
               if selected_dir == -1:
                   feasible_transformations = ["swap", "split_swap"]
               else:
                   feasible_transformations = ["swap", "migrate"]
           if selected_metric == "power":
               if selected_dir == -1:
                   feasible_transformations = ["migrate", "swap", "split_swap"]
           if selected_metric == "area":
               if selected_dir == -1:
                    feasible_transformations = ["migrate", "swap","split_swap"]
               else:
                   feasible_transformations = ["migrate", "swap", "split"]

        # ------------------------
        # based on locality, generate feasible transformations
        # ------------------------
        if can_improve_locality and ('transfer' in config.all_available_transformations):
            # locality not gonna improve area with the current set up
            if not selected_metric == "area" and selected_dir == -1:
                feasible_transformations.append("transfer")

        #------------------------
        # there is a on opportunity for routing
        #------------------------
        if can_improve_routing and ('routing' in config.all_available_transformations):
            transformation_list = list(feasible_transformations)
            transformation_list.append('routing')
            feasible_transformations = set(transformation_list)


        #------------------------
        # post processing of the destination blocks to eliminate transformations
        #------------------------
        # filter migrate
        if not found_blck_to_mig_to:
            # if can't find a block that is at least as good as the current block, can't migrate
            feasible_transformations =  set(list(set(feasible_transformations) - set(['migrate'])))

        # filter split
        number_of_task_on_block = 0
        if hot_blck_synced.type == "pe":
            number_of_task_on_block = len(hot_blck_synced.get_tasks_of_block())
        else:
            number_of_task_on_block = len(hot_blck_synced.get_tasks_of_block_by_dir("write"))
        if number_of_task_on_block == 1:  # can't split an accelerator
            feasible_transformations =  set(list(set(feasible_transformations) - set(['split', 'split_swap'] )))

        # filter swap
        block_metric_attr = self.get_block_attr(selected_metric)  # metric to pay attention to
        if getattr(imm_block, block_metric_attr) == getattr(hot_blck_synced, block_metric_attr):
            #if imm_block.get_generic_instance_name() == hot_blck_synced.get_generic_instance_name():
            # if can't swap improve, get rid of swap
            feasible_transformations = set(list(set(feasible_transformations) - set(['swap'])))

        # for IC's we don't use migrate
        if hot_blck_synced.type in ["ic"]:
            # we don't cover migrate for ICs at the moment
            # TODO: add this feature later
            feasible_transformations = set(list(set(feasible_transformations) - set(['migrate', 'split_swap'])))

        # if no valid transformation left, issue the identity transformation (where nothing changes and a simple copying is done)
        if len(list(set(feasible_transformations))) == 0:
            feasible_transformations = ["identity"]


        return feasible_transformations

    def set_design_space_size(self,  ex_dp, sim_dp):
        # if this knob is set, we randomly pick a transformation
        # THis is to illustrate the architectural awareness of FARSI a

        buses = [el for el in ex_dp.get_blocks() if el.type == "ic"]
        mems = [el for el in ex_dp.get_blocks() if el.type == "mem"]
        srams = [el for el in ex_dp.get_blocks() if el.type == "sram"]
        drams = [el for el in ex_dp.get_blocks() if el.type == "dram"]
        pes = [el for el in ex_dp.get_blocks() if el.type == "pe"]
        ips = [el for el in ex_dp.get_blocks() if el.subtype == "ip"]
        gpps = [el for el in ex_dp.get_blocks() if el.subtype == "gpp"]
        all_blocks = ex_dp.get_blocks()

        # per block
        # for PEs
        for pe in gpps:
            number_of_task_on_block = len(pe.get_tasks_of_block())
            #sim_dp.neighbouring_design_space_size["hardening"] += number_of_task_on_block + 1# +1 for swap, the rest is for split_swap
            sim_dp.neighbouring_design_space_size += number_of_task_on_block + 1  # +1 for swap, the rest is for split_swap
        for pe in ips:
            #sim_dp.neighbouring_design_space_size["softening"] += 1
            sim_dp.neighbouring_design_space_size += 1

        # for all
        for blck in all_blocks:
            for mode in ["frequency_modulation", "bus_width_modulation", "loop_iteration_modulation"]:
                if not blck.type =="pe":
                    if mode == "loop_iteration_modulation":
                        continue
                #sim_dp.neighbouring_design_space_size[mode] += 2  # going up or down
                sim_dp.neighbouring_design_space_size += 2  # going up or down

        for blck in all_blocks:
            for mode in ["allocation"]:
                if blck.type == "ic":
                    continue
                number_of_task_on_block = len(blck.get_tasks_of_block())
                #sim_dp.neighbouring_design_space_size[mode] += number_of_task_on_block + 1  # +1 is for split, the rest os for split_swap
                sim_dp.neighbouring_design_space_size += number_of_task_on_block + 1  # +1 is for split, the rest os for split_swap

        for blck in all_blocks:
            equal_imm_blocks_present_for_migration = self.dh.get_equal_immediate_blocks_present(ex_dp,
                                                                                                blck,
                                                                                                "latency",
                                                                                                -1,
                                                                                                    blck.get_tasks_of_block())

            equal_imm_blocks_present_for_migration.extend(self.dh.get_equal_immediate_blocks_present(ex_dp,
                                                                                                blck,
                                                                                                "latency",
                                                                                                +1,
                                                                                                    blck.get_tasks_of_block()))

            """ 
            imm_blocks_present_for_migration.extend([self.dh.get_immediate_block(
                                                                                                     blck,
                                                                                                     "latency",
                                                                                                     -1,
                                                                                                     blck.get_tasks_of_block())])
            """
            #other_blocks_to_map_to_lengths = len(equal_imm_blocks_present_for_migration) - len(imm_blocks_present_for_migration)  # subtract to avoid double counting
            other_blocks_to_map_to_lengths = 0
            for el in equal_imm_blocks_present_for_migration:
                if el == blck:
                    continue
                elif not el.type == blck.type:
                    continue
                else:
                    other_blocks_to_map_to_lengths +=1

            #other_blocks_to_map_to_lengths = len(equal_imm_blocks_present_for_migration)
            #sim_dp.neighbouring_design_space_size[blck.type+"_"+"mapping"] += len(blck.get_tasks_of_block())*other_blocks_to_map_to_lengths
            sim_dp.neighbouring_design_space_size += len(blck.get_tasks_of_block())*other_blocks_to_map_to_lengths


    def get_transformation_design_space_size(self, move_to_apply, ex_dp, sim_dp, block_of_interest, selected_metric, sorted_metric_dir):
        # if this knob is set, we randomly pick a transformation
        # THis is to illustrate the architectural awareness of FARSI a
        imm_block = self.dh.get_immediate_block_multi_metric(block_of_interest, selected_metric, sorted_metric_dir,  block_of_interest.get_tasks_of_block())

        task = (block_of_interest.get_tasks_of_block())[0] # any task for do
        feasible_transformations = set(config.metric_trans_dict[selected_metric])

        # find the block that is at least as good as the block (for migration)
        # if can't find any, we return the same block
        selected_metric = list(sorted_metric_dir.keys())[-1]
        selected_dir = sorted_metric_dir[selected_metric]

        equal_imm_blocks_present_for_migration = self.dh.get_equal_immediate_blocks_present(ex_dp, block_of_interest,
                                                                selected_metric, selected_dir, [task])
        if len(equal_imm_blocks_present_for_migration)  == 1 and equal_imm_blocks_present_for_migration[0] == block_of_interest:
            equal_imm_blocks_present_for_migration = []

        buses = [el for el in ex_dp.get_blocks() if el.type == "ic"]
        mems = [el for el in ex_dp.get_blocks() if el.type == "mem"]
        srams = [el for el in ex_dp.get_blocks() if el.type == "sram"]
        drams = [el for el in ex_dp.get_blocks() if el.type == "dram"]
        pes = [el for el in ex_dp.get_blocks() if el.type == "pe"]
        ips = [el for el in ex_dp.get_blocks() if el.subtype == "ip"]
        gpps = [el for el in ex_dp.get_blocks() if el.subtype == "gpp"]

        # per block
        # for PEs
        if block_of_interest.subtype == "gpp":
            number_of_task_on_block = len(block_of_interest.get_tasks_of_block())
            move_to_apply.design_space_size["hardening"] += number_of_task_on_block + 1# +1 for swap, the rest is for split_swap
            move_to_apply.design_space_size["pe_allocation"] += (number_of_task_on_block + 1) # +1 is for split, the rest os for split_swap
        elif block_of_interest.subtype == "ip":
            move_to_apply.design_space_size["softening"] += 1

        # for all
        for mode in ["frequency_modulation", "bus_width_modulation", "loop_iteration_modulation", "allocation"]:
            if not block_of_interest.type =="pe":
                if mode == "loop_iteration_modulation":
                    continue
            value = self.dh.get_all_compatible_blocks_of_certain_char(ex_dp, block_of_interest,
                                                                selected_metric, selected_dir, [task], mode)
            if mode in ["bus_width_modulation","loop_iteration_modulation"]:
                move_to_apply.design_space_size[mode] += len(value)
            else:
                move_to_apply.design_space_size[block_of_interest.type + "_"+ mode] += len(value)


        for block_type in ["pe", "mem", "ic"]:
            if block_type == block_of_interest.type:
                move_to_apply.design_space_size[block_type +"_"+"mapping"] += (len(equal_imm_blocks_present_for_migration) - 1)
            else:
                move_to_apply.design_space_size[block_type +"_"+"mapping"] += 0

        can_improve_routing = self.can_improve_routing(ex_dp, sim_dp, block_of_interest, task)
        if can_improve_routing:
            move_to_apply.design_space_size["routing"] += (len(buses) - 1)
        move_to_apply.design_space_size["transfer"] += (len(buses)-1)
        move_to_apply.design_space_size["identity"] +=  1

    #    pick which transformation to apply
    # Variables:
    #      hot_blck_synced: the block bottleneck
    #      selected_metric: metric to focus on
    #      selected_krnl: the kernel to focus on
    # ------------------------------
    def select_transformation(self, ex_dp, sim_dp, hot_blck_synced, selected_metric, selected_krnl, sorted_metric_dir):
        feasible_transformations = self.get_feasible_transformations(ex_dp, sim_dp, hot_blck_synced, selected_metric,
                                                                    selected_krnl, sorted_metric_dir)
        if config.print_info_regularly:
            print(list(feasible_transformations))
        random.seed(datetime.now().microsecond)
        # pick randomly at the moment.
        # TODO: possibly can do better
        transformation = random.choice(list(feasible_transformations))

        #if len(hot_blck_synced.get_tasks_of_block_by_dir("write")) > 1:
        #    transformation = "split_swap"
        #else:
        #    transformation = "swap"
        if transformation == "migrate":
            batch_mode = "single"
            transformation_sub_name = "irrelevant"
        elif transformation == "split":
            # see if any task can run in parallel
            batch_mode = "batch"
            transformation_sub_name = "irrelevant"
        elif transformation == "split_swap":
            batch_mode = "single"
            transformation_sub_name = "irrelevant"
        elif transformation == "transfer":
            batch_mode = "irrelevant"
            transformation_sub_name = "locality_improvement"
        elif transformation == "routing":
            batch_mode = "irrelevant"
            transformation_sub_name = "routing_improvement"
        else:
            transformation_sub_name = "irrelevant"
            batch_mode = "irrelevant"


        return transformation, transformation_sub_name, batch_mode, len(list(feasible_transformations))

    # calculate the cost impact of a kernel improvement
    def get_swap_improvement_cost(self, sim_dp, kernels, selected_metric, dir):
        def get_subtype_for_cost(block):
            if block.type == "pe" and block.subtype == "ip":
                return "ip"
            if block.type == "pe" and block.subtype == "gpp":
                if "A53" in block.instance_name or "ARM" in block.instance_name:
                    return "arm"
                if "G3" in block.instance_name:
                    return "dsp"
            else:
                return block.type

        # Figure out whether there is a mapping that improves kernels performance
        def no_swap_improvement_possible(sim_dp, selected_metric, metric_dir, krnl):
            hot_block = sim_dp.get_dp_stats().get_hot_block_of_krnel(krnl.get_task_name(), selected_metric)
            imm_block = self.dh.get_immediate_block_multi_metric(hot_block, metric_dir, [krnl.get_task()])
            blah  = hot_block.get_generic_instance_name()
            blah2  = imm_block.get_generic_instance_name()
            return hot_block.get_generic_instance_name() == imm_block.get_generic_instance_name()


        # find the cost of improvement by comparing the current and accelerated design (for the kernel)
        kernel_improvement_cost = {}
        kernel_name_improvement_cost = {}
        for krnel in kernels:
            hot_block = sim_dp.get_dp_stats().get_hot_block_of_krnel(krnel.get_task_name(), selected_metric)
            hot_block_subtype = get_subtype_for_cost(hot_block)
            current_cost = self.database.db_input.porting_effort[hot_block_subtype]
            #if hot_block_subtype == "ip":
            #    print("what")
            imm_block = self.dh.get_immediate_block_multi_metric(hot_block,selected_metric, metric_dir,[krnel.get_task()])
            imm_block_subtype = get_subtype_for_cost(imm_block)
            imm_block_cost =  self.database.db_input.porting_effort[imm_block_subtype]
            improvement_cost = (imm_block_cost - current_cost)
            kernel_improvement_cost[krnel] = improvement_cost

        # calcualte inverse so lower means worse
        max_val =  max(kernel_improvement_cost.values()) # multiply by
        kernel_improvement_cost_inverse = {}
        for k, v in kernel_improvement_cost.items():
            kernel_improvement_cost_inverse[k] = max_val - kernel_improvement_cost[k]

        # get sum and normalize
        sum_ = sum(list(kernel_improvement_cost_inverse.values()))
        for k, v in kernel_improvement_cost_inverse.items():
            # normalize
            if not (sum_ == 0):
                kernel_improvement_cost_inverse[k] = kernel_improvement_cost_inverse[k]/sum_
            kernel_improvement_cost_inverse[k] = max(kernel_improvement_cost_inverse[k], .0000001)
            if no_swap_improvement_possible(sim_dp, selected_metric, dir, k):
                kernel_improvement_cost_inverse[k] = .0000001
            kernel_name_improvement_cost[k.get_task_name()] = kernel_improvement_cost_inverse[k]

        return kernel_improvement_cost_inverse

    def get_identity_cost(self):
        return self.database.db_input.porting_effort["ip"]


    # calculate the cost impact of a kernel improvement
    def get_swap_cost(self, sim_dp, krnl, selected_metric, sorted_metric_dir):
        def get_subtype_for_cost(block):
            if block.type == "pe" and block.subtype == "ip":
                return "ip"
            if block.type == "pe" and block.subtype == "gpp":
                if "A53" in block.instance_name or "ARM" in block.instance_name:
                    return "arm"
                if "G3" in block.instance_name:
                    return "dsp"
            else:
                return block.type

        hot_block = sim_dp.get_dp_stats().get_hot_block_of_krnel(krnl.get_task_name(), selected_metric)
        hot_block_subtype = get_subtype_for_cost(hot_block)
        current_cost = self.database.db_input.porting_effort[hot_block_subtype]
        imm_block = self.dh.get_immediate_block_multi_metric(hot_block,selected_metric, sorted_metric_dir,[krnl.get_task()])
        imm_block_subtype = get_subtype_for_cost(imm_block)
        imm_block_cost =  self.database.db_input.porting_effort[imm_block_subtype]
        improvement_cost = (imm_block_cost - current_cost)
        return improvement_cost

    def get_migrate_cost(self):
        return 0

    def get_transfer_cost(self):
        return 0

    def get_routing_cost(self):
        return 0

    def get_split_cost(self):
        return 1

    def get_migration_split_cost(self, transformation):
        if transformation == "migrate":
            return self.get_migrate_cost()
        elif transformation == "split":
            return self.get_split_cost()
        else:
            print("this transformation" + transformation + " is not supported for cost calculation")
            exit(0)

    # how much does it cost to improve the kernel for different transformations
    def get_krnl_improvement_cost(self, ex_dp, sim_dp, krnls, selected_metric, move_sorted_metric_dir):
        # whether you can apply the transformation for the krnel's block
        def get_transformation_cost(sim_dp, selected_metric, move_sorted_metric_dir, krnl, transformation):
            if transformation == "swap":
                cost = self.get_swap_cost(sim_dp, krnl, selected_metric, move_sorted_metric_dir)
            elif transformation in ["split", "migrate"]:
                cost = self.get_migration_split_cost(transformation)
            elif transformation in ["split_swap"]:
                cost = self.get_migration_split_cost("split")
                cost += self.get_swap_cost(sim_dp, krnl, selected_metric, move_sorted_metric_dir)
            elif transformation in ["identity"]:
                cost = self.get_identity_cost()
            elif transformation in ["transfer"]:
                cost = self.get_transfer_cost()
            elif transformation in ["routing"]:
                cost = self.get_routing_cost()
            if cost == 0:
                cost = self.min_cost_to_consider
            return cost

        krnl_improvement_cost = {}

        # iterate through the kernels, find their feasible transformations and
        # find cost
        for krnl in krnls:
            hot_block = sim_dp.get_dp_stats().get_hot_block_of_krnel(krnl.get_task_name(), selected_metric)
            imm_block = self.dh.get_immediate_block_multi_metric(hot_block, selected_metric, move_sorted_metric_dir, [krnl.get_task()])
            hot_blck_synced = self.dh.find_cores_hot_kernel_blck_bottlneck(ex_dp, hot_block)
            feasible_trans = self.get_feasible_transformations(ex_dp, sim_dp, hot_blck_synced, selected_metric,
                                              krnl,move_sorted_metric_dir)
            for trans in feasible_trans:
                cost = get_transformation_cost(sim_dp, selected_metric, move_sorted_metric_dir, krnl, trans)
                krnl_improvement_cost[(krnl, trans)] = cost
        return krnl_improvement_cost

    # select a metric to improve on
    def select_metric(self, sim_dp):
        # prioritize metrics based on their distance contribution to goal
        metric_prob_dict = {}  # (metric:priority value) each value is in [0 ,1] interval
        for metric in config.budgetted_metrics:
            metric_prob_dict[metric] = sim_dp.dp_stats.dist_to_goal_per_metric(metric, config.metric_sel_dis_mode)/\
                                                   sim_dp.dp_stats.dist_to_goal(["power", "area", "latency"],
                                                                                config.metric_sel_dis_mode)

        # sort the metric based on distance (and whether the sort is probabilistic or exact).
        # probabilistic sorting, first sort exactly, then use the exact value as a probability of selection
        metric_prob_dict_sorted = {k: v for k, v in sorted(metric_prob_dict.items(), key=lambda item: item[1])}
        if config.move_metric_ranking_mode== "exact":
            selected_metric = list(metric_prob_dict_sorted.keys())[len(metric_prob_dict_sorted.keys()) -1]
        else:
            selected_metric = self.pick_from_prob_dict(metric_prob_dict_sorted)

        sorted_low_to_high_metric_dir = {}
        for metric, prob in metric_prob_dict_sorted.items():
            move_dir = 1  # try to increase the metric value
            if not sim_dp.dp_stats.fits_budget_for_metric_for_SOC(metric, 1):
                move_dir = -1  # try to reduce the metric value
            sorted_low_to_high_metric_dir[metric] = move_dir

        # Delete later. for now for validation
        #selected_metric = "latency"
        #sorted_low_to_high_metric_dir=  {'area':1, 'power':-1, 'latency':-1}
        #metric_prob_dict_sorted =  {'area':.1, 'power':.1, 'latency':.8}

        return selected_metric, metric_prob_dict_sorted, sorted_low_to_high_metric_dir

    # select direction for the move
    def select_dir(self, sim_dp, metric):
        move_dir = 1  # try to increase the metric value
        if not sim_dp.dp_stats.fits_budget_for_metric_for_SOC(metric, 1):
            move_dir = -1  # try to reduce the metric value
        return move_dir

    def filter_in_kernels_meeting_budget(self, selected_metric, sim_dp):
        krnls = sim_dp.get_dp_stats().get_kernels()

        # filter the kernels whose workload already met the budget
        workload_tasks = sim_dp.database.db_input.workload_tasks
        task_workload = sim_dp.database.db_input.task_workload
        workloads_to_consider = []
        for workload in workload_tasks.keys():
            if sim_dp.dp_stats.workload_fits_budget(workload, 1):
                continue
            workloads_to_consider.append(workload)

        krnls_to_consider = []
        for krnl in krnls:
            if task_workload[krnl.get_task_name()] in workloads_to_consider and not krnl.get_task().is_task_dummy():
                krnls_to_consider.append(krnl)

        return krnls_to_consider

    # get each kernels_contribution to the metric of interest
    def get_kernels_s_contribution(self, selected_metric, sim_dp):
        krnl_prob_dict = {}  # (kernel, metric_value)


        #krnls = sim_dp.get_dp_stats().get_kernels()
        # filter it kernels whose workload meet the budget
        krnls = self.filter_in_kernels_meeting_budget(selected_metric, sim_dp)
        if krnls == []: # the design meets the budget, hence all kernels can be improved for cost improvement
            krnls = sim_dp.get_dp_stats().get_kernels()

        metric_total = sum([krnl.stats.get_metric(selected_metric) for krnl in krnls])
        # sort kernels based on their contribution to the metric of interest
        for krnl in krnls:
            krnl_prob_dict[krnl] = krnl.stats.get_metric(selected_metric)/metric_total

        if not "bottleneck" in self.move_s_krnel_selection:
            for krnl in krnls:
                krnl_prob_dict[krnl] = 1
        return krnl_prob_dict

    # get each_kernels_improvement_ease (ease = 1/cost)
    def get_kernels_s_improvement_ease(self, ex_dp, sim_dp, selected_metric, move_sorted_metric_dir):
        krnls = sim_dp.get_dp_stats().get_kernels()
        krnl_improvement_ease = {}
        if not "improvement_ease" in self.move_s_krnel_selection:
            for krnl in krnls:
                krnl_improvement_ease[krnl] = 1
        else:
            krnl_trans_improvement_cost = self.get_krnl_improvement_cost(ex_dp, sim_dp, krnls, selected_metric, move_sorted_metric_dir)
            # normalize
            # normalized and reverse (we need to reverse, so higher cost is worse, i.e., smaller)
            krnl_trans_improvement_ease = {}
            for krnl_trans, cost in krnl_trans_improvement_cost.items():
                krnl_trans_improvement_ease[krnl_trans] = 1 / (cost)
            max_ease = max(krnl_trans_improvement_ease.values())
            for krnl_trans, ease in krnl_trans_improvement_ease.items():
                krnl_trans_improvement_ease[krnl_trans] = ease / max_ease

            for krnl in krnls:
                krnl_improvement_ease[krnl] = 0

            for krnl_trans, ease in krnl_trans_improvement_ease.items():
                krnl, trans = krnl_trans
                krnl_improvement_ease[krnl] = max(ease, krnl_improvement_ease[krnl])


        return krnl_improvement_ease

    # select the kernel for the move
    def select_kernel(self, ex_dp, sim_dp, selected_metric, move_sorted_metric_dir):

        # get each kernel's contributions
        krnl_contribution_dict = self.get_kernels_s_contribution(selected_metric, sim_dp)
        # get each kernel's improvement cost
        krnl_improvement_ease = self.get_kernels_s_improvement_ease(ex_dp, sim_dp, selected_metric, move_sorted_metric_dir)




        # combine the selections methods
        # multiply the probabilities for a more complex metric
        krnl_prob_dict = {}
        for krnl in krnl_contribution_dict.keys():
            krnl_prob_dict[krnl] = krnl_contribution_dict[krnl] * krnl_improvement_ease[krnl]

        # give zero probablity to the krnls that you filtered out
        for krnl in sim_dp.get_dp_stats().get_kernels():
            if krnl not in krnl_prob_dict.keys():
                krnl_prob_dict[krnl] = 0
        # sort
        #krnl_prob_dict_sorted = {k: v for k, v in sorted(krnl_prob_dict.items(), key=lambda item: item[1])}
        krnl_prob_dict_sorted = sorted(krnl_prob_dict.items(), key=lambda item: item[1], reverse=True)

        # get the worse kernel
        if config.move_krnel_ranking_mode == "exact":  # for area to allow us pick scenarios that are not necessarily the worst
            #selected_krnl = list(krnl_prob_dict_sorted.keys())[
            #    len(krnl_prob_dict_sorted.keys()) - 1 - self.krnel_rnk_to_consider]
            for krnl, prob in krnl_prob_dict_sorted:
                if krnl.get_task_name() in self.krnels_not_to_consider:
                    continue
                selected_krnl = krnl
                break
        else:
            selected_krnl = self.pick_from_prob_dict(krnl_prob_dict_sorted)

        if config.transformation_selection_mode == "random":
            krnls = sim_dp.get_dp_stats().get_kernels()
            random.seed(datetime.now().microsecond)
            selected_krnl = random.choice(krnls)

        return selected_krnl, krnl_prob_dict, krnl_prob_dict_sorted

    # select blocks for the move
    def select_block(self, sim_dp, ex_dp, selected_krnl, selected_metric):
        # get the hot block for the kernel. Hot means the most contributing block for the kernel/metric of interest
        hot_blck = sim_dp.get_dp_stats().get_hot_block_of_krnel(selected_krnl.get_task_name(), selected_metric)

        # randomly pick one
        if config.transformation_selection_mode =="random":
            random.seed(datetime.now().microsecond)
            hot_blck = any_block = random.choice(ex_dp.get_hardware_graph().get_blocks())  # this is just dummmy to prevent breaking the plotting

        # hot_blck_synced is the same block but ensured that the block instance
        # is chosen from ex instead of sim, so it can be modified
        hot_blck_synced = self.dh.find_cores_hot_kernel_blck_bottlneck(ex_dp, hot_blck)
        block_prob_dict = sim_dp.get_dp_stats().get_hot_block_of_krnel_sorted(selected_krnl.get_task_name(), selected_metric)
        return hot_blck_synced, block_prob_dict

    def select_block_without_sync(self, sim_dp, selected_krnl, selected_metric):
        # get the hot block for the kernel. Hot means the most contributing block for the kernel/metric of interest
        hot_blck = sim_dp.get_dp_stats().get_hot_block_of_krnel(selected_krnl.get_task_name(), selected_metric)
        # hot_blck_synced is the same block but ensured that the block instance
        # is chosen from ex instead of sim, so it can be modified
        block_prob_dict = sim_dp.get_dp_stats().get_hot_block_of_krnel_sorted(selected_krnl.get_task_name(), selected_metric)
        return hot_blck, block_prob_dict

    def change_read_task_to_write_if_necessary(self, ex_dp, sim_dp, move_to_apply, selected_krnl):
        tasks_synced = [task__ for task__ in move_to_apply.get_block_ref().get_tasks_of_block() if
                        task__.name == selected_krnl.get_task_name()]
        if len(tasks_synced) == 0:  # this condition happens when we have a read task and we have unloaded reads
            krnl_s_tsk = ex_dp.get_hardware_graph().get_task_graph().get_task_by_name(
                move_to_apply.get_kernel_ref().get_task_name())
            parents_s_task = [el.get_name() for el in
                              ex_dp.get_hardware_graph().get_task_graph().get_task_s_parents(krnl_s_tsk)]
            tasks_on_block = [el.get_name() for el in move_to_apply.get_block_ref().get_tasks_of_block()]
            for parent_task in parents_s_task:
                if parent_task in tasks_on_block:
                    parents_task_obj = ex_dp.get_hardware_graph().get_task_graph().get_task_by_name(parent_task)
                    krnl = sim_dp.get_kernel_by_task_name(parents_task_obj)
                    move_to_apply.set_krnel_ref(krnl)
        return

    def find_block_with_sharing_tasks(self, ex_dp, selected_block, selected_krnl):
        succeeded = False
        # not covering ic at the moment
        if selected_block.type =="ic":
            return succeeded, "_"

        # get task of block
        cur_block_tasks = [el.get_name() for el in selected_block.get_tasks_of_block()]
        krnl_task = selected_krnl.get_task().get_name()
        # get other blocks in the system
        all_blocks = ex_dp.get_hardware_graph().get_blocks()
        all_blocks_minus_src_block = list(set(all_blocks) - set([selected_block]))
        assert(len(all_blocks) == len(all_blocks_minus_src_block) +1), "all_blocks must have one more block in it"

        if selected_block.type == "pe":
            blocks_with_sharing_task_type = "mem"
        elif selected_block.type == "mem":
            blocks_with_sharing_task_type = "pe"

        blocks_to_look_at = [blck for blck in all_blocks_minus_src_block if blck.type == blocks_with_sharing_task_type]

        # iterate through ic neighs (of oppotie type, i.e., for mem, look for pe, and for pe look for mem),
        # and look for shared tasks. If there is no shared task, we should move the block somewhere where there is
        # sort the neighbours based on the number of sharings.
        blocks_sorted_based_on_sharing = sorted(blocks_to_look_at,  key=lambda blck: len(list(set(cur_block_tasks) - set([el.get_name() for el in blck.get_tasks_of_block()]))))
        for block_with_sharing in blocks_sorted_based_on_sharing:
            block_tasks = [el.get_name() for el in block_with_sharing.get_tasks_of_block()]
            if krnl_task in block_tasks:
                return True, block_with_sharing
        else:
            return False, "_"



    def find_improve_routing(self, ex_dp, sim_dp, selected_block, selected_krnl_task):
        if not selected_block.type == "ic":
            return None, None
        result = True
        task_name = selected_krnl_task.get_task_name()
        task_s_blocks = ex_dp.get_hardware_graph().get_blocks_of_task_by_name(task_name)
        pe = [blk for blk in task_s_blocks if blk.type == "pe"][0]
        mems =[blk for blk in task_s_blocks if blk.type == "mem"]

        ic_entry, ic_exit = None, None
        for mem in mems:
            path= sim_dp.dp.get_hardware_graph().get_path_between_two_vertecies(pe, mem)
            if len(path)> 4:  # more than two ICs
                ic_entry = path[1]
                ic_exit = path[-2]
                break

        success =  not(ic_entry==None)
        return success, ic_exit

    def can_improve_routing(self, ex_dp, sim_dp, selected_block, selected_krnl_task):
        if not selected_block.type == "ic":
            return False
        result = True
        task_name = selected_krnl_task.get_name()
        task_s_blocks = ex_dp.get_hardware_graph().get_blocks_of_task_by_name(task_name)
        pe =[blk for blk in task_s_blocks if blk.type == "pe"][0]
        mems =[blk for blk in task_s_blocks if blk.type == "mem"]

        for mem in mems:
            path= ex_dp.get_hardware_graph().get_path_between_two_vertecies(pe, mem)
            if len(path) > 4:  # more than two ICs
                return True
        return False

    def can_improve_locality(self, ex_dp, selected_block, selected_krnl_task):
        result = True

        # not covering ic at the moment
        if selected_block.type =="ic":
            return False

        # get task of block
        cur_block_tasks = list(set([el.get_name() for el in selected_block.get_tasks_of_block()]))
        # get neighbouring ic
        ic = [neigh for neigh in selected_block.get_neighs() if neigh.type == "ic"][0]
        if selected_block.type == "pe":
            blocks_with_sharing_task_type = "mem"
        elif selected_block.type == "mem":
            blocks_with_sharing_task_type = "pe"

        ic_neighs = [neigh for neigh in ic.get_neighs() if neigh.type == blocks_with_sharing_task_type]

        # iterate through ic neighs (of oppotie type, i.e., for mem, look for pe, and for pe look for mem),
        # and look for shared tasks. If there is no shared task, we should move the block somewhere where there is
        for block in ic_neighs:
            block_tasks = list(set([el.get_name() for el in block.get_tasks_of_block()]))
            shared_tasks_exist = len(list(set(cur_block_tasks) - set(block_tasks))) < len(list(set(cur_block_tasks)))
            if shared_tasks_exist:
                result = False
                break
        return result

    def find_absorbing_block_tuple(self, ex_dp, sim_dp, task_name, block):
        task_pe = None
        for blk in ex_dp.get_hardware_graph().get_blocks():
            # only cover absorbing for PEs
            if not blk.type == "pe" :
                continue
            blk_tasks =  [el.get_name() for el in el.get_tasks_of_block()]
            if task_name in blk_tasks:
                task_pe = blk
                break

        ic_neigh = [neigh for neigh in task_pe.get_neighs() if neigh.type == "ic"][0]
        ic_neigh_neigh = [neigh for neigh in ic_neigh.get_neighs() if neigh.type == "ic"][0]

        # we don't mess with system ic
        if self.is_system_ic(ex_dp, sim_dp, ic_neigh):
            absorbee, absorber = None, None
        else:
            # if the ic didn't have any memory attached to it, return true
            mem_neighs = [neigh for neigh in ic_neigh.get_neighs() if neigh.type == "mem"]
            if len(mem_neighs) == 0:
                absorbee, absorber = ic_neigh, ic_neigh_neigh
            else:
                absorbee, absorber = None, None

        return absorbee, absorber

    def can_absorb_block(self, ex_dp, sim_dp, task_name):
        task_pe = None
        for blk in ex_dp.get_hardware_graph().get_blocks():
            # only cover absorbing for PEs
            if not blk.type == "pe" :
                continue
            blk_tasks =  [el.get_name() for el in blk.get_tasks_of_block()]
            if task_name in blk_tasks:
                task_pe = blk
                break

        ic_neigh = [neigh for neigh in task_pe.get_neighs() if neigh.type == "ic"][0]
        # we don't mess with system ic
        if self.is_system_ic(ex_dp, sim_dp, ic_neigh):
            result = False
        else:
            # if the ic didn't have any memory attached to it, return true
            mem_neighs = [neigh for neigh in ic_neigh.get_neighs() if neigh.type == "mem"]
            if len(mem_neighs) == 0:
                result = True
            else:
                result = False

        return result

    # ------------------------------
    # Functionality:
    #    generate a move to apply.  A move consists of a metric, direction, kernel, block and transformation.
    #    At the moment, we target the metric that is most further from the budget. Kernel and block are chosen
    #    based on how much they contribute to the distance.
    #
    # Variables:
    #      des_tup: (design, simulated design)
    # ------------------------------
    def sel_moves_based_on_dis(self, des_tup):
        ex_dp, sim_dp = des_tup
        if config.DEBUG_FIX: random.seed(0)
        else: time.sleep(.00001), random.seed(datetime.now().microsecond)

        # select move components
        t_0 = time.time()
        selected_metric, metric_prob_dir_dict, sorted_metric_dir = self.select_metric(sim_dp)
        t_1 = time.time()
        move_dir = self.select_dir(sim_dp, selected_metric)
        t_2 = time.time()
        selected_krnl, krnl_prob_dict, krnl_prob_dir_dict_sorted = self.select_kernel(ex_dp, sim_dp, selected_metric, sorted_metric_dir)
        t_3 = time.time()
        selected_block, block_prob_dict = self.select_block(sim_dp, ex_dp, selected_krnl, selected_metric)
        t_4 = time.time()
        transformation_name,transformation_sub_name, transformation_batch_mode, total_transformation_cnt = self.select_transformation(ex_dp, sim_dp, selected_block, selected_metric, selected_krnl, sorted_metric_dir)
        t_5 = time.time()

        self.set_design_space_size(des_tup[0], des_tup[1])


        """
        if sim_dp.dp_stats.fits_budget(1) and self.dram_feasibility_check_pass(ex_dp) and self.can_improve_locality(selected_block, selected_krnl):
            transformation_sub_name = "transfer_no_prune"
            transformation_name = "improve_locality"
            transformation_batch_mode = "single"
            selected_metric = "cost"
        """
        # prepare for move
        # if bus, (forgot which exception), if IP, avoid split .
        """
        if sim_dp.dp_stats.fits_budget(1) and not self.dram_feasibility_check_pass(ex_dp):
            transformation_name = "dram_fix"
            transformation_sub_name = "dram_fix_no_prune"
            transformation_batch_mode = "single"
            selected_metric = "cost"
        """
        if self.is_cleanup_iter():
            transformation_name = "cleanup"
            transformation_sub_name = "non"
            transformation_batch_mode = "single"
            selected_metric = "cost"
            #config.VIS_GR_PER_GEN = True
            self.cleanup_ctr += 1
            #config.VIS_GR_PER_GEN = False

        # log the data for future profiling/data collection/debugging
        move_to_apply = move(transformation_name, transformation_sub_name, transformation_batch_mode, move_dir, selected_metric, selected_block, selected_krnl, krnl_prob_dir_dict_sorted)
        move_to_apply.set_sorted_metric_dir(sorted_metric_dir)
        move_to_apply.set_logs(sim_dp.database.db_input.task_workload[selected_krnl.get_task_name()],"workload")
        move_to_apply.set_logs(sim_dp.dp_stats.get_system_complex_metric("cost"), "cost")
        move_to_apply.set_logs(krnl_prob_dict, "kernels")
        move_to_apply.set_logs(metric_prob_dir_dict, "metrics")
        move_to_apply.set_logs(block_prob_dict, "blocks")
        move_to_apply.set_logs(self.krnel_rnk_to_consider, "kernel_rnk_to_consider")
        move_to_apply.set_logs(sim_dp.dp_stats.dist_to_goal(["power", "area", "latency", "cost"],
                                                                                config.metric_sel_dis_mode),"ref_des_dist_to_goal_all")
        move_to_apply.set_logs(sim_dp.dp_stats.dist_to_goal(["power", "area", "latency"],
                                                                                config.metric_sel_dis_mode),"ref_des_dist_to_goal_non_cost")

        for blck_of_interest in ex_dp.get_blocks():
            self.get_transformation_design_space_size(move_to_apply, ex_dp, sim_dp, blck_of_interest, selected_metric, sorted_metric_dir)


        move_to_apply.set_logs(t_1 - t_0, "metric_selection_time")
        move_to_apply.set_logs(t_2 - t_1, "dir_selection_time")
        move_to_apply.set_logs(t_3 - t_2, "kernel_selection_time")
        move_to_apply.set_logs(t_4 - t_3, "block_selection_time")
        move_to_apply.set_logs(t_5 - t_4, "transformation_selection_time")

        blck_ref = move_to_apply.get_block_ref()
        gc.disable()
        blck_ref_cp = cPickle.loads(cPickle.dumps(blck_ref, -1))
        gc.enable()
        # ------------------------
        # prepare for the move
        # ------------------------
        if move_to_apply.get_transformation_name() == "identity":
            pass
            #move_to_apply.set_validity(False, "NoValidTransformationException")
        if move_to_apply.get_transformation_name() == "swap":
            self.dh.unload_read_mem(ex_dp)  # unload memories
            if not blck_ref.type == "ic":
                self.dh.unload_buses(ex_dp)  # unload buses
            else:
                self.dh.unload_read_buses(ex_dp)  # unload buses
            # get immediate superior/inferior block (based on the desired direction)
            imm_block = self.dh.get_immediate_block_multi_metric(blck_ref,
                                                 move_to_apply.get_metric(), move_to_apply.get_sorted_metric_dir(),
                                                 blck_ref.get_tasks_of_block())  # immediate block either superior or
            move_to_apply.set_dest_block(imm_block)
            move_to_apply.set_customization_type(blck_ref, imm_block)

            move_to_apply.set_tasks(blck_ref.get_tasks_of_block())
        elif move_to_apply.get_transformation_name() in ["split_swap"]:
            self.dh.unload_buses(ex_dp)  # unload buses
            # get immediate superior/inferior block (based on the desired direction)
            succeeded,migrant = blck_ref.get_tasks_by_name(move_to_apply.get_kernel_ref().get_task_name())
            if not succeeded:
                move_to_apply.set_validity(False, "NoMigrantException")
            else:
                imm_block = self.dh.get_immediate_block_multi_metric(blck_ref,
                                                     move_to_apply.get_metric(), move_to_apply.get_sorted_metric_dir(),
                                                     [migrant])  # immediate block either superior or
                move_to_apply.set_dest_block(imm_block)

                self.dh.unload_read_mem(ex_dp)  # unload memories
                self.change_read_task_to_write_if_necessary(ex_dp, sim_dp, move_to_apply, selected_krnl)
                migrant_tasks = self.dh.migrant_selection(ex_dp, sim_dp, blck_ref, blck_ref_cp, move_to_apply.get_kernel_ref(),
                                                          move_to_apply.get_transformation_batch())
                #migrant_tasks  = list(set(move_to_apply.get_block_ref().get_tasks()) - set(migrant_tasks_))  # reverse the order to allow for swap to happen on the ref_block
                move_to_apply.set_tasks(migrant_tasks)
                move_to_apply.set_customization_type(blck_ref, imm_block)
        elif move_to_apply.get_transformation_name() in ["split"]:
            # select tasks to migrate
            #self.change_read_task_to_write_if_necessary(ex_dp, sim_dp, move_to_apply, selected_krnl)
            migrant_tasks = self.dh.migrant_selection(ex_dp, sim_dp, blck_ref, blck_ref_cp, move_to_apply.get_kernel_ref(),
                                                      move_to_apply.get_transformation_batch())


            # determine the parallelism type
            parallelism_type_ = [] #with repetition
            parallelism_type = []
            migrant_tasks_names = [el.get_name() for el in migrant_tasks]
            for task_ in migrant_tasks_names:
                parallelism_type_.append(self.get_task_parallelism_type(sim_dp, task_, selected_krnl.get_task_name()))
            parallelism_type = list(set(parallelism_type_))

            move_to_apply.set_parallelism_type(parallelism_type)
            move_to_apply.set_tasks(migrant_tasks)
            if len(migrant_tasks) == 0:
                move_to_apply.set_validity(False, "NoParallelTaskException")
            if blck_ref.subtype == "ip": # makes no sense to split the IPs,
                                                              # it can actually cause problems where
                                                              # we end up duplicating the hardware
                move_to_apply.set_validity(False, "IPSplitException")
        elif move_to_apply.get_transformation_name() == "migrate":
            if not selected_block.type == "ic":  # ic migration is not supported
                # check and see if tasks exist (if not, it was a read)
                imm_block_present, found_blck_to_mig_to, mig_selection_mode, parallelism_type, locality_type = self.select_block_to_migrate_to(ex_dp,
                                                                                                              sim_dp,
                                                                                                              blck_ref_cp,
                                                                                                              move_to_apply.get_metric(),
                                                                                                              move_to_apply.get_sorted_metric_dir(),
                                                                                                              move_to_apply.get_kernel_ref())



                move_to_apply.set_parallelism_type(parallelism_type)
                move_to_apply.set_locality_type(locality_type)
                self.dh.unload_buses(ex_dp)  # unload buses
                self.dh.unload_read_mem(ex_dp)  # unload memories
                if not imm_block_present.subtype == "ip":
                    self.change_read_task_to_write_if_necessary(ex_dp, sim_dp, move_to_apply, selected_krnl)
                if not found_blck_to_mig_to:
                    move_to_apply.set_validity(False, "NoMigrantException")
                    imm_block_present = blck_ref
                elif move_to_apply.get_kernel_ref().get_task_name() in ["souurce", "siink", "dummy_last"]:
                    move_to_apply.set_validity(False, "NoMigrantException")
                    imm_block_present = blck_ref
                else:
                    migrant_tasks = self.dh.migrant_selection(ex_dp, sim_dp, blck_ref, blck_ref_cp, move_to_apply.get_kernel_ref(),
                                                              mig_selection_mode)

                    move_to_apply.set_tasks(migrant_tasks)
                    move_to_apply.set_dest_block(imm_block_present)
            else:
                move_to_apply.set_validity(False, "ICMigrationException")
        elif move_to_apply.get_transformation_name() == "dram_fix":
            any_block = ex_dp.get_hardware_graph().get_blocks()[0]  # this is just dummmy to prevent breaking the plotting
            any_task =  any_block.get_tasks_of_block()[0]
            move_to_apply.set_tasks([any_task]) # this is just dummmy to prevent breaking the plotting
            move_to_apply.set_dest_block(any_block)
            pass
        elif move_to_apply.get_transformation_name() == "transfer":
            if move_to_apply.get_transformation_sub_name() == "locality_improvement":
                succeeded, dest_block = self.find_block_with_sharing_tasks(ex_dp, selected_block, selected_krnl)
                if succeeded:
                    move_to_apply.set_dest_block(dest_block)
                    move_to_apply.set_tasks([move_to_apply.get_kernel_ref().get_task()])
                else:
                    move_to_apply.set_validity(False, "TransferException")
            else:
                move_to_apply.set_validity(False, "TransferException")
            pass
        elif move_to_apply.get_transformation_name() == "routing":
            if move_to_apply.get_transformation_sub_name() == "routing_improvement":
                succeeded, dest_block = self.find_improve_routing(ex_dp, sim_dp, selected_block, selected_krnl)
                if succeeded:
                    move_to_apply.set_dest_block(dest_block)
                    move_to_apply.set_tasks([move_to_apply.get_kernel_ref().get_task()])
                else:
                    move_to_apply.set_validity(False, "RoutingException")
            else:
                move_to_apply.set_validity(False, "RoutingException")
            pass
        elif move_to_apply.get_transformation_name() == "cleanup":
            if self.can_absorb_block(ex_dp, sim_dp, move_to_apply.get_kernel_ref().get_task_name()):
                move_to_apply.set_transformation_sub_name("absorb")
                absorbee, absorber = self.find_absorbing_block_tuple(ex_dp, sim_dp, move_to_apply.get_kernel_ref().get_task_name())
                if absorber == None or absorbee == "None":
                    move_to_apply.set_validity(False, "NoAbsorbee(er)Exception")
                else:
                    move_to_apply.set_ref_block(absorbee)
                    move_to_apply.set_dest_block(absorber)
            else:
                move_to_apply.set_validity(False, "CostPairingException")
                self.dh.unload_buses(ex_dp)  # unload buses
                self.dh.unload_read_mem(ex_dp)  # unload memories
                task_1, block_task_1, task_2, block_task_2 = self.find_task_with_similar_mappable_ips(des_tup)
                # we also randomize
                if not (task_1 is None) and (random.choice(np.arange(0,1,.1))>.5):
                    move_to_apply.set_ref_block(block_task_1)
                    migrant_tasks = [task_1]
                    imm_block_present = block_task_2
                    move_to_apply.set_tasks(migrant_tasks)
                    move_to_apply.set_dest_block(imm_block_present)
                else:
                    pair = self.gen_block_match_cleanup_move(des_tup)
                    if len(pair) == 0:
                        move_to_apply.set_validity(False, "CostPairingException")
                    else:
                        ref_block = pair[0]
                        if not ref_block.type == "ic":  # ic migration is not supported
                            move_to_apply.set_ref_block(ref_block)
                            migrant_tasks = ref_block.get_tasks_of_block()
                            imm_block_present = pair[1]
                            move_to_apply.set_tasks(migrant_tasks)
                            move_to_apply.set_dest_block(imm_block_present)


        move_to_apply.set_breadth_depth(self.SA_current_breadth, self.SA_current_depth, self.SA_current_mini_breadth)  # set depth and breadth (for debugging/ plotting)
        return move_to_apply, total_transformation_cnt

    # ------------------------------
    # Functionality:
    #       How to choose the move.
    # Variables:
    #      des_tup: (design, simulated design)
    # ------------------------------
    def sel_moves(self, des_tup, mode="dist_rank"):  # TODO: add mode
        if mode == "dist_rank":  # rank and choose probabilistically based on distance
            return self.sel_moves_based_on_dis(des_tup)
        else:
            print("mode" + mode + " is not supported")
            exit(0)

    # ------------------------------
    # Functionality:
    #       Calculate possible neighbours, though not randomly.
    #       des_tup: design tuple. Contains a design tuple (ex_dp, sim_dp). ex_dp: design to find neighbours for.
    #                                                                       sim_dp: simulated ex_dp.
    # ------------------------------
    def gen_some_neighs_orchestrated(self, des_tup):
        all_possible_moves = config.navigation_moves
        ctr = 0
        kernel_pos_to_hndl = self.hot_krnl_pos  # for now, but change it

        # generate neighbours until you hit the threshold
        while(ctr < self.num_neighs_to_try):
            ex_dp, sim_dp = des_tup
            # Copy to avoid modifying the current designs.
            new_ex_dp_1 = copy.deepcopy(ex_dp)
            new_sim_dp_1 = copy.deepcopy(sim_dp)
            new_ex_dp = copy.deepcopy(new_ex_dp_1)
            new_sim_dp = copy.deepcopy(new_sim_dp_1)

            # apply the move
            yield self.dh.apply_move(new_ex_dp, new_sim_dp, all_possible_moves[ctr%len(all_possible_moves)], kernel_pos_to_hndl)
            ctr += 1
        return 0

    def simulated_annealing_energy(self, sim_dp_stats):
        return ()

    def find_best_design(self,sim_stat_ex_dp_dict, sim_dp_stat_ann_delta_energy_dict, sim_dp_stat_ann_delta_energy_dict_all_metrics, best_sim_dp_stat_so_far, best_ex_dp_so_far):
        if config.heuristic_type == "SA" or config.heuristic_type == "moos":
            return self.find_best_design_SA(sim_stat_ex_dp_dict, sim_dp_stat_ann_delta_energy_dict, sim_dp_stat_ann_delta_energy_dict_all_metrics, best_sim_dp_stat_so_far, best_ex_dp_so_far)
        else:
            return self.find_best_design_others(sim_stat_ex_dp_dict, sim_dp_stat_ann_delta_energy_dict, sim_dp_stat_ann_delta_energy_dict_all_metrics, best_sim_dp_stat_so_far, best_ex_dp_so_far)


    # find the best design from a list
    def find_best_design_SA(self, sim_stat_ex_dp_dict, sim_dp_stat_ann_delta_energy_dict, sim_dp_stat_ann_delta_energy_dict_all_metrics, best_sim_dp_stat_so_far, best_ex_dp_so_far):
        # for all metrics, we only return true if there is an improvement,
        # it does not make sense to look at block equality (as energy won't be zero in cases that there is a difficult trade off)
        if config.sel_next_dp == "all_metrics":
            sorted_sim_dp_stat_ann_delta_energy_dict_all_metrics = sorted(sim_dp_stat_ann_delta_energy_dict_all_metrics.items(),
                                                              key=lambda x: x[1])

            if sorted_sim_dp_stat_ann_delta_energy_dict_all_metrics[0][1] < -.0001: # a very small number
                # if a better design (than the best exist), return
                return sorted_sim_dp_stat_ann_delta_energy_dict_all_metrics[0], True
            else:
                return sorted_sim_dp_stat_ann_delta_energy_dict_all_metrics[0], False


        # two blocks are equal if they have the same generic instance name
        # and have been scheduled the same tasks
        def blocks_are_equal(block_1, block_2):
            if not selected_block.get_generic_instance_name() == best_sim_selected_block.get_generic_instance_name():
                return False
            elif selected_block.get_generic_instance_name() == best_sim_selected_block.get_generic_instance_name():
                # make sure tasks are not the same.
                # this is to avoid scenarios where a block is improved (but it's generic name) equal to the
                # next block bottleneck. Here we make sure tasks are different
                block_1_tasks = [tsk.name for tsk in block_1.get_tasks_of_block()]
                block_2_tasks = [tsk.name for tsk in block_1.get_tasks_of_block()]
                task_diff = list(set(block_1_tasks) - set(block_2_tasks))
                return len(task_diff) == 0

        # get the best_sim info
        sim_dp = best_sim_dp_stat_so_far.dp
        ex_dp = best_ex_dp_so_far

        best_sim_selected_metric, metric_prob_dict, best_sorted_metric_dir = self.select_metric(sim_dp)
        best_sim_move_dir = self.select_dir(sim_dp, best_sim_selected_metric)
        best_sim_selected_krnl, krnl_prob_dict = self.select_kernel(ex_dp, sim_dp, best_sim_selected_metric, best_sorted_metric_dir)
        best_sim_selected_block, block_prob_dict = self.select_block_without_sync(sim_dp, best_sim_selected_krnl, best_sim_selected_metric)

        # sort the design base on distance
        sorted_sim_dp_stat_ann_delta_energy_dict = sorted(sim_dp_stat_ann_delta_energy_dict.items(), key=lambda x: x[1])

        #best_neighbour_stat, best_neighbour_delta_energy = sorted_sim_dp_stat_ann_delta_energy_dict[0]  # here we can be smarter
        if sorted_sim_dp_stat_ann_delta_energy_dict[0][1] < 0:
            # if a better design (than the best exist), return
            return sorted_sim_dp_stat_ann_delta_energy_dict[0], True
        elif sorted_sim_dp_stat_ann_delta_energy_dict[0][1] == 0:
            # if no better design
            if len(sorted_sim_dp_stat_ann_delta_energy_dict[0]) == 1:
                # if no better design (only one design means that our original design is the one)
                return sorted_sim_dp_stat_ann_delta_energy_dict[0], False
            else:
                # filter out the designs  which hasn't seen a distance improvement
                sim_dp_to_select_from = []
                for sim_dp_stat, energy in sorted_sim_dp_stat_ann_delta_energy_dict:
                    #sim_dp_to_select_from.append((sim_dp_stat, energy))

                    designs_to_consider = []
                    sim_dp = sim_dp_stat.dp
                    ex_dp = sim_stat_ex_dp_dict[sim_dp_stat]
                    selected_metric, metric_prob_dict, sorted_metric_dir = self.select_metric(sim_dp)
                    move_dir = self.select_dir(sim_dp, selected_metric)
                    selected_krnl, krnl_prob_dict = self.select_kernel(ex_dp, sim_dp, selected_metric, sorted_metric_dir)
                    selected_block, block_prob_dict = self.select_block_without_sync(sim_dp, selected_krnl,
                                                                                     selected_metric)
                    if energy > 0:
                        designs_to_consider.append((sim_dp_stat, energy))
                        return sim_dp_to_select_from[0], False
                    elif not selected_krnl.get_task_name() == best_sim_selected_krnl.get_task_name():
                        designs_to_consider.append((sim_dp_stat, energy))
                        return designs_to_consider[0], True
                    elif not blocks_are_equal(selected_block, best_sim_selected_block):
                    #elif not selected_block.get_generic_instance_name() == best_sim_selected_block.get_generic_instance_name():
                        designs_to_consider.append((sim_dp_stat, energy))
                        return designs_to_consider[0], True

                    return sim_dp_to_select_from[0], False


    # find the best design from a list
    def find_best_design_others(self, sim_stat_ex_dp_dict, sim_dp_stat_ann_delta_energy_dict, sim_dp_stat_ann_delta_energy_dict_all_metrics, best_sim_dp_stat_so_far, best_ex_dp_so_far):
        # for all metrics, we only return true if there is an improvement,
        # it does not make sense to look at block equality (as energy won't be zero in cases that there is a difficult trade off)
        if config.sel_next_dp == "all_metrics":
            sorted_sim_dp_stat_ann_delta_energy_dict_all_metrics = sorted(sim_dp_stat_ann_delta_energy_dict_all_metrics.items(),
                                                              key=lambda x: x[1])

            if sorted_sim_dp_stat_ann_delta_energy_dict_all_metrics[0][1] < -.0001: # a very small number
                # if a better design (than the best exist), return
                return sorted_sim_dp_stat_ann_delta_energy_dict_all_metrics[0], True
            else:
                return sorted_sim_dp_stat_ann_delta_energy_dict_all_metrics[0], False


        # two blocks are equal if they have the same generic instance name
        # and have been scheduled the same tasks
        def blocks_are_equal(block_1, block_2):
            if not selected_block.get_generic_instance_name() == best_sim_selected_block.get_generic_instance_name():
                return False
            elif selected_block.get_generic_instance_name() == best_sim_selected_block.get_generic_instance_name():
                # make sure tasks are not the same.
                # this is to avoid scenarios where a block is improved (but it's generic name) equal to the
                # next block bottleneck. Here we make sure tasks are different
                block_1_tasks = [tsk.name for tsk in block_1.get_tasks_of_block()]
                block_2_tasks = [tsk.name for tsk in block_1.get_tasks_of_block()]
                task_diff = list(set(block_1_tasks) - set(block_2_tasks))
                return len(task_diff) == 0

        # get the best_sim info
        sim_dp = best_sim_dp_stat_so_far.dp
        ex_dp = best_ex_dp_so_far

        best_sim_selected_metric, metric_prob_dict, best_sorted_metric_dir = self.select_metric(sim_dp)
        best_sim_move_dir = self.select_dir(sim_dp, best_sim_selected_metric)
        best_sim_selected_krnl, krnl_prob_dict = self.select_kernel(ex_dp, sim_dp, best_sim_selected_metric, best_sorted_metric_dir)
        best_sim_selected_block, block_prob_dict = self.select_block_without_sync(sim_dp, best_sim_selected_krnl, best_sim_selected_metric)

        # sort the design base on distance
        sorted_sim_dp_stat_ann_delta_energy_dict = sorted(sim_dp_stat_ann_delta_energy_dict.items(), key=lambda x: x[1])

        #best_neighbour_stat, best_neighbour_delta_energy = sorted_sim_dp_stat_ann_delta_energy_dict[0]  # here we can be smarter
        if sorted_sim_dp_stat_ann_delta_energy_dict[0][1] < 0:
            # if a better design (than the best exist), return
            return sorted_sim_dp_stat_ann_delta_energy_dict[0], True
        elif sorted_sim_dp_stat_ann_delta_energy_dict[0][1] == 0:
            # if no better design
            if len(sorted_sim_dp_stat_ann_delta_energy_dict[0]) == 1:
                # if no better design (only one design means that our original design is the one)
                return sorted_sim_dp_stat_ann_delta_energy_dict[0], False
            else:
                # filter out the designs  which hasn't seen a distance improvement
                sim_dp_to_select_from = []
                for sim_dp_stat, energy in sorted_sim_dp_stat_ann_delta_energy_dict:
                    if energy == 0:
                        sim_dp_to_select_from.append((sim_dp_stat, energy))


                designs_to_consider = []
                for sim_dp_stat, energy in sim_dp_to_select_from:
                    sim_dp = sim_dp_stat.dp
                    ex_dp = sim_stat_ex_dp_dict[sim_dp_stat]
                    selected_metric, metric_prob_dict, sorted_metric_dir = self.select_metric(sim_dp)
                    move_dir = self.select_dir(sim_dp, selected_metric)
                    selected_krnl, krnl_prob_dict = self.select_kernel(ex_dp, sim_dp, selected_metric, sorted_metric_dir)
                    selected_block, block_prob_dict = self.select_block_without_sync(sim_dp, selected_krnl,
                                                                                     selected_metric)
                    if not selected_krnl.get_task_name() == best_sim_selected_krnl.get_task_name():
                        designs_to_consider.append((sim_dp_stat, energy))
                    elif not blocks_are_equal(selected_block, best_sim_selected_block):
                    #elif not selected_block.get_generic_instance_name() == best_sim_selected_block.get_generic_instance_name():
                        designs_to_consider.append((sim_dp_stat, energy))

                if len(designs_to_consider) == 0:
                    return sim_dp_to_select_from[0], False
                else:
                    return designs_to_consider[0], True # can be smarter here


    # use simulated annealing to pick the next design(s).
    # Use this link to understand simulated annealing (SA) http://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/we /glossary/anneal.html
    # cur_temp: current temperature for simulated annealing
    def moos_greedy_design_selection(self, sim_stat_ex_dp_dict, sim_dp_stat_list, best_ex_dp_so_far, best_sim_dp_so_far_stats, cur_temp):
        def get_kernel_not_to_consider(krnels_not_to_consider, cur_best_move_applied, random_move_applied):
            if cur_best_move_applied == None: # only none for the first iteration
                move_applied = random_move_applied
            else:
                move_applied = cur_best_move_applied
            if move_applied == None:
                return None

            krnl_prob_dict_sorted = move_applied.krnel_prob_dict_sorted
            for krnl, prob in krnl_prob_dict_sorted:
                if krnl.get_task_name() in krnels_not_to_consider:
                    continue
                return krnl.get_task_name()



        # get the kernel of interest using this for now to collect cached designs
        best_sim_selected_metric, metric_prob_dict,best_sorted_metric_dir = self.select_metric(best_sim_dp_so_far_stats.dp)
        best_sim_move_dir = self.select_dir(best_sim_dp_so_far_stats.dp, best_sim_selected_metric)
        best_sim_selected_krnl, _, _= self.select_kernel(best_ex_dp_so_far, best_sim_dp_so_far_stats.dp, best_sim_selected_metric, best_sorted_metric_dir)
        if best_sim_selected_krnl.get_task_name() not in self.recently_cached_designs.keys():
           self.recently_cached_designs[best_sim_selected_krnl.get_task_name()]  = []


        # get the worse case cost for normalizing the cost when calculating the distance
        best_cost = min([sim_dp.get_system_complex_metric("cost") for sim_dp in (sim_dp_stat_list + [best_sim_dp_so_far_stats])])
        self.database.set_ideal_metric_value("cost", "glass", best_cost)

        # find if any of the new designs meet the budget
        new_designs_meeting_budget = []  # designs that are meeting the budget
        for sim_dp_stat in sim_dp_stat_list:
            if sim_dp_stat.fits_budget(1):
                new_designs_meeting_budget.append(sim_dp_stat)

        new_designs_meeting_budget_with_dram = []  # designs that are meeting the budget
        for sim_dp_stat in sim_dp_stat_list:
            if sim_dp_stat.fits_budget(1):
                ex_dp = sim_stat_ex_dp_dict[sim_dp_stat]
                if ex_dp.has_system_bus():
                    new_designs_meeting_budget_with_dram.append(sim_dp_stat)
        dram_fixed = False
        if len(new_designs_meeting_budget_with_dram) > 0 and not self.dram_feasibility_check_pass(best_ex_dp_so_far):
            dram_fixed = True


        # find each design's simulated annealing Energy difference with the best design's energy
        # if any of the designs meet the budget or it's a cleanup iteration, include cost in distance calculation.
        # note that when we compare, we need to use the same dist_to_goal calculation, hence
        # ann_energy_best_dp_so_far needs to use the same calculation
        metric_to_target , metric_prob_dict, sorted_metric_dir = self.select_metric(best_sim_dp_so_far_stats.dp)
        include_cost_in_distance = best_sim_dp_so_far_stats.fits_budget(1) or (len(new_designs_meeting_budget) > 0) or self.is_cleanup_iter() or (len(new_designs_meeting_budget_with_dram)>0)
        if include_cost_in_distance:
            ann_energy_best_dp_so_far = best_sim_dp_so_far_stats.dist_to_goal(["cost", "latency", "power", "area"],
                                                                              "eliminate")
            ann_energy_best_dp_so_far_all_metrics = best_sim_dp_so_far_stats.dist_to_goal(["cost", "latency", "power", "area"],
                                                                              "eliminate")
        else:
            ann_energy_best_dp_so_far = best_sim_dp_so_far_stats.dist_to_goal([metric_to_target], "dampen")
            ann_energy_best_dp_so_far_all_metrics = best_sim_dp_so_far_stats.dist_to_goal(["power", "area", "latency"],
                                                                              "dampen")
        sim_dp_stat_ann_delta_energy_dict = {}
        sim_dp_stat_ann_delta_energy_dict_all_metrics = {}
        # deleteee the following debugging lines
        if config.print_info_regularly:
            print("--------%%%%%%%%%%%---------------")
            print("--------%%%%%%%%%%%---------------")
            print("first the best design from the previous iteration")
            print(" des" + " latency:" + str(best_sim_dp_so_far_stats.get_system_complex_metric("latency")))
            print(" des" + " power:" + str(
                best_sim_dp_so_far_stats.get_system_complex_metric("power")))
            print("energy :" + str(ann_energy_best_dp_so_far))



        sim_dp_to_look_at = [] # which designs to look at.
        # only look at the designs that meet the budget (if any), basically  prioritize these designs first
        if len(new_designs_meeting_budget_with_dram) > 0:
            sim_dp_to_look_at = new_designs_meeting_budget_with_dram
        elif len(new_designs_meeting_budget) > 0:
            sim_dp_to_look_at = new_designs_meeting_budget
        else:
            sim_dp_to_look_at = sim_dp_stat_list

        for sim_dp_stat in sim_dp_to_look_at:
            if include_cost_in_distance:
                sim_dp_stat_ann_delta_energy_dict[sim_dp_stat] = sim_dp_stat.dist_to_goal(
                    ["cost", "latency", "power", "area"], "eliminate") - ann_energy_best_dp_so_far
                sim_dp_stat_ann_delta_energy_dict_all_metrics[sim_dp_stat] = sim_dp_stat.dist_to_goal(
                    ["cost", "latency", "power", "area"], "eliminate") - ann_energy_best_dp_so_far_all_metrics
            else:
                new_design_energy = sim_dp_stat.dist_to_goal([metric_to_target], "dampen")
                sim_dp_stat_ann_delta_energy_dict[sim_dp_stat] = new_design_energy - ann_energy_best_dp_so_far
                new_design_energy_all_metrics = sim_dp_stat.dist_to_goal(["power", "latency", "area"], "dampen")
                sim_dp_stat_ann_delta_energy_dict_all_metrics[sim_dp_stat] = new_design_energy_all_metrics - ann_energy_best_dp_so_far_all_metrics

        # changing the seed for random selection
        if config.DEBUG_FIX: random.seed(0)
        else: time.sleep(.00001), random.seed(datetime.now().microsecond)

        result, design_improved = self.find_best_design(sim_stat_ex_dp_dict, sim_dp_stat_ann_delta_energy_dict,
                                                        sim_dp_stat_ann_delta_energy_dict_all_metrics, best_sim_dp_so_far_stats, best_ex_dp_so_far)
        best_neighbour_stat, best_neighbour_delta_energy = result

        if config.print_info_regularly:
            print("all the designs tried")
            for el, energy in sim_dp_stat_ann_delta_energy_dict_all_metrics.items():
                print("----------------")
                sim_dp_ =  el.dp
                if not sim_dp_.move_applied == None:
                    sim_dp_.move_applied.print_info()
                    print("energy" + str(energy))
                    print("design's latency: " + str(el.get_system_complex_metric("latency")))
                    print("design's power: " + str(el.get_system_complex_metric("power")))
                    print("design's area: " + str(el.get_system_complex_metric("area")))
                    print("design's sub area: " + str(el.get_system_complex_area_stacked_dram()))

        # if any negative (desired move) value is detected or there is a design in the new batch
        #  that meet the budget, but the previous best design didn't, we have at least one improved solution
        found_an_improved_solution = (len(new_designs_meeting_budget)>0 and not(best_sim_dp_so_far_stats).fits_budget(1)) or design_improved or dram_fixed


        # for debugging. delete later
        if (len(new_designs_meeting_budget)>0 and not(best_sim_dp_so_far_stats).fits_budget(1)):
            print("what")

        if not found_an_improved_solution:
            # avoid not improving
            self.krnel_stagnation_ctr +=1
            self.des_stag_ctr += 1
            if self.krnel_stagnation_ctr > config.max_krnel_stagnation_ctr:
                self.krnel_rnk_to_consider = min(self.krnel_rnk_to_consider + 1, len(best_sim_dp_so_far_stats.get_kernels()) -1)
                krnel_not_to_consider = get_kernel_not_to_consider(self.krnels_not_to_consider, best_sim_dp_so_far_stats.dp.move_applied, sim_dp_to_look_at[-1].dp.move_applied)
                if not krnel_not_to_consider == None:
                    self.krnels_not_to_consider.append(krnel_not_to_consider)
                #self.krnel_stagnation_ctr = 0
                #self.recently_seen_design_ctr = 0
        elif best_neighbour_stat.dp.dp_rep.get_hardware_graph().get_SOC_design_code() in self.recently_cached_designs[best_sim_selected_krnl.get_task_name()] and False:
            # avoid circular exploration
            self.recently_seen_design_ctr += 1
            self.des_stag_ctr += 1
            if self.recently_seen_design_ctr > config.max_recently_seen_design_ctr:
                self.krnel_rnk_to_consider = min(self.krnel_rnk_to_consider + 1,
                                                 len(best_sim_dp_so_far_stats.get_kernels()) - 1)
                self.krnel_stagnation_ctr = 0
                #self.recently_seen_design_ctr = 0
        else:
            self.krnel_stagnation_ctr = max(0, self.krnel_stagnation_ctr -1)
            if self.krnel_stagnation_ctr == 0:
                if not len(self.krnels_not_to_consider) == 0:
                    self.krnels_not_to_consider = self.krnels_not_to_consider[:-1]
                self.krnel_rnk_to_consider = max(0, self.krnel_rnk_to_consider - 1)
            self.cleanup_ctr +=1
            self.des_stag_ctr = 0
            self.recently_seen_design_ctr = 0

        # initialize selected_sim_dp
        selected_sim_dp = best_sim_dp_so_far_stats.dp
        if found_an_improved_solution:
            selected_sim_dp = best_neighbour_stat.dp
        else:
            try:
                #if math.e**(best_neighbour_delta_energy/max(cur_temp, .001)) < random.choice(range(0, 1)):
                #    selected_sim_dp = best_neighbour_stat.dp
                if random.choice(range(0, 1)) < math.e**(best_neighbour_delta_energy/max(cur_temp, .001)):
                    selected_sim_dp = best_neighbour_stat.dp
            except:
                selected_sim_dp = best_neighbour_stat.dp

        # cache the best design
        if len(self.recently_cached_designs[best_sim_selected_krnl.get_task_name()]) < config.recently_cached_designs_queue_size:
            self.recently_cached_designs[best_sim_selected_krnl.get_task_name()].append(selected_sim_dp.dp_rep.get_hardware_graph().get_SOC_design_code())
        else:
            self.recently_cached_designs[best_sim_selected_krnl.get_task_name()][self.population_generation_cnt%config.recently_cached_designs_queue_size] = selected_sim_dp.dp_rep.get_hardware_graph().get_SOC_design_code()

        return selected_sim_dp, found_an_improved_solution

    # use simulated annealing to pick the next design(s).
    # Use this link to understand simulated annealing (SA) http://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/we /glossary/anneal.html
    # cur_temp: current temperature for simulated annealing
    def SA_design_selection(self, sim_stat_ex_dp_dict, sim_dp_stat_list, best_ex_dp_so_far, best_sim_dp_so_far_stats, cur_temp):

        def get_kernel_not_to_consider(krnels_not_to_consider, cur_best_move_applied, random_move_applied):
            if cur_best_move_applied == None: # only none for the first iteration
                move_applied = random_move_applied
            else:
                move_applied = cur_best_move_applied
            if move_applied == None:
                return None

            krnl_prob_dict_sorted = move_applied.krnel_prob_dict_sorted
            for krnl, prob in krnl_prob_dict_sorted:
                if krnl.get_task_name() in krnels_not_to_consider:
                    continue
                return krnl.get_task_name()


        # get the kernel of interest using this for now to collect cached designs
        best_sim_selected_metric, metric_prob_dict,best_sorted_metric_dir = self.select_metric(best_sim_dp_so_far_stats.dp)
        best_sim_move_dir = self.select_dir(best_sim_dp_so_far_stats.dp, best_sim_selected_metric)
        best_sim_selected_krnl, _, _= self.select_kernel(best_ex_dp_so_far, best_sim_dp_so_far_stats.dp, best_sim_selected_metric, best_sorted_metric_dir)
        if best_sim_selected_krnl.get_task_name() not in self.recently_cached_designs.keys():
           self.recently_cached_designs[best_sim_selected_krnl.get_task_name()]  = []


        # get the worse case cost for normalizing the cost when calculating the distance
        best_cost = min([sim_dp.get_system_complex_metric("cost") for sim_dp in (sim_dp_stat_list + [best_sim_dp_so_far_stats])])
        self.database.set_ideal_metric_value("cost", "glass", best_cost)

        # find if any of the new designs meet the budget
        new_designs_meeting_budget = []  # designs that are meeting the budget
        for sim_dp_stat in sim_dp_stat_list:
            if sim_dp_stat.fits_budget(1):
                new_designs_meeting_budget.append(sim_dp_stat)

        new_designs_meeting_budget_with_dram = []  # designs that are meeting the budget
        for sim_dp_stat in sim_dp_stat_list:
            if sim_dp_stat.fits_budget(1):
                ex_dp = sim_stat_ex_dp_dict[sim_dp_stat]
                if ex_dp.has_system_bus():
                    new_designs_meeting_budget_with_dram.append(sim_dp_stat)
        dram_fixed = False
        if len(new_designs_meeting_budget_with_dram) > 0 and not self.dram_feasibility_check_pass(best_ex_dp_so_far):
            dram_fixed = True


        # find each design's simulated annealing Energy difference with the best design's energy
        # if any of the designs meet the budget or it's a cleanup iteration, include cost in distance calculation.
        # note that when we compare, we need to use the same dist_to_goal calculation, hence
        # ann_energy_best_dp_so_far needs to use the same calculation
        metric_to_target , metric_prob_dict, sorted_metric_dir = self.select_metric(best_sim_dp_so_far_stats.dp)
        include_cost_in_distance = best_sim_dp_so_far_stats.fits_budget(1) or (len(new_designs_meeting_budget) > 0) or self.is_cleanup_iter() or (len(new_designs_meeting_budget_with_dram)>0)
        if include_cost_in_distance:
            ann_energy_best_dp_so_far = best_sim_dp_so_far_stats.dist_to_goal(["cost", "latency", "power", "area"],
                                                                              "eliminate")
            ann_energy_best_dp_so_far_all_metrics = best_sim_dp_so_far_stats.dist_to_goal(["cost", "latency", "power", "area"],
                                                                              "eliminate")
        else:
            ann_energy_best_dp_so_far = best_sim_dp_so_far_stats.dist_to_goal([metric_to_target], "dampen")
            ann_energy_best_dp_so_far_all_metrics = best_sim_dp_so_far_stats.dist_to_goal(["power", "area", "latency"],
                                                                              "dampen")
        sim_dp_stat_ann_delta_energy_dict = {}
        sim_dp_stat_ann_delta_energy_dict_all_metrics = {}
        # deleteee the following debugging lines
        print("--------%%%%%%%%%%%---------------")
        print("--------%%%%%%%%%%%---------------")
        print("first the best design from the previous iteration")
        print(" des" + " latency:" + str(best_sim_dp_so_far_stats.get_system_complex_metric("latency")))
        print(" des" + " power:" + str(
            best_sim_dp_so_far_stats.get_system_complex_metric("power")))
        print("energy :" + str(ann_energy_best_dp_so_far))



        sim_dp_to_look_at = [] # which designs to look at.
        # only look at the designs that meet the budget (if any), basically  prioritize these designs first
        if len(new_designs_meeting_budget_with_dram) > 0:
            sim_dp_to_look_at = new_designs_meeting_budget_with_dram
        elif len(new_designs_meeting_budget) > 0:
            sim_dp_to_look_at = new_designs_meeting_budget
        else:
            sim_dp_to_look_at = sim_dp_stat_list

        for sim_dp_stat in sim_dp_to_look_at:
            if include_cost_in_distance:
                sim_dp_stat_ann_delta_energy_dict[sim_dp_stat] = sim_dp_stat.dist_to_goal(
                    ["cost", "latency", "power", "area"], "eliminate") - ann_energy_best_dp_so_far
                sim_dp_stat_ann_delta_energy_dict_all_metrics[sim_dp_stat] = sim_dp_stat.dist_to_goal(
                    ["cost", "latency", "power", "area"], "eliminate") - ann_energy_best_dp_so_far_all_metrics
            else:
                new_design_energy = sim_dp_stat.dist_to_goal([metric_to_target], "dampen")
                sim_dp_stat_ann_delta_energy_dict[sim_dp_stat] = new_design_energy - ann_energy_best_dp_so_far
                new_design_energy_all_metrics = sim_dp_stat.dist_to_goal(["power", "latency", "area"], "dampen")
                sim_dp_stat_ann_delta_energy_dict_all_metrics[sim_dp_stat] = new_design_energy_all_metrics - ann_energy_best_dp_so_far_all_metrics

        # changing the seed for random selection
        if config.DEBUG_FIX: random.seed(0)
        else: time.sleep(.00001), random.seed(datetime.now().microsecond)

        result, design_improved = self.find_best_design(sim_stat_ex_dp_dict, sim_dp_stat_ann_delta_energy_dict,
                                                        sim_dp_stat_ann_delta_energy_dict_all_metrics, best_sim_dp_so_far_stats, best_ex_dp_so_far)
        best_neighbour_stat, best_neighbour_delta_energy = result

        print("all the designs tried")
        for el, energy in sim_dp_stat_ann_delta_energy_dict_all_metrics.items():
            print("----------------")
            sim_dp_ =  el.dp
            if not sim_dp_.move_applied == None:
                sim_dp_.move_applied.print_info()
                print("energy" + str(energy))
                print("design's latency: " + str(el.get_system_complex_metric("latency")))
                print("design's power: " + str(el.get_system_complex_metric("power")))
                print("design's area: " + str(el.get_system_complex_metric("area")))
                print("design's sub area: " + str(el.get_system_complex_area_stacked_dram()))

        # if any negative (desired move) value is detected or there is a design in the new batch
        #  that meet the budget, but the previous best design didn't, we have at least one improved solution
        found_an_improved_solution = (len(new_designs_meeting_budget)>0 and not(best_sim_dp_so_far_stats).fits_budget(1)) or design_improved or dram_fixed


        # for debugging. delete later
        if (len(new_designs_meeting_budget)>0 and not(best_sim_dp_so_far_stats).fits_budget(1)):
            print("what")

        if not found_an_improved_solution:
            # avoid not improving
            self.krnel_stagnation_ctr +=1
            self.des_stag_ctr += 1
            if self.krnel_stagnation_ctr > config.max_krnel_stagnation_ctr:
                self.krnel_rnk_to_consider = min(self.krnel_rnk_to_consider + 1, len(best_sim_dp_so_far_stats.get_kernels()) -1)
                krnel_not_to_consider = get_kernel_not_to_consider(self.krnels_not_to_consider, best_sim_dp_so_far_stats.dp.move_applied, sim_dp_to_look_at[-1].dp.move_applied)
                if not krnel_not_to_consider == None:
                    self.krnels_not_to_consider.append(krnel_not_to_consider)
                #self.krnel_stagnation_ctr = 0
                #self.recently_seen_design_ctr = 0
        elif best_neighbour_stat.dp.dp_rep.get_hardware_graph().get_SOC_design_code() in self.recently_cached_designs[best_sim_selected_krnl.get_task_name()] and False:
            # avoid circular exploration
            self.recently_seen_design_ctr += 1
            self.des_stag_ctr += 1
            if self.recently_seen_design_ctr > config.max_recently_seen_design_ctr:
                self.krnel_rnk_to_consider = min(self.krnel_rnk_to_consider + 1,
                                                 len(best_sim_dp_so_far_stats.get_kernels()) - 1)
                self.krnel_stagnation_ctr = 0
                #self.recently_seen_design_ctr = 0
        else:
            self.krnel_stagnation_ctr = max(0, self.krnel_stagnation_ctr -1)
            if self.krnel_stagnation_ctr == 0:
                if not len(self.krnels_not_to_consider) == 0:
                    self.krnels_not_to_consider = self.krnels_not_to_consider[:-1]
                self.krnel_rnk_to_consider = max(0, self.krnel_rnk_to_consider - 1)
            self.cleanup_ctr +=1
            self.des_stag_ctr = 0
            self.recently_seen_design_ctr = 0

        # initialize selected_sim_dp
        selected_sim_dp = best_sim_dp_so_far_stats.dp
        if found_an_improved_solution:
            selected_sim_dp = best_neighbour_stat.dp
        else:
            try:
                #if math.e**(best_neighbour_delta_energy/max(cur_temp, .001)) < random.choice(range(0, 1)):
                #    selected_sim_dp = best_neighbour_stat.dp
                if random.choice(range(0, 1)) < math.e**(best_neighbour_delta_energy/max(cur_temp, .001)):
                    selected_sim_dp = best_neighbour_stat.dp

            except:
                selected_sim_dp = best_neighbour_stat.dp

        # cache the best design
        if len(self.recently_cached_designs[best_sim_selected_krnl.get_task_name()]) < config.recently_cached_designs_queue_size:
            self.recently_cached_designs[best_sim_selected_krnl.get_task_name()].append(selected_sim_dp.dp_rep.get_hardware_graph().get_SOC_design_code())
        else:
            self.recently_cached_designs[best_sim_selected_krnl.get_task_name()][self.population_generation_cnt%config.recently_cached_designs_queue_size] = selected_sim_dp.dp_rep.get_hardware_graph().get_SOC_design_code()

        return selected_sim_dp, found_an_improved_solution

    def find_design_scalarized_value_from_moos_perspective(self, sim, lambdas):
        sim_metric_values = {}
        value = []
        for metric_name in config.budgetted_metrics:
            for type, id in sim.get_designs_SOCs():
                if metric_name == "latency":
                    metric_val = sum(list(sim.dp_stats.get_SOC_metric_value(metric_name, type, id).values()))
                else:
                    metric_val = sim.dp_stats.get_SOC_metric_value(metric_name, type, id)

                value.append(Decimal(metric_val)*lambdas[metric_name])


        #return max(value)
        return sum(value)

                # ------------------------------
    # Functionality:
    #     select the next best design (from the sorted dp)
    # Variables
    #       ex_sim_dp_dict: example_simulate_design_point_list. List of designs to pick from.
    # ------------------------------
    def sel_start_dp_moos(self, ex_sim_dp_dict, best_sim_dp_so_far, best_ex_dp_so_far, lambda_list):
        # convert to stats
        sim_dp_list = list(ex_sim_dp_dict.values())
        sim_dp_stat_list = [sim_dp.dp_stats for sim_dp in sim_dp_list]
        sim_stat_ex_dp_dict = {}
        for k, v in ex_sim_dp_dict.items():
            sim_stat_ex_dp_dict[v.dp_stats] = k


        # find the ones that fit the expanded budget (note that budget radius shrinks)
        sim_scalarized_value = {}
        for ex, sim in ex_sim_dp_dict.items():
            value = self.find_design_scalarized_value_from_moos_perspective(sim, lambda_list)
            sim_scalarized_value[sim] = value

        min_scalarized_value = float('inf')
        min_sim = ""
        for sim,value in  sim_scalarized_value.items():
            if value <  min_scalarized_value:
                min_sim = sim
                min_ex = sim_stat_ex_dp_dict[sim.dp_stats]
                min_scalarized_value = value

        if min_sim == "":
            print("some thing went wrong. should have at least one minimum")
        return min_ex, min_sim


        # extract the design
        for key, val in ex_sim_dp_dict.items():
            key.sanity_check()
            if val == selected_sim_dp:
                selected_ex_dp = key
                break

        # generate verification data
        if found_improved_solution and config.RUN_VERIFICATION_PER_IMPROVMENT:
            self.gen_verification_data(selected_sim_dp, selected_ex_dp)
        return selected_ex_dp, selected_sim_dp



    # ------------------------------
    # Functionality:
    #     select the next best design (from the sorted dp)
    # Variables
    #       ex_sim_dp_dict: example_simulate_design_point_list. List of designs to pick from.
    # ------------------------------
    def sel_next_dp(self, ex_sim_dp_dict, best_sim_dp_so_far, best_ex_dp_so_far, cur_temp):
        # convert to stats
        sim_dp_list = list(ex_sim_dp_dict.values())
        sim_dp_stat_list = [sim_dp.dp_stats for sim_dp in sim_dp_list]
        sim_stat_ex_dp_dict = {}
        for k, v in ex_sim_dp_dict.items():
            sim_stat_ex_dp_dict[v.dp_stats] = k


        # find the ones that fit the expanded budget (note that budget radius shrinks)
        selected_sim_dp, found_improved_solution = self.SA_design_selection(sim_stat_ex_dp_dict, sim_dp_stat_list,
                                                                            best_ex_dp_so_far, best_sim_dp_so_far.dp_stats,
                                                                            cur_temp)

        self.found_any_improvement = self.found_any_improvement or found_improved_solution

        if not found_improved_solution:
            selected_sim_dp = self.so_far_best_sim_dp
            selected_ex_dp = self.so_far_best_ex_dp
        else:
            # extract the design
            for key, val in ex_sim_dp_dict.items():
                key.sanity_check()
                if val == selected_sim_dp:
                    selected_ex_dp = key
                    break

        # generate verification data
        if found_improved_solution and config.RUN_VERIFICATION_PER_IMPROVMENT:
            self.gen_verification_data(selected_sim_dp, selected_ex_dp)
        return selected_ex_dp, selected_sim_dp

    # ------------------------------
    # Functionality:
    #    simulate one design.
    # Variables
    #      ex_dp: example design point. Design point to simulate.
    #      database: hardware/software data base to simulated based off of.
    # ------------------------------
    def sim_one_design(self, ex_dp, database):
        if config.simulation_method == "power_knobs":
            sim_dp = self.dh.convert_to_sim_des_point(ex_dp)
            power_knob_sim_dp = self.dh.convert_to_sim_des_point(ex_dp)
            OSA = OSASimulator(sim_dp, database, power_knob_sim_dp)
        else:
            sim_dp = self.dh.convert_to_sim_des_point(ex_dp)
            # Simulator initialization
            OSA = OSASimulator(sim_dp, database)  # change

        # Does the actual simulation
        t = time.time()
        OSA.simulate()
        sim_time = time.time() - t

        # profile info
        sim_dp.set_population_generation_cnt(self.population_generation_cnt)
        sim_dp.set_population_observed_number(self.population_observed_ctr)
        sim_dp.set_depth_number(self.SA_current_depth)
        sim_dp.set_simulation_time(sim_time)

        #print("sim time" + str(sim_time))
        #exit(0)
        return sim_dp

    # ------------------------------
    # Functionality:
    #       Sampling from the task distribution. This is used for jitter  incorporation.
    # Variables:
    #       ex_dp: example design point.
    # ------------------------------
    def generate_sample(self, ex_dp, hw_sampling):
        #new_ex_dp = copy.deepcopy(ex_dp)
        gc.disable()
        new_ex_dp = cPickle.loads(cPickle.dumps(ex_dp, -1))
        gc.enable()
        new_ex_dp.sample_hardware_graph(hw_sampling)
        return new_ex_dp




    def transform_to_most_inferior_design(self, ex_dp:ExDesignPoint):
        new_ex_dp = cPickle.loads(cPickle.dumps(ex_dp, -1))
        move_to_try = move("swap", "swap", "irrelevant", "-1", "latency", "", "", "")
        all_blocks = new_ex_dp.get_blocks()
        for block in  all_blocks:
            self.dh.unload_read_mem(new_ex_dp)  # unload memories
            if not block.type == "ic":
                self.dh.unload_buses(new_ex_dp)  # unload buses
            else:
                self.dh.unload_read_buses(new_ex_dp)  # unload buses

            move_to_try.set_ref_block(block)
            # get immediate superior/inferior block (based on the desired direction)
            most_inferior_block = self.dh.get_most_inferior_block(block, block.get_tasks_of_block())
            move_to_try.set_dest_block(most_inferior_block)
            move_to_try.set_customization_type(block, most_inferior_block)
            move_to_try.set_tasks(block.get_tasks_of_block())
            self.dh.unload_read_mem(new_ex_dp)    # unload read memories
            move_to_try.validity_check()  # call after unload rad mems, because we need to check the scenarios where
                                          # task is unloaded from the mem, but was decided to be migrated/swapped
            new_ex_dp_res, succeeded = self.dh.apply_move([new_ex_dp,""], move_to_try)
            #self.dh.load_tasks_to_read_mem_and_ic(new_ex_dp_res)  # loading the tasks on to memory and ic
            new_ex_dp_res.hardware_graph.pipe_design()
            new_ex_dp_res.sanity_check()
            new_ex_dp = new_ex_dp_res
            #cPickle.loads(cPickle.dumps(new_ex_dp_res, -1))
            #self.dh.load_tasks_to_read_mem_and_ic(new_ex_dp)  # loading the tasks on to memory and ic


        self.dh.load_tasks_to_read_mem_and_ic(new_ex_dp)  # loading the tasks on to memory and ic
        return new_ex_dp

    def transform_to_most_inferior_design_before_loop_unrolling(self, ex_dp: ExDesignPoint):
        new_ex_dp = cPickle.loads(cPickle.dumps(ex_dp, -1))
        move_to_try = move("swap", "swap", "irrelevant", "-1", "latency", "", "", "")
        all_blocks = new_ex_dp.get_blocks()
        for block in all_blocks:
            self.dh.unload_read_mem(new_ex_dp)  # unload memories
            if not block.type == "ic":
                self.dh.unload_buses(new_ex_dp)  # unload buses
            else:
                self.dh.unload_read_buses(new_ex_dp)  # unload buses

            move_to_try.set_ref_block(block)
            # get immediate superior/inferior block (based on the desired direction)
            most_inferior_block = self.dh.get_most_inferior_block_before_unrolling(block,  block.get_tasks_of_block())
            #most_inferior_block = self.dh.get_most_inferior_block(block, block.get_tasks_of_block())
            move_to_try.set_dest_block(most_inferior_block)
            move_to_try.set_customization_type(block, most_inferior_block)
            move_to_try.set_tasks(block.get_tasks_of_block())
            self.dh.unload_read_mem(new_ex_dp)  # unload read memories
            move_to_try.validity_check()  # call after unload rad mems, because we need to check the scenarios where
            # task is unloaded from the mem, but was decided to be migrated/swapped
            new_ex_dp_res, succeeded = self.dh.apply_move([new_ex_dp, ""], move_to_try)
            # self.dh.load_tasks_to_read_mem_and_ic(new_ex_dp_res)  # loading the tasks on to memory and ic
            new_ex_dp_res.hardware_graph.pipe_design()
            new_ex_dp_res.sanity_check()
            new_ex_dp = new_ex_dp_res
            # cPickle.loads(cPickle.dumps(new_ex_dp_res, -1))
            # self.dh.load_tasks_to_read_mem_and_ic(new_ex_dp)  # loading the tasks on to memory and ic

        self.dh.load_tasks_to_read_mem_and_ic(new_ex_dp)  # loading the tasks on to memory and ic
        return new_ex_dp

    def single_out_workload(self,ex_dp, database, workload, workload_tasks):
        new_ex_dp = cPickle.loads(cPickle.dumps(ex_dp, -1))
        database_ = cPickle.loads(cPickle.dumps(database, -1))
        for block in new_ex_dp.get_blocks():
            for dir in ["loop_back","write","read"]:
                tasks= block.get_tasks_of_block_by_dir(dir)
                for task in tasks:
                    if task.get_name() not in workload_tasks:
                        block.unload((task,dir))

        tasks = new_ex_dp.get_tasks()
        for task in tasks:
            children = task.get_children()[:]
            parents = task.get_parents()[:]
            for child in children:
                if child.get_name() not in workload_tasks:
                    task.remove_child(child)
            for parent in parents:
                if parent.get_name() not in workload_tasks:
                    task.remove_parent(parent)

        database_.set_workloads_last_task({workload: database_.db_input.workloads_last_task[workload]})
        return new_ex_dp, database_

    # ------------------------------
    # Functionality:
    #       Evaluate the design. 1. simulate 2. collect (profile) data.
    # Variables:
    #       ex_dp: example design point.
    #       database: database containing hardware/software modeled characteristics.
    # ------------------------------
    def eval_design(self, ex_dp:ExDesignPoint, database):
        #start = time.time()
        # according to config singular runs
        if config.eval_mode == "singular":
            print("this mode is deprecated. just use statistical. singular is simply a special case")
            exit(0)
            return self.sim_one_design(ex_dp, database) # evaluation the design directly
        elif config.eval_mode == "statistical":
            # generate a population (geneate_sample), evaluate them and reduce to some statistical indicator
            ex_dp_pop_sample = [self.generate_sample(ex_dp, database.hw_sampling) for i in range(0, self.database.hw_sampling["population_size"])] # population sample
            ex_dp.get_tasks()[0].task_id_for_debugging_static += 1
            sim_dp_pop_sample = list(map(lambda ex_dp_: self.sim_one_design(ex_dp_, database), ex_dp_pop_sample)) # evaluate the population sample

            # collect profiling information
            sim_dp_statistical = SimDesignPointContainer(sim_dp_pop_sample, database, config.statistical_reduction_mode)
            #print("time is:" + str(time.time() -start))
            return sim_dp_statistical
        else:
            print("mode" + config.eval_mode + " is not defined for eval design")

    # ------------------------------
    # Functionality:
    #       generate verification (platform architect digestible) designs.
    # Variables:
    #       sim_dp: simulated design point.
    # ------------------------------
    def gen_verification_data(self, sim_dp_, ex_dp_):
        #from data_collection.FB_private.verification_utils.PA_generation.PA_generators import *
        import_ver = importlib.import_module("data_collection.FB_private.verification_utils.PA_generation.PA_generators")
        # iterate till you can make a directory
        while True:
            date_time = datetime.now().strftime('%m-%d_%H-%M_%S')
            #result_folder = os.path.join(self.result_dir, "data_per_design",
            #                             date_time+"_"+str(self.name_ctr))
            # one for PA data collection
            result_folder = os.path.join(self.result_dir+"/../../", "data_per_design",
                                         date_time+"_"+str(self.name_ctr))

            if not os.path.isdir(result_folder):
                os.makedirs(result_folder)
                collection_ctr = self.name_ctr # used to realize which results to compare
                break

        ex_with_PA = [] #
        pa_ver_obj = import_ver.PAVerGen()  # initialize a PA generator
        # make all the combinations
        knobs_list, knob_order = pa_ver_obj.gen_all_PA_knob_combos(import_ver.PA_knobs_to_explore)  # generate knob combinations
        #   for different PA designs. Since PA has extra knobs, we'd like sweep this knobs for verification purposes.
        knob_ctr = 0
        # Iterate though the knob combos and generate a (PA digestible) design accordingly
        for knobs in knobs_list:
            result_folder_for_knob = os.path.join(result_folder, "PA_knob_ctr_"+str(knob_ctr))
            for sim_dp in sim_dp_.get_design_point_list():
                sim_dp.reset_PA_knobs()
                sim_dp.update_ex_id(date_time+"_"+str(collection_ctr)+"_" + str(self.name_ctr))
                sim_dp.update_FARSI_ex_id(date_time+"_"+str(collection_ctr))
                sim_dp.update_PA_knob_ctr_id(str(knob_ctr))
                sim_dp.update_PA_knobs(knobs, knob_order, import_ver.auto_tuning_knobs)
                PA_result_folder = os.path.join(result_folder_for_knob, str(sim_dp.id))
                os.makedirs(PA_result_folder)

                #  dump data for bus-memory data with connection (used for bw calculation)
                sim_dp.dump_mem_bus_connection_bw(PA_result_folder) # write the results into a file
                # initialize and do some clean up
                #vis_hardware.vis_hardware(sim_dp, config.hw_graphing_mode, PA_result_folder)
                sim_dp.dp_stats.dump_stats(PA_result_folder)
                if config.VIS_SIM_PROG: vis_sim.plot_sim_data(sim_dp.dp_stats, sim_dp, PA_result_folder)
                block_names = [block.instance_name for block in sim_dp.hardware_graph.blocks]
                vis_hardware.vis_hardware(sim_dp, config.hw_graphing_mode, PA_result_folder)
                pa_obj = import_ver.PAGen(self.database, self.database.db_input.proj_name, sim_dp, PA_result_folder, config.sw_model)

                pa_obj.gen_all()  # generate the PA digestible design
                sim_dp.dump_props(PA_result_folder)  # write the results into a file
                #  pickle the results for (out of run) verifications.
                ex_dp_pickled_file = open(os.path.join(PA_result_folder, "ex_dp_pickled.txt"), "wb")
                dill.dump(ex_dp_, ex_dp_pickled_file)
                ex_dp_pickled_file.close()

                database_pickled_file = open(os.path.join(PA_result_folder, "database_pickled.txt"), "wb")
                dill.dump(self.database, database_pickled_file)
                database_pickled_file.close()

                sim_dp_pickled_file = open(os.path.join(PA_result_folder, "sim_dp_pickled.txt"), "wb")
                dill.dump(sim_dp, sim_dp_pickled_file)
                sim_dp_pickled_file.close()
                self.name_ctr += 1
            knob_ctr += 1




    # ------------------------------
    # Functionality:
    #       generate one neighbour and evaluate it.
    # Variables:
    #       des_tup: starting point design point tuple (design point, simulated design point)
    # ------------------------------
    #@profile
    def gen_neigh_and_eval(self, des_tup):
        # "delete this later"
        print("------ depth ------")
        # generate on neighbour
        move_strt_time = time.time()
        ex_dp, move_to_try,total_trans_cnt = self.gen_one_neigh(des_tup)
        move_end_time = time.time()
        move_to_try.set_generation_time(move_end_time- move_strt_time)

        # generate a code for the design (that specifies the topology, mapping and scheduling).
        # look into cache and see if this design has been seen before. If so, just use the
        # cached value, other wise just use the sim from cache
        design_unique_code = ex_dp.get_hardware_graph().get_SOC_design_code()  # cache index
        if move_to_try.get_transformation_name() == "identity" or not move_to_try.is_valid():
            # if nothing has changed, just copy the sim from before
            sim_dp = des_tup[1]
        elif design_unique_code not in self.cached_SOC_sim.keys():
            self.population_observed_ctr += 1
            sim_dp = self.eval_design(ex_dp, self.database)  # evaluate the designs
            #if config.cache_seen_designs: # this seems to be slower than just simulation, because of deepcopy
            #    self.cached_SOC_sim[design_unique_code] = (ex_dp, sim_dp)
        else:
            ex_dp = self.cached_SOC_sim[design_unique_code][0]
            sim_dp = self.cached_SOC_sim[design_unique_code][1]

        # collect the moves for debugging/visualization
        if config.DEBUG_MOVE:
            if (self.population_generation_cnt % config.vis_reg_ctr_threshold) == 0 and self.SA_current_mini_breadth == 0:
                self.move_profile.append(move_to_try)  # for debugging
            self.last_move = move_to_try
            sim_dp.set_move_applied(move_to_try)

        # visualization and verification
        if config.VIS_GR_PER_GEN:
            vis_hardware.vis_hardware(sim_dp.get_dp_rep())
        if config.RUN_VERIFICATION_PER_GEN or \
                (config.RUN_VERIFICATION_PER_NEW_CONFIG and
                 not(sim_dp.dp.get_hardware_graph().get_SOC_design_code() in self.seen_SOC_design_codes)):
            self.gen_verification_data(sim_dp, ex_dp)
        self.seen_SOC_design_codes.append(sim_dp.dp.get_hardware_graph().get_SOC_design_code())


        if not sim_dp.move_applied == None and config.print_info_regularly:
            sim_dp.move_applied.print_info()
            print("design's latency: " + str(sim_dp.dp_stats.get_system_complex_metric("latency")))
            print("design's power: " + str(sim_dp.dp_stats.get_system_complex_metric("power")))
            print("design's area: " + str(sim_dp.dp_stats.get_system_complex_metric("area")))
            print("design's sub area: " + str(sim_dp.dp_stats.get_system_complex_area_stacked_dram()))


        return (ex_dp, sim_dp), total_trans_cnt


    def protected_gen_neigh_and_eval(self, des_tup):
        ctr = 0
        while True and ctr <100:
            ctr +=1
            try:
                des_tup_new, possible_des_cnt = self.gen_neigh_and_eval(des_tup)
                return des_tup_new, possible_des_cnt
                break
            except SystemExit:
                print("caught an exit")
                continue
            except Exception as e:
                print("caught an exception")
        print("return too many exception or exits")
        exit(0)

    # ------------------------------
    # Functionality:
    #       generate neighbours and evaluate them.
    #       neighbours are generated based on the depth and breath count determined in the config file.
    #       Depth means vertical, i.e., daisy chaining of the moves). Breadth means horizontal exploration.
    # Variables:
    #       des_tup: starting point design point tuple (design point, simulated design point)
    #       breadth: the breadth according to which to generate designs  (used for breadth wise search)
    #       depth: the depth according to which to generate designs (used for look ahead)
    # ------------------------------
    def gen_some_neighs_and_eval(self, des_tup, breath_length, depth_length, des_tup_list):
        # base case
        if depth_length == 0:
            return [des_tup]
        #des_tup_list = []
        # iterate on breath
        for i in range(0, breath_length):
            self.SA_current_mini_breadth = 0
            if not(breath_length == 1):
                self.SA_current_breadth += 1
                self.SA_current_depth = -1
                print("--------breadth--------")
            # iterate on depth (generate one neighbour and evaluate it)
            self.SA_current_depth += 1
            #des_tup_new, possible_des_cnt = self.gen_neigh_and_eval(des_tup)
            des_tup_new, possible_des_cnt = self.protected_gen_neigh_and_eval(des_tup)

            #self.total_iteration_ctr += 1

            # collect the generate design in a list and run sanity check on it
            des_tup_list.append(des_tup_new)

            # do more coverage if needed
            """ 
            for i in range(0, max(possible_des_cnt,1)-1):
                self.SA_current_mini_breadth += 1
                des_tup_new_breadth, _ = self.gen_neigh_and_eval(des_tup)
                des_tup_list.append(des_tup_new_breadth)
            """
            # just a quick optimization, since there is not need
            # to go deeper if we encounter identity.
            # This is because we will keep repeating the identity at the point
            if des_tup_new[1].move_applied.get_transformation_name() == "identity":
                break

            self.gen_some_neighs_and_eval(des_tup_new, 1, depth_length-1, des_tup_list)
            #des_tup_list.extend(self.gen_some_neighs_and_eval(des_tup_new, 1, depth_length-1))

            # visualization and sanity checks
            if config.VIS_MOVE_TRAIL:
                if (self.population_generation_cnt % config.vis_reg_ctr_threshold) == 0:
                    best_design_sim_cpy = copy.deepcopy(self.so_far_best_sim_dp)
                    self.des_trail_list.append((best_design_sim_cpy, des_tup_list[-1][1]))
                    self.last_des_trail = (best_design_sim_cpy, des_tup_list[-1][1])
                    #self.des_trail_list.append((cPickle.loads(cPickle.dumps(self.so_far_best_sim_dp, -1)),cPickle.loads(cPickle.dumps(des_tup_list[-1][1], -1))))
                    #self.last_des_trail = (cPickle.loads(cPickle.dumps(self.so_far_best_sim_dp, -1)),cPickle.loads(cPickle.dumps(des_tup_list[-1][1], -1)))


            #self.vis_move_ctr += 1
            if config.DEBUG_SANITY: des_tup[0].sanity_check()
        #return des_tup_list

    # simple simulated annealing
    def simple_SA(self):
        # define the result dictionary
        this_itr_ex_sim_dp_dict:Dict[ExDesignPoint: SimDesignPoint] = {}
        this_itr_ex_sim_dp_dict[self.so_far_best_ex_dp] = self.so_far_best_sim_dp  # init the res dict

        # navigate the space using depth and breath parameters
        strt = time.time()
        print("------------------------ itr:" + str(self.population_generation_cnt) + " ---------------------")
        self.SA_current_breadth = -1
        self.SA_current_depth = -1

        # generate some neighbouring design points and evaluate them
        des_tup_list =[]
        #config.SA_depth = 3*len(self.so_far_best_ex_dp.get_hardware_graph().get_blocks_by_type("mem"))+ len(self.so_far_best_ex_dp.get_hardware_graph().get_blocks_by_type("ic"))
        self.gen_some_neighs_and_eval((self.so_far_best_ex_dp, self.so_far_best_sim_dp), config.SA_breadth, config.SA_depth, des_tup_list)
        exploration_and_simulation_approximate_time_per_iteration = (time.time() - strt)/max(len(des_tup_list), 1)
        #print("sim time + neighbour generation per design point " + str((time.time() - strt)/max(len(des_tup_list), 1)))

        # convert (outputed) list to dictionary of (ex:sim) specified above.
        # Also, run sanity check on the design, making sure everything is alright
        for ex_dp, sim_dp in des_tup_list:
            sim_dp.add_exploration_and_simulation_approximate_time(exploration_and_simulation_approximate_time_per_iteration)
            this_itr_ex_sim_dp_dict[ex_dp] = sim_dp
            if config.DEBUG_SANITY:
                ex_dp.sanity_check()
        return this_itr_ex_sim_dp_dict


    def convert_tuple_list_to_parsable_csv(self, list_):
        result = ""
        for k, v in list_:
            result +=str(k) + "=" + str(v) + "___"
        return result

    def convert_dictionary_to_parsable_csv_with_underline(self, dict_):
        result = ""
        for k, v in dict_.items():
            phase_value_dict = list(v.values())[0]
            value = list(phase_value_dict.values())[0]
            result +=str(k) + "=" + str(value) + "___"
        return result

    def convert_dictionary_to_parsable_csv_with_semi_column(self, dict_):
        result = ""
        for k, v in dict_.items():
            result +=str(k) + "=" + str(v) + ";"
        return result

    # ------------------------------
    # Functionality:
    #      Explore the initial design. Basically just simulated the initial design
    # Variables
    #       it uses the config parameters that are used to instantiate the object.
    # ------------------------------
    def explore_one_design(self):
        self.so_far_best_ex_dp = self.init_ex_dp
        self.init_sim_dp = self.so_far_best_sim_dp = self.eval_design(self.so_far_best_ex_dp, self.database)
        this_itr_ex_sim_dp_dict = {}
        this_itr_ex_sim_dp_dict[self.so_far_best_ex_dp] = self.so_far_best_sim_dp
        #self.init_sim_dp = self.eval_design(self.so_far_best_ex_dp, self.database)

        # collect statistics about the design
        self.log_data(this_itr_ex_sim_dp_dict)
        self.collect_stats(this_itr_ex_sim_dp_dict)

        # visualize/checkpoint/PA generation
        vis_hardware.vis_hardware(self.so_far_best_sim_dp.get_dp_rep())
        if config.RUN_VERIFICATION_PER_GEN or config.RUN_VERIFICATION_PER_IMPROVMENT or config.RUN_VERIFICATION_PER_NEW_CONFIG:
            self.gen_verification_data(self.so_far_best_sim_dp, self.so_far_best_ex_dp)

    def get_log_data(self):
        return self.log_data_list

    def write_data_log(self, log_data, reason_to_terminate, case_study, result_dir_specific, unique_number, file_name):
        output_file_all = os.path.join(result_dir_specific, file_name + "_all_reults.csv")
        csv_columns = list(log_data[0].keys())
        # minimal output
        with open(output_file_all, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in log_data:
                writer.writerow(data)

    # ------------------------------
    # Functionality:
    #      log the data for plotting and such
    # ------------------------------
    def log_data(self, this_itr_ex_sim_dp_dict):
        ctr = len(self.log_data_list)
        for sim_dp in this_itr_ex_sim_dp_dict.values():
            sim_dp.add_exploration_and_simulation_approximate_time(self.neighbour_selection_time/len(list(this_itr_ex_sim_dp_dict.keys())))
            ma = sim_dp.get_move_applied()  # move applied
            if not ma == None:
                sorted_metrics = self.convert_tuple_list_to_parsable_csv(
                    [(el, val) for el, val in ma.sorted_metrics.items()])
                metric = ma.get_metric()
                transformation_name = ma.get_transformation_name()
                task_name = ma.get_kernel_ref().get_task_name()
                block_type = ma.get_block_ref().type
                dir = ma.get_dir()
                generation_time = ma.get_generation_time()
                sorted_blocks = self.convert_tuple_list_to_parsable_csv(
                    [(el.get_generic_instance_name(), val) for el, val in ma.sorted_blocks])
                sorted_kernels = self.convert_tuple_list_to_parsable_csv(
                    [(el.get_task_name(), val) for el, val in ma.sorted_kernels.items()])
                blk_instance_name = ma.get_block_ref().get_generic_instance_name()
                blk_type = ma.get_block_ref().type
                comm_comp = (ma.get_system_improvement_log())["comm_comp"]
                high_level_optimization = (ma.get_system_improvement_log())["high_level_optimization"]
                exact_optimization = (ma.get_system_improvement_log())["exact_optimization"]
                architectural_principle = (ma.get_system_improvement_log())["architectural_principle"]
                block_selection_time = ma.get_logs("block_selection_time")
                kernel_selection_time = ma.get_logs("kernel_selection_time")
                transformation_selection_time = ma.get_logs("transformation_selection_time")
                pickling_time = ma.get_logs("pickling_time")
                metric_selection_time = ma.get_logs("metric_selection_time")
                dir_selection_time = ma.get_logs("dir_selection_time")
                move_validity = ma.is_valid()
                ref_des_dist_to_goal_all = ma.get_logs("ref_des_dist_to_goal_all")
                ref_des_dist_to_goal_non_cost = ma.get_logs("ref_des_dist_to_goal_non_cost")
                #neighbouring_design_space_size = self.convert_dictionary_to_parsable_csv_with_semi_column(ma.get_design_space_size())
                neighbouring_design_space_size = sim_dp.get_neighbouring_design_space_size()
                workload = ma.get_logs("workload")
            else:  # happens at the very fist iteration
                pickling_time = 0
                sorted_metrics = ""
                metric = ""
                transformation_name = ""
                task_name = ""
                block_type = ""
                dir = ""
                generation_time = ''
                sorted_blocks = ''
                sorted_kernels = {}
                blk_instance_name = ''
                blk_type = ''
                comm_comp = ""
                high_level_optimization = ""
                exact_optimization = ""
                architectural_principle = ""
                block_selection_time = ""
                kernel_selection_time = ""
                metric_selection_time = ""
                dir_selection_time = ""
                transformation_selection_time = ""
                move_validity = ""
                ref_des_dist_to_goal_all = ""
                ref_des_dist_to_goal_non_cost = ""
                neighbouring_design_space_size = ""
                workload = ""


            sub_block_area_break_down = self.convert_dictionary_to_parsable_csv_with_underline(sim_dp.dp_stats.SOC_area_subtype_dict)
            block_area_break_down = self.convert_dictionary_to_parsable_csv_with_underline(sim_dp.dp_stats.SOC_area_dict)
            routing_complexity = sim_dp.dp_rep.get_hardware_graph().get_routing_complexity()
            area_non_dram = sim_dp.dp_stats.get_system_complex_area_stacked_dram()["non_dram"]
            area_dram = sim_dp.dp_stats.get_system_complex_area_stacked_dram()["dram"]
            simple_topology = sim_dp.dp_rep.get_hardware_graph().get_simplified_topology_code()
            channel_cnt = sim_dp.dp_rep.get_hardware_graph().get_number_of_channels()
            blk_cnt = sum([int(el) for el in simple_topology.split("_")])
            bus_cnt = [int(el) for el in simple_topology.split("_")][0]
            mem_cnt = [int(el) for el in simple_topology.split("_")][1]
            pe_cnt = [int(el) for el in simple_topology.split("_")][2]
            task_cnt = len(list(sim_dp.dp_rep.krnl_phase_present.keys()))
            #itr_depth_multiplied = sim_dp.dp_rep.get_iteration_number() * config.SA_depth + sim_dp.dp_rep.get_depth_number()

            self.total_iteration_cnt = ctr
            data = {
                    "data_number": ctr,
                    "iteration cnt" : self.total_iteration_cnt,
                    "exploration_plus_simulation_time" : sim_dp.get_exploration_and_simulation_approximate_time(),
                    #"phase_calculation_time": sim_dp.get_phase_calculation_time(),
                    # "task_update_time": sim_dp.get_task_update_time(),
                    #"phase_scheduling_time": sim_dp.get_phase_scheduling_time(),
                    "observed population number" : sim_dp.dp_rep.get_population_observed_number(),
                    "SA_total_depth": str(config.SA_depth),
                    "transformation_selection_mode": str(config.transformation_selection_mode),
                    "workload": workload,
                    "heuristic_type": config.heuristic_type,
                    "population generation cnt": sim_dp.dp_rep.get_population_generation_cnt(),
                    "simulation time" : sim_dp.dp_rep.get_simulation_time(),
                    "transformation generation time" : generation_time,
                    "metric selection time" :metric_selection_time,
                    "dir selection time" :dir_selection_time,
                    "kernel selection time" :kernel_selection_time,
                    "block selection time" : block_selection_time,
                    "transformation selection time" : transformation_selection_time,
                "design duplication time": pickling_time,
                "neighbour selection time": self.neighbour_selection_time,
                "dist_to_goal_all" : sim_dp.dp_stats.dist_to_goal(metrics_to_look_into=["area", "latency", "power", "cost"],
                                                                  mode="eliminate"),
                    "dist_to_goal_non_cost" : sim_dp.dp_stats.dist_to_goal(metrics_to_look_into=["area", "latency", "power"],
                                                                           mode="eliminate"),
                    "ref_des_dist_to_goal_all" : ref_des_dist_to_goal_all,
                    "ref_des_dist_to_goal_non_cost" : ref_des_dist_to_goal_non_cost,
                    "best_des_so_far_dist_to_goal_non_cost": self.so_far_best_sim_dp.dp_stats.dist_to_goal(metrics_to_look_into=["area", "latency", "power"],
                                                                           mode="eliminate"),
                    "best_des_so_far_dist_to_goal_all": self.so_far_best_sim_dp.dp_stats.dist_to_goal(metrics_to_look_into=["area", "latency", "power"],
                                                                           mode="eliminate"),
                    "best_des_so_far_area_non_dram": self.so_far_best_sim_dp.dp_stats.get_system_complex_area_stacked_dram()["non_dram"],
                    "best_des_so_far_area_dram": self.so_far_best_sim_dp.dp_stats.get_system_complex_area_stacked_dram()["dram"],
                    #"area_breakdown_subtype":self.convert_dictionary_to_parsable_csv_with_semi_column(sim_dp.dp_stats.SOC_area_subtype_dict.keys()),
                    #"best_des_so_far_area_breakdown_subtype":self.so_far_best_sim_dp.dp_stats.convert_dictionary_to_parsable_csv_with_semi_column(sim_dp.dp_stats.SOC_area_subtype_dict.keys()),
                    "system block count" : blk_cnt,
                    "system PE count" : pe_cnt,
                    "system bus count" : bus_cnt,
                    "system memory count" : mem_cnt,
                    "routing complexity" : routing_complexity,
                     "workload_set" : '_'.join(sim_dp.database.db_input.workload_tasks.keys()),
            "block_impact_sorted" : sorted_blocks,
                    "kernel_impact_sorted" : sorted_kernels,
                    "metric_impact_sorted" : sorted_metrics,
                    "transformation_metric" : metric,
                    "move validity" : move_validity,
                    "move name" : transformation_name,
                    "transformation_kernel" : task_name,
                    "transformation_block_name" : blk_instance_name,
                    "transformation_block_type" : blk_type,
                    "transformation_dir" : dir,
                    "comm_comp" : comm_comp,
                    "high level optimization name" : high_level_optimization,
                    "exact optimization name": exact_optimization,
                    "architectural principle" : architectural_principle,
                    "neighbouring design space size" : neighbouring_design_space_size,
                    "block_area_break_down":block_area_break_down,
                    "sub_block_area_break_down":sub_block_area_break_down,
                "task_cnt": task_cnt,
                "channel_cnt":channel_cnt,
                "area_dram":area_dram,
                "area_non_dram":area_non_dram
            }

            for metric in config.all_metrics:
                # convert dictionary to a parsable data
                data_ =  sim_dp.dp_stats.get_system_complex_metric(metric)
                if isinstance(data_, dict):
                    data__ =self.convert_dictionary_to_parsable_csv_with_semi_column(data_)
                else:
                    data__ = data_
                data[metric] = data__

                if metric in sim_dp.database.db_input.get_budget_dict("glass").keys():
                    # convert dictionary to a parsable rsult
                    data_ =  sim_dp.database.db_input.get_budget_dict("glass")[metric]
                    if isinstance(data_, dict):
                        data__ = self.convert_dictionary_to_parsable_csv_with_semi_column(data_)
                    else:
                        data__ = data_
                    data[metric +"_budget"] = data__

            for metric in config.all_metrics:
                # convert dictionary to a parsable data
                data_ = self.so_far_best_sim_dp.dp_stats.get_system_complex_metric(metric)
                if isinstance(data_, dict):
                    data__ = self.convert_dictionary_to_parsable_csv_with_semi_column(data_)
                else:
                    data__ = data_
                data["best_des_so_far_"+metric] = data__

            ctr +=1
            self.log_data_list.append(data)


    def sel_next_dp_for_moos(self, ex_sim_dp_dict, best_sim_dp_so_far, best_ex_dp_so_far, cur_temp):
        # convert to stats
        sim_dp_list = list(ex_sim_dp_dict.values())
        sim_dp_stat_list = [sim_dp.dp_stats for sim_dp in sim_dp_list]
        sim_stat_ex_dp_dict = {}
        for k, v in ex_sim_dp_dict.items():
            sim_stat_ex_dp_dict[v.dp_stats] = k


        # find the ones that fit the expanded budget (note that budget radius shrinks)
        selected_sim_dp, found_improved_solution = self.moos_greedy_design_selection(sim_stat_ex_dp_dict, sim_dp_stat_list,
                                                                            best_ex_dp_so_far, best_sim_dp_so_far.dp_stats,
                                                                            cur_temp)

        if not found_improved_solution:
            selected_sim_dp = self.so_far_best_sim_dp
            selected_ex_dp = self.so_far_best_ex_dp
        else:
            # extract the design
            for key, val in ex_sim_dp_dict.items():
                key.sanity_check()
                if val == selected_sim_dp:
                    selected_ex_dp = key
                    break

        # generate verification data
        if found_improved_solution and config.RUN_VERIFICATION_PER_IMPROVMENT:
            self.gen_verification_data(selected_sim_dp, selected_ex_dp)
        return selected_ex_dp, selected_sim_dp, found_improved_solution


    def greedy_for_moos(self, starting_ex_sim, moos_greedy_mode = "ctr"):
        this_itr_ex_sim_dp_dict_all = {}
        greedy_ctr_run = 0
        design_collected_ctr = 0

        if moos_greedy_mode == 'ctr':
            while greedy_ctr_run < config.MOOS_GREEDY_CTR_RUN and design_collected_ctr < config.DESIGN_COLLECTED_PER_GREEDY:
                this_itr_ex_sim_dp_dict = self.simple_SA()  # run simple simulated annealing
                self.cur_best_ex_dp, self.cur_best_sim_dp, found_improvement = self.sel_next_dp_for_moos(this_itr_ex_sim_dp_dict,
                                                                             self.so_far_best_sim_dp, self.so_far_best_ex_dp, 1)

                for ex, sim in this_itr_ex_sim_dp_dict.items():
                    this_itr_ex_sim_dp_dict_all[ex] = sim

                self.so_far_best_sim_dp = self.cur_best_sim_dp
                self.so_far_best_ex_dp = self.cur_best_ex_dp
                self.found_any_improvement  = self.found_any_improvement or found_improvement
                design_collected_ctr = len(this_itr_ex_sim_dp_dict_all)
                greedy_ctr_run +=1
        elif moos_greedy_mode == 'neighbour':
            found_improvement = True
            while found_improvement:
                this_itr_ex_sim_dp_dict = self.simple_SA()  # run simple simulated annealing
                self.cur_best_ex_dp, self.cur_best_sim_dp, found_improvement = self.sel_next_dp_for_moos(
                    this_itr_ex_sim_dp_dict,
                    self.so_far_best_sim_dp, self.so_far_best_ex_dp, 1)

                for ex, sim in this_itr_ex_sim_dp_dict.items():
                    this_itr_ex_sim_dp_dict_all[ex] = sim

                self.so_far_best_sim_dp = self.cur_best_sim_dp
                self.so_far_best_ex_dp = self.cur_best_ex_dp
                self.found_any_improvement = self.found_any_improvement or found_improvement
                design_collected_ctr = len(this_itr_ex_sim_dp_dict_all)
                greedy_ctr_run += 1
        elif moos_greedy_mode == "phv":
            phv_improvement = True
            hyper_volume_ref = [300, 2, 2]
            local_pareto = {}
            phv_so_far = 0
            while phv_improvement and greedy_ctr_run < config.MOOS_GREEDY_CTR_RUN:
                # run hill climbing
                this_itr_ex_sim_dp_dict = self.simple_SA()  # run simple simulated annealing
                # get best neighbour
                self.cur_best_ex_dp, self.cur_best_sim_dp, found_improvement = self.sel_next_dp_for_moos(
                    this_itr_ex_sim_dp_dict,
                    self.so_far_best_sim_dp, self.so_far_best_ex_dp, 1)

                # find best neighbour
                for ex, sim in this_itr_ex_sim_dp_dict.items():
                    if ex == self.cur_best_ex_dp:
                        best_neighbour_ex =   ex
                        best_neighbour_sim = sim
                        break

                for ex, sim in this_itr_ex_sim_dp_dict.items():
                    this_itr_ex_sim_dp_dict_all[ex] = sim

                # update the pareto with new best neighbour
                new_pareto = {}
                for ex, sim in local_pareto.items():
                    new_pareto[ex] = sim
                new_pareto[best_neighbour_ex] = best_neighbour_sim
                pareto_designs = self.get_pareto_designs(new_pareto)
                pareto_with_best_neighbour = self.evaluate_pareto(new_pareto, hyper_volume_ref)
                phv_improvement = pareto_with_best_neighbour > phv_so_far

                # if phv improved, add the neighbour to the local pareto

                if phv_improvement:
                    local_pareto = {}
                    for ex, sim in new_pareto.items():
                        local_pareto[ex] = sim
                    phv_so_far = pareto_with_best_neighbour

                greedy_ctr_run +=1

        #result = {self.cur_best_ex_dp: self.cur_best_sim_dp}
        result = this_itr_ex_sim_dp_dict_all
        return result


    def get_pareto_designs(self, ex_sim_designs):
        pareto_designs = {}
        point_list = []
        # iterate through the designs and generate points ([latency, power, area] tuple or points)
        for ex, sim in ex_sim_designs.items():
            point = []
            for metric_name in config.budgetted_metrics:
                for type, id in sim.dp_stats.get_designs_SOCs():
                    if metric_name == "latency":
                        metric_val = sum(list(sim.dp_stats.get_SOC_metric_value(metric_name, type, id).values()))
                        #metric_val = format(metric_val, ".10f")
                    else:
                        metric_val = sim.dp_stats.get_SOC_metric_value(metric_name, type, id)
                        #metric_val = format(metric_val, ".10f")


                    point.append(metric_val)
            point_list.append(point)

        # find the pareto points
        pareto_points = self.find_pareto_points(point_list)
        remove_list = []
        # extract the designs according to the pareto points
        for ex, sim in ex_sim_designs.items():
            point = []
            for metric_name in config.budgetted_metrics:
                for type, id in sim.dp_stats.get_designs_SOCs():
                    if metric_name == "latency":
                        metric_val = sum(list(sim.dp_stats.get_SOC_metric_value(metric_name, type, id).values()))
                    else:
                        metric_val = sim.dp_stats.get_SOC_metric_value(metric_name, type, id)

                    #metric_val = format(metric_val, ".10f")
                    point.append(metric_val)
            if point in pareto_points:
                pareto_points.remove(point)  # no double counting
                pareto_designs[ex] = sim
            else:
                remove_list.append(ex)

        for el in remove_list:
            del ex_sim_designs[el]

        if pareto_designs == {}:
            print("hmm there shoujld be a point in the pareto design")
        return pareto_designs

    def find_pareto_points(self, points):
        def is_pareto_efficient_dumb(costs):
            is_efficient = np.ones(costs.shape[0], dtype=bool)
            for i, c in enumerate(costs):
                is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(np.any(costs[i + 1:] > c, axis=1))
            return is_efficient

        # removing the duplicates (other wise we get wrong results for pareto front)
        points.sort()
        points =  list(k for k, _ in itertools.groupby(points))
        efficients = is_pareto_efficient_dumb(np.array(points))
        pareto_points_array = [points[idx] for idx, el in enumerate(efficients) if el]

        return pareto_points_array

        """
        pareto_points = []
        for el in pareto_points_array:
            list_ = []
            for el_ in el:
                list.append(el)
            pareto_points.append(list_)

        return pareto_points
        """

    def evaluate_pareto(self, pareto_ex_sim, ref):
        point_list = []
        for ex, sim in pareto_ex_sim.items():
            point = []
            for metric_name in config.budgetted_metrics:
                for type, id in sim.get_designs_SOCs():
                    if metric_name == "latency":
                        metric_val = sum(list(sim.dp_stats.get_SOC_metric_value(metric_name, type, id).values()))
                    else:
                        metric_val = sim.dp_stats.get_SOC_metric_value(metric_name, type, id)

                    #metric_val = format(metric_val, ".10f")
                    point.append(metric_val)
            point_list.append(point)


        hv = hypervolume(point_list)
        hv_value = hv.compute(ref)

        return hv_value

    # ------------------------------
    # Functionality:
    #      Explore the design space.
    # Variables
    #       it uses the config parameters that are used to instantiate the object.
    # ------------------------------
    def explore_ds_with_moos(self):
        #gc.DEBUG_SAVEALL = True
        self.so_far_best_ex_dp = self.init_ex_dp
        self.so_far_best_sim_dp = self.eval_design(self.so_far_best_ex_dp, self.database)
        self.init_sim_dp = self.eval_design(self.so_far_best_ex_dp, self.database)

        # visualize/checkpoint/PA generation
        vis_hardware.vis_hardware(self.so_far_best_sim_dp.get_dp_rep())
        if config.RUN_VERIFICATION_PER_GEN or config.RUN_VERIFICATION_PER_IMPROVMENT or config.RUN_VERIFICATION_PER_NEW_CONFIG:
            self.gen_verification_data(self.so_far_best_sim_dp, self.so_far_best_ex_dp)


        #num_of_workloads = len(self.so_far_best_sim_dp.dp_stats.database.get_workloads_last_task().values())
        # initializing the tree
        self.pareto_global = {}
        self.pareto_global [self.so_far_best_ex_dp] = self.so_far_best_sim_dp
        hyper_volume_ref = [300,2,2]

        pareto_global_child_evaluation = self.evaluate_pareto(self.pareto_global, hyper_volume_ref)
        root_node = self.moos_tree.get_root()
        root_node.update_evaluation(pareto_global_child_evaluation)
        best_leaf = self.moos_tree.get_root()

        des_per_iteration = [0]
        start = True
        cur_temp = config.annealing_max_temp

        pareto_global_init = {}
        pareto_global_child = {}
        should_terminate = False
        reason_to_termiante = ""
        ctr = 0
        while not should_terminate:
            # get pareto designs
            self.pareto_global = self.get_pareto_designs(self.pareto_global)
            # expand the tree
            best_leaf = self.moos_tree.find_node_to_expand()
            best_leaf_evaluation = best_leaf.get_evaluation()
            expanded = self.moos_tree.expand(best_leaf)
            ctr +=1
            # if expansion failed, terminate
            # usually happens when the intervals are too small
            if not expanded:
                should_terminate, reason_to_terminate = True, "no_interval_for_moos"
            ctr +=1
            # populate the pareto (local pareto)
            pareto_global_init.clear()
            for ex, sim in self.pareto_global.items():
                pareto_global_init[ex] = sim

            # iterate through the children and run greedy heuristic
            for name, child in best_leaf.get_children().items():
                if should_terminate:
                    continue

                if name == "center":
                    pareto_global_child_evaluation = best_leaf_evaluation
                    child.update_evaluation(pareto_global_child_evaluation)
                    continue

                # populate the pareto front with pareto global
                pareto_global_child.clear()
                for ex, sim in pareto_global_init.items():
                    pareto_global_child[ex] = sim
                lambdas = child.get_lambdas()
                self.cur_best_ex_dp, self.cur_best_sim_dp = self.sel_start_dp_moos(pareto_global_child,
                                                                         self.so_far_best_sim_dp, self.so_far_best_ex_dp, lambdas)

                # use the follow as a the starting point for greedy heuristic
                self.so_far_best_ex_dp  = self.cur_best_ex_dp
                self.so_far_best_sim_dp  = self.cur_best_sim_dp
                #this_itr_ex_sim_dp_dict = {self.so_far_best_ex_dp:  self.so_far_best_sim_dp}
                this_itr_ex_sim_dp_dict = self.greedy_for_moos(self.so_far_best_ex_dp, config.moos_greedy_mode)   # run simple simulated annealing


                self.total_iteration_ctr += len(list(this_itr_ex_sim_dp_dict.keys()))

                """
                # collect profiling information about moves and designs generated
                if config.VIS_MOVE_TRAIL and (self.population_generation_cnt% config.vis_reg_ctr_threshold) == 0 and len(self.des_trail_list) > 0:
                    plot.des_trail_plot(self.des_trail_list, self.move_profile, des_per_iteration)
                    plot.move_profile_plot(self.move_profile)
                """

                # get new pareto design and merge, and evaluate (update the tree)
                self.log_data(this_itr_ex_sim_dp_dict)
                self.collect_stats(this_itr_ex_sim_dp_dict)
                pareto_designs = self.get_pareto_designs(this_itr_ex_sim_dp_dict)
                pareto_global_child.update(pareto_designs)
                pareto_global_child = self.get_pareto_designs(pareto_global_child)
                pareto_global_child_evaluation = self.evaluate_pareto(pareto_global_child, hyper_volume_ref)
                print("pareto evluation" + str(pareto_global_child_evaluation))
                child.update_evaluation(pareto_global_child_evaluation)

                # update pareto global
                for ex, sim in pareto_global_child.items():
                    self.pareto_global[ex] = sim
                #self.pareto_global.update(pareto_global_child)

                """
                if config.VIS_GR_PER_ITR and (self.population_generation_cnt% config.vis_reg_ctr_threshold) == 0:
                    vis_hardware.vis_hardware(self.cur_best_sim_dp.get_dp_rep())
                """
                # collect statistics about the design
                gc.collect()

                # update and check for termination
                print("memory usage ===================================== " +str(psutil.virtual_memory()))
                # check terminattion status
                should_terminate, reason_to_terminate = self.update_ctrs()
                mem = psutil.virtual_memory()
                mem_used = int(mem.percent)
                if mem_used > config.out_of_memory_percentage:
                    should_terminate, reason_to_terminate = True, "out_of_memory"

            if should_terminate:
                # get the best design
                self.cur_best_ex_dp, self.cur_best_sim_dp = self.sel_next_dp(self.pareto_global,
                                                                             self.so_far_best_sim_dp,
                                                                             self.so_far_best_ex_dp, config.annealing_max_temp)
                self.so_far_best_ex_dp  = self.cur_best_ex_dp
                self.so_far_best_sim_dp  = self.cur_best_sim_dp

                print("reason to terminate is:" + reason_to_terminate)
                vis_hardware.vis_hardware(self.cur_best_sim_dp.get_dp_rep())
                if not (self.last_des_trail == None):
                    if self.last_des_trail == None:
                        self.last_des_trail = (
                        copy.deepcopy(self.so_far_best_sim_dp), copy.deepcopy(self.so_far_best_sim_dp))
                        # self.last_des_trail = (cPickle.loads(cPickle.dumps(self.so_far_best_sim_dp, -1)),cPickle.loads(cPickle.dumps(self.so_far_best_sim_dp)))
                    else:
                        self.des_trail_list.append(self.last_des_trail)
                if not (self.last_move == None):
                    self.move_profile.append(self.last_move)

                if config.VIS_MOVE_TRAIL:
                    plot.des_trail_plot(self.des_trail_list, self.move_profile, des_per_iteration)
                    plot.move_profile_plot(self.move_profile)
                self.reason_to_terminate = reason_to_terminate

                return

            print(" >>>>> des" + " latency:" + str(self.so_far_best_sim_dp.dp_stats.get_system_complex_metric("latency")))

            """ 
            stat_result = self.so_far_best_sim_dp.dp_stats
            if stat_result.fits_budget(1):
                should_terminate = True
                reason_to_terminate = "met the budget"
            elif self.population_generation_cnt > self.TOTAL_RUN_THRESHOLD:
                should_terminate = True
                reason_to_terminate = "exploration (total itr_ctr) iteration threshold reached"

            """


    def explore_simple_greedy_one_sample(self, orig_ex_dp, mode="random"):
        orig_sim_dp = self.eval_design(self.init_ex_dp, self.database)
        des_tup = [orig_ex_dp, orig_sim_dp]
        des_tup_new, possible_des_cnt = self.gen_neigh_and_eval(des_tup)

    # ------------------------------
    # Functionality:
    #      Explore the design space.
    # Variables
    #       it uses the config parameters that are used to instantiate the object.
    # ------------------------------
    def explore_ds(self):
        self.so_far_best_ex_dp = self.init_ex_dp
        self.so_far_best_sim_dp = self.eval_design(self.so_far_best_ex_dp, self.database)
        self.init_sim_dp = self.eval_design(self.so_far_best_ex_dp, self.database)

        # visualize/checkpoint/PA generation
        vis_hardware.vis_hardware(self.so_far_best_sim_dp.get_dp_rep())
        if config.RUN_VERIFICATION_PER_GEN or config.RUN_VERIFICATION_PER_IMPROVMENT or config.RUN_VERIFICATION_PER_NEW_CONFIG:
            self.gen_verification_data(self.so_far_best_sim_dp, self.so_far_best_ex_dp)

        des_per_iteration = [0]
        start = True
        cur_temp = config.annealing_max_temp

        while True:
            this_itr_ex_sim_dp_dict = self.simple_SA()   # run simple simulated annealing
            self.total_iteration_ctr += len(list(this_itr_ex_sim_dp_dict.keys()))

            # collect profiling information about moves and designs generated
            if config.VIS_MOVE_TRAIL and (self.population_generation_cnt% config.vis_reg_ctr_threshold) == 0 and len(self.des_trail_list) > 0:
                plot.des_trail_plot(self.des_trail_list, self.move_profile, des_per_iteration)
                plot.move_profile_plot(self.move_profile)

            # select the next best design
            t1 = time.time()
            self.cur_best_ex_dp, self.cur_best_sim_dp = self.sel_next_dp(this_itr_ex_sim_dp_dict,
                                                                         self.so_far_best_sim_dp, self.so_far_best_ex_dp, cur_temp)
            t2 = time.time()
            self.neighbour_selection_time = t2-t1
            self.log_data(this_itr_ex_sim_dp_dict)
            print("-------:):):):):)----------")
            print("Best design's latency: " + str(self.cur_best_sim_dp.dp_stats.get_system_complex_metric("latency")))
            print("Best design's power: " + str(self.cur_best_sim_dp.dp_stats.get_system_complex_metric("power")))
            print("Best design's sub area: " + str(self.cur_best_sim_dp.dp_stats.get_system_complex_area_stacked_dram()))

            if  not self.cur_best_sim_dp.move_applied == None:
                self.cur_best_sim_dp.move_applied.print_info()

            if config.VIS_GR_PER_ITR and (self.population_generation_cnt% config.vis_reg_ctr_threshold) == 0:
                vis_hardware.vis_hardware(self.cur_best_sim_dp.get_dp_rep())

            # collect statistics about the design
            self.collect_stats(this_itr_ex_sim_dp_dict)

            # determine if the design has met the budget, if so, terminate
            mem = psutil.virtual_memory()
            mem_used = int(mem.percent)
            print("memory usage ===================================== " + str(psutil.virtual_memory()))
            if mem_used > config.out_of_memory_percentage:
                should_terminate, reason_to_terminate = True, "out_of_memory"
            else:
                should_terminate, reason_to_terminate = self.update_ctrs()

            if should_terminate:
                print("reason to terminate is:" + reason_to_terminate)
                vis_hardware.vis_hardware(self.cur_best_sim_dp.get_dp_rep())
                if not (self.last_des_trail == None):
                    if self.last_des_trail == None:
                        self.last_des_trail = (copy.deepcopy(self.so_far_best_sim_dp), copy.deepcopy(self.so_far_best_sim_dp))
                        #self.last_des_trail = (cPickle.loads(cPickle.dumps(self.so_far_best_sim_dp, -1)),cPickle.loads(cPickle.dumps(self.so_far_best_sim_dp)))
                    else:
                        self.des_trail_list.append(self.last_des_trail)
                if not (self.last_move == None):
                    self.move_profile.append(self.last_move)

                if config.VIS_MOVE_TRAIL:
                    plot.des_trail_plot(self.des_trail_list, self.move_profile, des_per_iteration)
                    plot.move_profile_plot(self.move_profile)
                self.reason_to_terminate = reason_to_terminate
                return
            cur_temp -= config.annealing_temp_dec
            self.vis_move_trail_ctr += 1

    # ------------------------------
    # Functionality:
    #       generating plots for data analysis
    # -----------------------------
    def plot_data(self):
        iterations = [iter*config.num_neighs_to_try for iter in self.design_itr]
        if config.DATA_DELIVEYRY == "obfuscate":
            plot.scatter_plot(iterations, [area/self.area_explored[0] for area in self.area_explored], ("iteration", "area"), self.database)
            plot.scatter_plot(iterations, [power/self.power_explored[0] for power in self.power_explored], ("iteration", "power"), self.database)
            latency_explored_normalized = [el/self.latency_explored[0] for el in self.latency_explored]
            plot.scatter_plot(iterations, latency_explored_normalized, ("iteration", "latency"), self.database)
        else:
            plot.scatter_plot(iterations, [1000000*area/self.area_explored[0] for area in self.area_explored], ("iteration", "area"), self.database)
            plot.scatter_plot(iterations, [1000*power/self.power_explored[0] for power in self.power_explored], ("iteration", "power"), self.database)
            plot.scatter_plot(iterations, self.latency_explored/self.latency_explored[0], ("iteration", "latency"), self.database)

    # ------------------------------
    # Functionality:
    #       report the data collected in a humanly readable way.
    # Variables:
    #      explorations_start_time: to exploration start time used to determine the end-to-end exploration time.
    # -----------------------------
    def report(self, exploration_start_time):
        exploration_end_time = time.time()
        total_sim_time = exploration_end_time - exploration_start_time
        print("*********************************")
        print("------- Best Designs Metrics ----")
        print("*********************************")
        print("Best design's latency: " + str(self.so_far_best_sim_dp.dp_stats.get_system_complex_metric("latency")) + \
              ", ---- time budget:" + str(config.budgets_dict["glass"]["latency"]))
        print("Best design's thermal power: " + str(self.so_far_best_sim_dp.dp_stats.get_system_complex_metric("power"))+
              ", ---- thermal power budget:" + str(config.budgets_dict["glass"]["power"]))
        print("Best design's area: " + str(self.so_far_best_sim_dp.dp_stats.get_system_complex_metric("area")) + \
              ", ---- area budget:" + str(config.budgets_dict["glass"]["area"]))
        print("Best design's energy: " + str(self.so_far_best_sim_dp.dp_stats.get_system_complex_metric("energy")))
        print("*********************************")
        print("------- DSE performance --------")
        print("*********************************")
        print("Initial design's latency: " + str(self.init_sim_dp.dp_stats.get_system_complex_metric("latency")))
        print("Speed up: " + str(self.init_sim_dp.dp_stats.get_system_complex_metric("latency")/self.so_far_best_sim_dp.dp_stats.get_system_complex_metric("latency")))
        print("Number of design points examined:" + str(self.population_generation_cnt*config.num_neighs_to_try))
        print("Time spent per design point:" + str(total_sim_time/(self.population_generation_cnt*config.num_neighs_to_try)))
        print("The design meet the latency requirement: " + str(self.so_far_best_sim_dp.dp_stats.get_system_complex_metric("latency") < config.objective_budget))
        vis_hardware.vis_hardware(self.so_far_best_ex_dp)
        if config.VIS_FINAL_RES:
            vis_hardware.vis_hardware(self.so_far_best_ex_dp, config.hw_graphing_mode)

        # write the output
        home_dir = os.getcwd()
        FARSI_result_dir = config.FARSI_result_dir
        FARSI_result_directory_path = os.path.join(home_dir, 'data_collection/data/', FARSI_result_dir)
        output_file_verbose = os.path.join(FARSI_result_directory_path, config.FARSI_outputfile_prefix_verbose +".txt")
        output_file_minimal = os.path.join(FARSI_result_directory_path, config.FARSI_outputfile_prefix_minimal +".csv")

        # minimal output
        output_fh_minimal = open(output_file_minimal, "w")
        for metric in config.all_metrics:
            output_fh_minimal.write(metric+ ",")
        output_fh_minimal.write("\n")
        for metric in config.all_metrics:
            output_fh_minimal.write(str(self.so_far_best_sim_dp.dp_stats.get_system_complex_metric(metric))+ ",")
        output_fh_minimal.close()

        # verbose
        output_fh_verbose = open(output_file_verbose, "w")
        output_fh_verbose.write("iter_cnt" + ": ")
        for el in range(0, len(self.power_explored)):
            output_fh_verbose.write(str(el) +",")

        output_fh_verbose.write("\npower" + ": ")
        for el in self.power_explored:
            output_fh_verbose.write(str(el) +",")

        output_fh_verbose.write("\nlatency" + ": ")
        for el in self.latency_explored:
            output_fh_verbose.write(str(el) +",")
        output_fh_verbose.write("\narea" + ": ")
        for el in self.area_explored:
            output_fh_verbose.write(str(el) +",")

        output_fh_verbose.close()

    # ------------------------------
    # Functionality:
    #      collect the profiling information for all the design generated  by the explorer.  For data analysis.
    # Variables:
    #     ex_sim_dp_dict: example_design_simulated_design_dictionary. A dictionary containing the
    #     (example_design, simulated_design) tuple.
    # -----------------------------
    def collect_stats(self, ex_sim_dp_dict):
        for sim_dp in ex_sim_dp_dict.values():
            self.area_explored.append(sim_dp.dp_stats.get_system_complex_metric("area"))
            self.power_explored.append(sim_dp.dp_stats.get_system_complex_metric("power"))
            self.latency_explored.append(sim_dp.dp_stats.get_system_complex_metric("latency"))
            self.design_itr.append(self.population_generation_cnt)

    # ------------------------------
    # Functionality:
    #       calculate the budget coefficients. This is used for simulated annealing purposes.
    #       Concretely, first we use relax budgets to allow wider exploration, and then
    #       incrementally tighten the budget to direct the explorer more toward the goal.
    # ------------------------------
    def calc_budget_coeff(self):
        self.budget_coeff = int(((self.TOTAL_RUN_THRESHOLD - self.population_generation_cnt)/self.coeff_slice_size) + 1)

    def reset_ctrs(self):
        should_terminate = False
        reason_to_terminate = ""
        self.fitted_budget_ctr = 0
        self.krnels_not_to_consider = []
        self.des_stag_ctr = 0
        self.krnel_stagnation_ctr = 0
        self.krnel_rnk_to_consider = 0
        self.cleanup_ctr  = 0
        self.des_stag_ctr = 0
        self.recently_seen_design_ctr = 0
        self.counters.reset()
        self.moos_tree = moosTreeModel(config.budgetted_metrics)  # only used for moos heuristic
        self.found_any_improvement = False



    # ------------------------------
    # Functionality:
    #       Update the counters to determine the exploration (navigation heuristic) control path to follow.
    # ------------------------------
    def update_ctrs(self):
        should_terminate = False
        reason_to_terminate = ""

        self.so_far_best_ex_dp = self.cur_best_ex_dp
        self.so_far_best_sim_dp = self.cur_best_sim_dp

        self.population_generation_cnt += 1
        stat_result = self.so_far_best_sim_dp.dp_stats

        tasks_not_meeting_budget = [el.get_task_name() for el in self.filter_in_kernels_meeting_budget("", self.so_far_best_sim_dp)]
        tsks_left_to_optimize = list(set(tasks_not_meeting_budget) - set(self.krnels_not_to_consider))

        if stat_result.fits_budget(1) :
            config.VIS_GR_PER_GEN = True  # visualize the graph per design point generation
            config.VIS_SIM_PER_GEN = True  # if true, we visualize the simulation progression
            self.fitted_budget_ctr +=1
        if (self.fitted_budget_ctr > config.fitted_budget_ctr_threshold):
            reason_to_terminate = "met the budget"
            should_terminate = True
        elif self.des_stag_ctr > self.DES_STAG_THRESHOLD:
            reason_to_terminate = "des_stag_ctr exceeded"
            should_terminate = True
        elif len(self.krnels_not_to_consider) >= (len(self.so_far_best_sim_dp.get_kernels()) - len(self.so_far_best_sim_dp.get_dummy_tasks())):
            if stat_result.fits_budget(1):
                reason_to_terminate = "met the budget"
            else:
                reason_to_terminate = "all kernels already targeted without improvement"
            should_terminate = True
        elif len(tsks_left_to_optimize) == 0:
            if stat_result.fits_budget(1):
                reason_to_terminate = "met the budget"
            else:
                reason_to_terminate = "all kernels already targeted without improvement"
            should_terminate = True
        elif self.total_iteration_ctr > self.TOTAL_RUN_THRESHOLD:
            if stat_result.fits_budget(1):
                reason_to_terminate = "met the budget"
            else:
                reason_to_terminate = "exploration (total itr_ctr) iteration threshold reached"
            should_terminate = True

        self.counters.update(self.krnel_rnk_to_consider, self.krnel_stagnation_ctr, self.fitted_budget_ctr, self.des_stag_ctr,
                             self.krnels_not_to_consider, self.population_generation_cnt, self.found_any_improvement, self.total_iteration_ctr)

        print(">>>>> total iteration count is: " + str(self.total_iteration_ctr))
        return should_terminate, reason_to_terminate


class moosTreeNode:
    def __init__(self, k_intervals):
        self.k_ins = k_intervals
        self.children = {}
        self.evaluation = "None"

    def update_evaluation(self, evaluation):
        self.evaluation = evaluation

    def get_evaluation(self):
        if self.evaluation == "None":
            print("must populate the evluation first")
            exit(0)
        return self.evaluation

    def get_k_ins(self):
        return self.k_ins

    def get_interval(self, metric):
        return self.get_k_ins()[metric]


    def get_interval_length(self, metric):
        assert(metric in self.k_ins.keys())
        interval = self.k_ins[metric]
        length = interval[1] - interval[0]
        if (length <=0):
            print("length" + str(length))
            print("interval is:")
            print(str(self.k_ins[metric]))
        assert(length > 0)
        return length

    def longest_dimension_name(self):
        key =  list(self.get_k_ins().keys())[0]
        max_dimension = self.get_interval_length(key)
        max_key = key
        #print("max_dimentions" + str(max_dimension))

        for k,v in self.get_k_ins().items():
            if self.get_interval_length(k) > max_dimension:
                max_dimension = self.get_interval_length(k)
                max_key = k

        return max_key

    def update_interval(self, key, interval):
        self.k_ins[key] = interval

    def add_children(self, left_children_, center_children_, right_children_):
        self.children["left"]= left_children_
        self.children["center"] = center_children_
        self.children["right"] = right_children_

    def get_children_with_position(self, position):
        return self.children[position]

    def get_children(self):
        return self.children

    def get_lambdas(self):
        lambdas = {}
        for metric_name, val in self.k_ins.items():
            lambdas[metric_name]  = (val[1] - val[0])/2
        return lambdas


class moosTreeModel:
    def __init__(self, metric_names):
        max = Decimal(1000000000)
        min = Decimal(0)
        node_val = []
        k_ins = {}
        for el in metric_names:
            k_ins[el] = [min, max]
        self.root = moosTreeNode(k_ins)

    def get_root(self):
        return self.root

    def get_leaves_with_depth(self):
        def get_leaves_helper(node, depth):
            result = []
            if node.get_children()  == {}:
                result = [(node, depth)]
            else:
                for position, child in node.get_children().items():
                    child_result = get_leaves_helper(child, depth+1)
                    result.extend(child_result)
            return result

        leaves_depth = get_leaves_helper(self.root, 0)
        return leaves_depth

    def expand(self, node):
        # initializing the longest
        longest_dimension_key = node.longest_dimension_name()
        longest_dimension_interval = node.get_interval(longest_dimension_key)
        longest_intr_min = min(longest_dimension_interval)
        longest_intr_max = max(longest_dimension_interval)
        longest_dimension_incr = Decimal(longest_intr_max - longest_intr_min)/3

        child_center = moosTreeNode(copy.deepcopy(node.get_k_ins()))
        child_left = moosTreeNode(copy.deepcopy(node.get_k_ins()))
        child_right = moosTreeNode(copy.deepcopy(node.get_k_ins()))

        if (longest_intr_min - longest_intr_min + 1*longest_dimension_incr == 0):
            print("this shouldn't happen. intervals should be the same")
            print("intrval" + str(longest_dimension_interval))
            print("incr" + str(longest_dimension_incr))
            print("min" +str(longest_intr_min))
            print("max" + str(longest_intr_max))
            print("first upper" + str(longest_intr_min + 1*longest_dimension_incr))
            print("second upper"+ str(longest_intr_min + 2 * longest_dimension_incr))
            return False

        if (longest_intr_min + 1 * longest_dimension_incr - longest_intr_min + 2 * longest_dimension_incr == 0):
            print("this shouldn't happen. intervals should be the same")
            print(longest_dimension_interval)
            print(longest_dimension_incr)
            print(longest_intr_min)
            print(longest_intr_min + 1*longest_dimension_incr)
            print(longest_intr_min + 2 * longest_dimension_incr)
            print(longest_intr_max)
            return False

        if (longest_intr_min + 2 * longest_dimension_incr -  longest_intr_max == 0):
            print("this shouldn't happen. intervals should be the same")
            print(longest_dimension_interval)
            print(longest_dimension_incr)
            print(longest_intr_min)
            print(longest_intr_min + 1*longest_dimension_incr)
            print(longest_intr_min + 2 * longest_dimension_incr)
            print(longest_intr_max)
            return False


        child_left.update_interval(longest_dimension_key, [longest_intr_min, longest_intr_min + 1*longest_dimension_incr])
        child_center.update_interval(longest_dimension_key, [longest_intr_min + 1*longest_dimension_incr, longest_intr_min + 2*longest_dimension_incr])
        child_right.update_interval(longest_dimension_key, [longest_intr_min + 2*longest_dimension_incr, longest_intr_max])
        node.add_children(child_left, child_center, child_right)
        return True

    def find_node_to_expand(self):
        node_star_list = []  # selected node
        leaves_with_depth = self.get_leaves_with_depth()

        # find max depth
        max_depth = max([depth for leaf,depth in leaves_with_depth])

        # split leaves to max and non max depth
        leaves_with_max_depth = [leaf for leaf,depth in leaves_with_depth if depth == max_depth]
        leaves_with_non_max_depth = [leaf for leaf,depth in leaves_with_depth if not depth == max_depth]

        # select node star
        for max_leaf in leaves_with_max_depth:
            leaf_is_better_list = []
            for non_max_leaf in leaves_with_non_max_depth:
                leaf_is_better_list.append(max_leaf.get_evaluation() >= non_max_leaf.get_evaluation())
            if all(leaf_is_better_list):
                node_star_list.append(max_leaf)


        best_node = node_star_list[0]
        for node in node_star_list:
            if node.get_evaluation() > best_node.get_evaluation():
                best_node = node
        """
        # for debugging. delete later
        if (len(node_star_list) == 0):
            for max_leaf in leaves_with_max_depth:
                leaf_is_better_list = []
                for non_max_leaf in leaves_with_non_max_depth:
                    leaf_is_better_list.append(max_leaf.get_evaluation() >= non_max_leaf.get_evaluation())
                if all(leaf_is_better_list):
                    node_star_list.append(max_leaf)

            if not (len(node_star_list) == 1):
                print("something went wrong")
        """
        return best_node











