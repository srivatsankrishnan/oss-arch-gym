#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

from sympy.functions.combinatorial import numbers
from sympy import factorial
import math
import itertools
from copy import *
import time
import numpy as np
import operator
import collections

# ------------------------------
# Functionality:
#       calculate stirling values, ie., the number of ways to partition a set. For mathematical understanding refer to
#       https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind
# Variables:
#       n, k are both  stirling inputs. refer to:
#       https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind
# ------------------------------
# n: balls, k: bins
def calc_stirling(n, k):
    # multiply by k! if you want the boxes to at least contain 1 value
    return numbers.stirling(n, k, d=None, kind=2, signed=False)


# ------------------------------
# Functionality:
#       calculate the migration cardinality
#       where we can migrate such that tasks can be distributed accross n+1 blocks when we introduce
#       n new blocks (with the restriction that each block needs to have at least one task on it)
# Variables:
#       num_tasks: number of tasks within the workload.
#       num_blcks_to_split_to: number of blocks to maps the tasks to.
# ------------------------------
def calc_mig_comb_cnt(num_task, num_blcks_to_split_to):
    return factorial(num_blcks_to_split_to) * calc_stirling(num_task, num_blcks_to_split_to)


def calc_mig_comb_idenitical_blocks_cnt(num_task, num_blcks_to_split_to):
    return  calc_stirling(num_task, num_blcks_to_split_to)

# ------------------------------
# Functionality:
#       calculates the upper bound combination associated with the reduced contention options
# Variables:
#       num_tasks: number of tasks within the workload.
#       num_blcks_to_split_to: number of blocks to maps the tasks to.
#       bocks_to_choose_from:  total number of blocks that we can choose our num_blcks_to_split_to set from.
# ------------------------------
def calc_red_cont_up_comb_cnt(num_tasks, num_blcks_to_split_to, blocks_to_choose_from):
    allocation_cardinality = (num_blcks_to_split_to - 1)**blocks_to_choose_from
    # the reason that this is upper bound is because if a hardware configuration
    # uses two blocks of the same kind, then migration can results in a setup that has been already seen.
    return allocation_cardinality * calc_mig_comb_cnt(num_tasks, num_blcks_to_split_to)

# ------------------------------
# Functionality:
#       give tne number of draws from a population, what is the statistical expected coverage value.
# Variables:
#       population_size: statistical population cardinality.
#       num_of_draws_threshold: bounding the number of samples drawn from the population.
# ------------------------------
def calc_coverage_exepctation_value(population_size, num_of_draws):
    expected_value_of_coverage = population_size * (1 - ((population_size-1)/population_size)**num_of_draws)
    return float(expected_value_of_coverage)

# ------------------------------
# Functionality:
#       Find how many samples we need to draw from the population to achieve a certain coverage. Note that
#       we use the expected value of the coverage, that is, on AVERAGE, what the coverage is if we make certain
#       num of draws.
# Variables:
#       population_size: statistical population cardinality.
#       desired_coverage: which percentage of the population size we'd like to cover.
#       num_of_draws_threshold: bounding the number of samples drawn from the population.
#       num_of_draw_incr: incrementally increase num_of-draws_threshold to meet the desired coverage.
# ------------------------------
def find_num_draws_to_satisfy_coverage(population_size, desired_coverage, num_of_draws_threshold, num_of_draw_incr):
    coverage_satisfied = False
    for num_of_draws in range(0, num_of_draws_threshold, num_of_draw_incr):
        coverage_expectation = calc_coverage_exepctation_value(population_size, num_of_draws)
        if coverage_expectation >= desired_coverage:
            coverage_satisfied = True
            break
    return coverage_satisfied, coverage_expectation, num_of_draws

# ------------------------------
# Functionality:
#       simple test for sanity check.
# ------------------------------
def simple_test():
    pop_size = calc_red_cont_up_comb_cnt(30, 2, 5)
    coverage_percentage = 0.50
    x, y, z  = find_num_draws_to_satisfy_coverage(pop_size, coverage_percentage*pop_size, 4000, 500)
    print(x)

#  we don't consider the infeasibility of allocating
# extra blocks despite of no-parallelism
def system_variation_count_1(gen_config):
    MAX_TASK_CNT = gen_config["MAX_TASK_CNT"]
    DB_MAX_PE_CNT = gen_config["DB_MAX_PE_CNT"]
    DB_MAX_MEM_CNT = gen_config["DB_MAX_MEM_CNT"]
    DB_MAX_BUS_CNT = gen_config["DB_MAX_BUS_CNT"]

    # assuming that we have 5 different tasks and hence (can have up to 5 different blocks).
    # we'd like to know how many different migration/allocation combinations are out there.
    # Assumptions:
    #           PE's are identical.
    #           Buses are identical
    #           memory is ignored for now

    # topologies
    MAX_PE_CNT = MAX_TASK_CNT
    MAX_BUS_CNT = MAX_TASK_CNT
    task_cnt = MAX_TASK_CNT
    system_variations = 0
    for bus_cnt in range(1, MAX_BUS_CNT+1):
        for pe_cnt in range(bus_cnt, MAX_PE_CNT+1):
            for mem_cnt in range(bus_cnt, pe_cnt+ 1):
                # first calculate the topological variations (split)
                topo_incr_1 = calc_stirling(pe_cnt, bus_cnt)*factorial(bus_cnt)  # at least one pe per bus
                                                                            # factorial is used because we
                                                                            # assume distinct buses because the
                                                                            # the relative positioning of the buses
                                                                            # impacts the topology
                topo_incr_2 = calc_stirling(mem_cnt, bus_cnt)*factorial(bus_cnt)  # at least one memory for
                                                                                  # each bus. Note that this estimation
                                                                                  # is a bit conservative as
                                                                                  # scenarios where number of mems
                                                                                  # exceeds the number of pes connected
                                                                                  # to a bus are not really reasonable
                                                                                  # however, they are considered here.

                # then calculate mapping (migrate)
                mapping = calc_stirling(task_cnt, pe_cnt)*factorial(pe_cnt)
                # then calculate customization (swap)
                swap = math.pow(2, (DB_MAX_BUS_CNT + DB_MAX_PE_CNT + DB_MAX_MEM_CNT))
                system_variations += topo_incr_1*topo_incr_2*mapping*swap

    #print("number of system variations: " + str(float(system_variations)))
    #print("exhaustive simulation time (hours): " + str(float(system_variations)/(20*3600)))
    return system_variations
    #print("{:e}".format(system_variations))


class SYSTEM_():
    def __init__(self):
        self.bus_cnt = 0
        self.pe_cnt = 0
        self.mem_cnt = 0
        self.similar_system_cnt = 0
        self.task_cnt = 0
        self.par_task_cnt = 0
        self.pe_set = []  # lay buses as the reference and decorate with mem
        self.mem_set = []
        self.mapping_variation_cnt = 0
        self.PE_list = []
        self.MEM_list = []
        self.BUS_list = []
        self.BUS_PE_list = {} # dictionary of bus index and the PEs hanging from them. buses are indexed based on BUS_list
        self.BUS_MEM_list = {}  # dictionary of bus index and the MEMs hanging from them. buses are indexed based on BUS_list

    def set_bus_cnt(self, bus_cnt):
        self.bus_cnt = bus_cnt

    def set_pe_cnt(self, pe_cnt):
        self.pe_cnt = pe_cnt

    def set_mem_cnt(self, mem_cnt):
        self.mem_cnt = mem_cnt

    def get_bus_cnt(self):
        return self.bus_cnt

    def get_pe_cnt(self):
        return self.pe_cnt

    def get_mem_cnt(self):
        return self.mem_cnt

    def append_pe_set(self, pe_cnt):
        self.pe_set.append(pe_cnt)

    def set_pe_set(self, pe_set):
        self.pe_set = pe_set
        self.pe_cnt = sum(self.get_pe_set())

    def get_mem_set(self):
        return self.mem_set

    def set_mem_set(self, mem_set):
        self.mem_set = mem_set
        self.mem_cnt = sum(self.get_mem_set())

    def get_pe_set(self):
        return self.pe_set

    def set_task_cnt(self, task_cnt):
        self.task_cnt = task_cnt

    def set_par_task_cnt(self, par_task_cnt):
        self.par_task_cnt = par_task_cnt

    def parallelism_check(self, gen_config):
        MAX_TASK_CNT = gen_config["DB_MAX_TASK_CNT"]
        DB_MAX_PE_CNT = gen_config["DB_MAX_PE_CNT"]

        par_task_cnt = self.par_task_cnt
        for pe_cnt in self.get_pe_set():
            if pe_cnt - DB_MAX_PE_CNT > 0:
                par_task_cnt -= (pe_cnt-DB_MAX_PE_CNT)
            if par_task_cnt < 0:
                return False

        return True

    # can't have more memory hanging from bus than it's pe's
    def pe_mem_check(self):
        for idx in range(0, len(self.get_pe_set())):
            if self.get_mem_set()[idx] > self.get_pe_set()[idx]:
                return False

        return True

    # can't have more pe's than the number of tasks
    def pe_cnt_check(self, gen_config):
        MAX_TASK_CNT = gen_config["DB_MAX_TASK_CNT"]
        if self.get_pe_cnt() > MAX_TASK_CNT:
            return False
        return True

    def set_pe_task_set(self, pe_task_set):
        self.pe_task_set = pe_task_set

    def set_mem_task_set(self, mem_task_set):
        self.mem_task_set = mem_task_set

    def get_mem_task_set(self):
        return self.mem_task_set

    # get tasks of a bus
    def get_task_per_bus(self):
        bus_tasks = []
        flattened_indecies = self.flatten_indecies(self.pe_set)
        for bus_idx in range(0, len(self.get_pe_set())):
            list_unflattened = self.get_pe_task_set()[flattened_indecies[bus_idx]:flattened_indecies[bus_idx+1]]
            list_flattened = list(itertools.chain(*list_unflattened))
            bus_tasks.append(list_flattened)

        return bus_tasks

    def get_pe_task_set(self):
        return self.pe_task_set

    def get_task_s_pe(self, task_name):
        for idx, el in enumerate(self.get_pe_task_set()):
            if task_name in el:
                return idx

        print("task not found")
        return -1

    def get_task_s_mem(self, task_name):
        for idx, el in enumerate(self.get_mem_task_set()):
            if task_name in el:
                return idx

        print("task not found")
        return -1

    def set_PE_list(self, PE_list):
        self.PE_list = PE_list

    def flatten_indecies(self, list):
        result = [0]
        for el in list:
            result.append(result[-1]+el)
        return result

    def set_BUS_PE_list(self, PE_list):
        flattened_indecies = self.flatten_indecies(self.pe_set)
        for idx, BUS in enumerate(self.BUS_list):
            self.BUS_PE_list[idx] = PE_list[flattened_indecies[idx]: flattened_indecies[idx + 1]]

    def set_BUS_MEM_list(self, MEM_list):
        flattened_indecies = self.flatten_indecies(self.mem_set)
        for idx, BUS in enumerate(self.BUS_list):
            self.BUS_MEM_list[idx] = MEM_list[flattened_indecies[idx]: flattened_indecies[idx + 1]]

    def get_BUS_list(self):
        return self.BUS_list

    def get_BUS_PE_list(self):
        return self.BUS_PE_list

    def get_BUS_MEM_list(self):
        return self.BUS_MEM_list

    # idx index of the bus we need to know the neighbors for
    def get_bus_s_pe_neighbours(self, idx):
        return self.BUS_PE_list[idx]

    def get_bus_s_mem_neighbours(self, idx):
        return self.BUS_MEM_list[idx]

    def set_MEM_list(self, MEM_list):
        self.MEM_list = MEM_list

    def set_BUS_list(self, PE_list):
        self.BUS_list = PE_list

    # -----------------
    # system counters
    # -----------------
    # at the moment only considering bus and PE (assuming that mem would scale with bus. not the the size but bandwidth)
    def simple_customization_variation_cnt(self, gen_config):
        DB_MAX_PE_CNT = gen_config["DB_MAX_PE_CNT"]
        DB_MAX_MEM_CNT = gen_config["DB_MAX_MEM_CNT"]
        DB_MAX_BUS_CNT = gen_config["DB_MAX_BUS_CNT"]

        return pow(DB_MAX_PE_CNT, self.get_pe_cnt())*pow(DB_MAX_BUS_CNT, self.get_bus_cnt())*pow(DB_MAX_MEM_CNT, self.get_mem_cnt())

    def calc_customization_variation_cnt(self, gen_config):
        return self.simple_customization_variation_cnt(gen_config)

    # at the moment, we are considering the upper bounds for the cuts,
    # so in reality, there is a gonna be smaller to cuts.
    def design_space_reduction(self, gen_config, task_cnt=1):
        DB_MAX_PE_CNT = gen_config["DB_MAX_PE_CNT"]
        DB_MAX_MEM_CNT = gen_config["DB_MAX_MEM_CNT"]
        DB_MAX_BUS_CNT = gen_config["DB_MAX_BUS_CNT"]
        MAX_PAR_TASK_CNT = gen_config["MAX_PAR_TASK_CNT"]
        MAX_TASK_CNT = gen_config["MAX_TASK_CNT"]

        if DB_MAX_BUS_CNT - task_cnt <= 0:
            return pow(DB_MAX_BUS_CNT - 1, task_cnt)/pow(DB_MAX_BUS_CNT, task_cnt)
        else:
            return (DB_MAX_BUS_CNT-task_cnt)/DB_MAX_BUS_CNT  # the task resides in one of the PEs,
                                                   # for that one decrement. You need to look at it
                                                   # per mapping scenarios.
                                                   # note that if task_cnt > 1,
                                                   # calculation is a bit harder as it depends on whether
                                                   # tasks are binded to the same PE or not. We
                                                   # make this assumption since this will results in more conservative (less)
                                                   # DS reduction


    # map the tasks to the pes
    # each pe needs to have at least one task
    # we get rid of per bus scheduling combinations since PEs
    # are not distinguished yet (note that we can't do this across buses
    # as the topology (placement) already makes PE's within each bus
    # different relative to others
    def simple_mapping_variation_cnt(self, gen_config):
        MAX_TASK_CNT = gen_config["DB_MAX_TASK_CNT"]

        bins = self.get_pe_cnt()
        balls = MAX_TASK_CNT
        comb_each_pe_at_least_one_task = calc_stirling(balls, bins)*numbers.factorial(bins)
        for pe_cnt in self.get_pe_set():   # since PEs are still the same, get rid of the combinations per bus
            comb_each_pe_at_least_one_task /= numbers.factorial(pe_cnt)

        return comb_each_pe_at_least_one_task

    def calc_mapping_variation_cnt(self, gen_config):
        return self.simple_mapping_variation_cnt(gen_config)

    def system_get_mapping_variation_cnt(self):
        return self.mapping_variation_cnt

#-------------------------
# system counters
#-------------------------
def system_variation_count_2(gen_config):
    full_potential_tasks_list = gen_config["full_potential_tasks_list"]
    DB_MAX_PE_CNT = gen_config["DB_MAX_PE_CNT"]
    DB_PE_list = gen_config["DB_PE_list"]
    DB_MEM_list = gen_config["DB_MEM_list"]
    DB_BUS_list = gen_config["DB_BUS_list"]
    DB_MAX_MEM_CNT = gen_config["DB_MAX_MEM_CNT"]
    DB_MAX_BUS_CNT = gen_config["DB_MAX_BUS_CNT"]
    MAX_PAR_TASK_CNT = gen_config["MAX_PAR_TASK_CNT"]
    MAX_TASK_CNT = gen_config["MAX_TASK_CNT"]

    # mapping assisted topology
    task_cnt = MAX_TASK_CNT
    task_list = full_potential_tasks_list[0:MAX_PAR_TASK_CNT-1]

    system_variations = 0
    system_list = []

    #------------------
    # spawn different system topologies
    #------------------
    all_lists = []
    for bus_cnt in range(1, max((DB_MAX_PE_CNT*DB_MAX_BUS_CNT), MAX_PAR_TASK_CNT) + 1):
        # generate all the permutations of the pe values
        all_lists = [list(range(1, max(DB_MAX_PE_CNT, MAX_PAR_TASK_CNT) + 1))] * bus_cnt
        all_lists_permuted = list(itertools.product(*all_lists))

        # now generated a system with each permutation
        for pe_set in all_lists_permuted:
            system_ = SYSTEM_()
            system_.set_bus_cnt(bus_cnt)
            system_.set_pe_set(list(pe_set))
            system_list.append(system_)

    # generate software
    for system in system_list:
        system.set_task_cnt(MAX_TASK_CNT)
        system.set_par_task_cnt(MAX_PAR_TASK_CNT)


    #------------------
    # filter out infeasible/unreasonable systems
    #------------------
    filtered_system_list = []
    for system in system_list:
        if not system.pe_cnt_check():
            continue
        if not system.parallelism_check():
            continue

        filtered_system_list.append(system)


    #------------------
    # generate different mapping associated with the system topologies
    #------------------
    total_system_variation = 0
    for system in filtered_system_list:
        mapping_variations = system.calc_mapping_variation_cnt(gen_config)

    #total_systems = sum([system_.system_get_mapping_variation_cnt() for system_ in system_list])
    #print("number of system variations :" + str(total_system_variation))
    return total_system_variation
    #print("exhaustive simulation time (hours):" + str(float(total_system_variation)/(20*3600)))


def get_DS_cnt(DS_mode, gen_config, output_mode):
    if DS_mode == "exhaustive_naive":
        DS_size = system_variation_count_1(gen_config)
    elif DS_mode == "exhaustive_reduction_DB_semantics":
        DS_size = system_variation_count_2()

    if DS_output_mode == "DS_time":
        return DS_size*sim_time_per_design
    else:
        return DS_size


#-------------------------
# system generators
#--------------------------
# binning balls in to bins where there is
# at least one ball in every bin
def binning(bin_cnt, balls):
    # use product
    for indices in itertools.product(range(0, bin_cnt), repeat=len(balls)):
        result = [[]for _ in range (0, bin_cnt)]
        for ball_index, bin_index in enumerate(indices):
            result[bin_index].append(balls[ball_index])

        # discard if any of the bins are empty
        res_valid = True
        for el in result:
            if len(el) == 0:
               res_valid = False
        if res_valid:
            yield result
            # yield indices
        else:
            continue

# this is for parrallelizaton. it attempts to spreads the workload across the processes equally.
# this can not be perfectly equal as we parallelize based on mapped tasks (and not customized tasks). So,
# there is gonna be imbalances, but this is still better than no parallelization
def shard_work_equally(mapping_customization_variation_cnt_list, process_cnt):
    process_id_work_bound_dict = {}
    total_mapping = sum([el[0] for el in mapping_customization_variation_cnt_list])
    ideal_work_per_process = sum([el[0]*el[1] for el in mapping_customization_variation_cnt_list])/process_cnt
    map_idx_list = [0]
    total_system_accumulated = 0
    mapped_system_accumulated = 0
    for map, cust in mapping_customization_variation_cnt_list:
        for i in range(1, map+1):
            if total_system_accumulated + cust > ideal_work_per_process:
                map_idx_list.append(mapped_system_accumulated + max((i-1), 0))
                total_system_accumulated = 0
            else:
                total_system_accumulated += cust

        mapped_system_accumulated +=map

    if len(map_idx_list) < process_cnt + 1:
        map_idx_list.append(total_mapping)

    for process_id in range(0, len(map_idx_list)-1):
        process_id_work_bound_dict[process_id] = (map_idx_list[process_id], map_idx_list[process_id +1])

    return process_id_work_bound_dict


# second map tasks to the mems
def generate_mem_mapping(mapped_systems, process_id_work_dictionary, process_id):
    mapped_systems_completed = []
    if not (process_id in process_id_work_dictionary.keys()): # if we couldn't shard properly
        exit(0)
    system_lower_bound = process_id_work_dictionary[process_id][0]
    system_upper_bound = process_id_work_dictionary[process_id][1]
    for system in mapped_systems[system_lower_bound: system_upper_bound]:
        exessive_memory = False  # flaggign scenarios where not enough tasks to mapp to the memoroies
        isolated_siink = False
        all_task_mappings = []  # all the mappings to the memories
        for tasks, mem_cnt in zip(system.get_task_per_bus(), system.get_mem_set()):

            mappings = list(binning(mem_cnt, tasks)) # generating half designs

            # this covers scenarios where there are too many memories, so
            # we can distribute tasks to them
            if len(mappings) == 0:
                exessive_memory = True
                break
            else:
                task_names_ = mappings[0][0]
                if any([True for name in task_names_ if "siink" in name]) and len(mappings[0][0]) == 1:  # siink can't be occuipying any memory in isolation as it doesn't use any memory
                    isolated_siink = True
                    break
                else:
                    all_task_mappings.append(mappings)

            #task_mapping_filtered.append(remove_permutation_per_bus_2(all_task_mappings, system.get_mem_set()))

        if exessive_memory or isolated_siink: # too many memories. One memory would be task less
            continue

        all_permutations_tuples = list(itertools.product(*all_task_mappings))
        all_permutations_listified = []#all_permutations_tuples


        for design in all_permutations_tuples:
            new_design = list(design)
            #if len(design) == 1: # tuple of size 1:
            #    new_design.append([[]])
            all_permutations_listified.append(new_design)

        all_permutations_listified_flattened = []
        for design in all_permutations_listified:
            #all_tasks = [] # for debugging
            design_fixed = []
            for bus_s_mems in design:
                for mem in bus_s_mems:
                    design_fixed.append(mem)
                    #all_tasks.extend(mem)
            all_permutations_listified_flattened.append(design_fixed)

            """
            for task in full_potential_tasks_list:
                if not task in all_tasks:
                    print("what")
            """

        task_mapping_filtered = remove_permutation_per_bus_2(all_permutations_listified_flattened, system.get_mem_set())
        for task_mapping in task_mapping_filtered:
            system_ = deepcopy(system)
            system_.set_mem_task_set(task_mapping)
            mapped_systems_completed.append(system_)

    mapped_systems_completed_filtered = []  # filter scenarios that siink is by itself on memory
    for system in mapped_systems_completed:
        isolated_siink = False
        for tasks_on_mem in system.get_mem_task_set():
            if any([True for task in tasks_on_mem if ("siink" in task and len(tasks_on_mem) == 1)]):
                isolated_siink = True
                break
        if isolated_siink:
            continue
        mapped_systems_completed_filtered.append(system)
    return mapped_systems_completed_filtered
    # """    # comment in this if you don't care about task to memory mapping


# generate all customization scenarios
def generate_customization(mapped_systems, gen_config):
    full_potential_tasks_list = gen_config["full_potential_tasks_list"]
    DB_MAX_PE_CNT = gen_config["DB_MAX_PE_CNT"]
    DB_PE_list = gen_config["DB_PE_list"]
    DB_MEM_list = gen_config["DB_MEM_list"]
    DB_BUS_list = gen_config["DB_BUS_list"]
    DB_MAX_MEM_CNT = gen_config["DB_MAX_MEM_CNT"]
    DB_MAX_BUS_CNT = gen_config["DB_MAX_BUS_CNT"]

    customized_systems = []

    for idx, system in enumerate(mapped_systems):
        pe_cnt = system.get_pe_cnt()
        mem_cnt = system.get_mem_cnt()
        bus_cnt = system.get_bus_cnt()
        PE_scenarios = list(itertools.product(DB_PE_list, repeat=pe_cnt))
        MEM_scenarios = list(itertools.product(DB_MEM_list, repeat=mem_cnt))
        BUS_scenarios = list(itertools.product(DB_BUS_list, repeat=bus_cnt))
        customization_scenarios = list(itertools.product(PE_scenarios, MEM_scenarios, BUS_scenarios))
        for customized_scn in customization_scenarios:
            system_ = deepcopy(system)

            system_.set_BUS_list(list(customized_scn[2]))
            system_.set_BUS_PE_list(list(customized_scn[0]))
            system_.set_BUS_MEM_list(list(customized_scn[1]))

            #system_.set_PE_list(list(customized_scn[0]))
            #system_.set_MEM_list(list(customized_scn[1]))
            #system_.set_BUS_list(list(customized_scn[2]))

            # quick sanity check
            for task_name in full_potential_tasks_list:
                if system.get_task_s_mem(task_name) == -1:
                    print("something went wrong")
                    exit(0)
            customized_systems.append(system_)
    return customized_systems


def generate_topologies(gen_config):
    DB_MAX_PE_CNT = gen_config["DB_MAX_PE_CNT"]
    DB_MAX_MEM_CNT = gen_config["DB_MAX_MEM_CNT"]
    DB_MAX_BUS_CNT = gen_config["DB_MAX_BUS_CNT"]
    MAX_PAR_TASK_CNT = gen_config["DB_MAX_PAR_TASK_CNT"]
    MAX_TASK_CNT = gen_config["DB_MAX_TASK_CNT"]

    DB_MIN_PE_CNT = gen_config["DB_MIN_PE_CNT"]
    DB_MIN_MEM_CNT = gen_config["DB_MIN_MEM_CNT"]
    DB_MIN_BUS_CNT = gen_config["DB_MIN_BUS_CNT"]
    #DB_MAX_SYSTEM_to_investigate = gen_config["DB_MAX_SYSTEM_to_investigate"]

    #DB_MIN_PE_CNT = 1
    #DB_MIN_MEM_CNT = 1
    #DB_MIN_BUS_CNT = 1



    system_list = []
    all_lists = []
    for bus_cnt in range(DB_MIN_BUS_CNT, min(DB_MAX_BUS_CNT, MAX_PAR_TASK_CNT) + 1):
        # generate all the permutations of the pe values
        all_lists = [list(range(1, min(DB_MAX_PE_CNT, MAX_PAR_TASK_CNT) + 1))] * bus_cnt
        pe_dist_perm = list(itertools.product(*all_lists))

        # some filtering, otherwise never finish
        pe_dist_perm_filtered = []
        for pe_dist in pe_dist_perm:
            if sum(pe_dist) <= MAX_TASK_CNT and sum(pe_dist) >= DB_MIN_PE_CNT and sum(pe_dist)<= DB_MAX_PE_CNT:
                pe_dist_perm_filtered.append(pe_dist)

        pe_mem_dist_perm = list(itertools.product(pe_dist_perm_filtered, repeat=2))

        # now generated a system with each permutation
        for pe_set, mem_set in pe_mem_dist_perm:
            system_ = SYSTEM_()
            system_.set_bus_cnt(bus_cnt)
            system_.set_pe_set(list(pe_set))
            system_.set_mem_set(list(mem_set))
            system_list.append(system_)

    # add some sw information
    for system in system_list[:len(system_list)]:
        system.set_task_cnt(MAX_TASK_CNT)
        system.set_par_task_cnt(MAX_PAR_TASK_CNT)

    #------------------
    # filter out infeasible/unreasonable topologies: using parallelism/customization at the moment
    #------------------
    filtered_system_list = []
    for system in system_list:
        if not system.pe_cnt_check(gen_config):
            continue
        if not system.parallelism_check(gen_config):
            continue
        #if not system.pe_mem_check():
        #    continue
        filtered_system_list.append(system)

    return filtered_system_list[:min(len(filtered_system_list), gen_config["DB_MAX_SYSTEM_to_investigate"])]

# count all the mapping and customizations
def count_mapping_customization(system_topologies, gen_config):
    mapping_customization_variation_cnt_list = []  # keep track of how many mapping variations per topology
    total_system_variation_cnt = 0
    for system in system_topologies:
        # get system counts per topology (for sanity checks)
        mapping_variation_cnt = system.calc_mapping_variation_cnt(gen_config)
        customization_variation_cnt = system.calc_customization_variation_cnt(gen_config)
        # customization_variation_cnt = 1
        total_system_variation_cnt += mapping_variation_cnt*customization_variation_cnt
        mapping_customization_variation_cnt_list.append((mapping_variation_cnt, customization_variation_cnt))
        #print(sum([el[0] for el in mapping_variation_cnt_list]))
    return mapping_customization_variation_cnt_list, total_system_variation_cnt


# generate all the mappings for the topologies specified in filtered_system_list
def generate_pe_mapping(filtered_system_list, gen_config):
    full_potential_tasks_list = gen_config["full_potential_tasks_list"]
    DB_MAX_PE_CNT = gen_config["DB_MAX_PE_CNT"]
    DB_MAX_MEM_CNT = gen_config["DB_MAX_MEM_CNT"]
    DB_MAX_BUS_CNT = gen_config["DB_MAX_BUS_CNT"]
    MAX_PAR_TASK_CNT = gen_config["DB_MAX_PAR_TASK_CNT"]
    MAX_TASK_CNT = gen_config["DB_MAX_TASK_CNT"]
    DB_MAX_PE_CNT_range = gen_config["DB_MAX_PE_CNT_range"]
    MAX_TASK_range = gen_config["DB_MAX_TASK_CNT_range"]
    MAX_TASK_CNT_range = gen_config["DB_MAX_TASK_CNT_range"]
    MAX_PAR_TASK_CNT_range = gen_config["DB_MAX_PAR_TASK_CNT_range"]
    DS_output_mode = gen_config["DS_output_mode"]
    DB_MAX_SYSTEM_to_investigate = gen_config["DB_MAX_SYSTEM_to_investigate"]


    mapping_customization_variation_cnt_list = []

    # ------------------
    # helpers
    # ------------------
    # find a the pe idx for a task
    def get_tasks_pe_helper(mapping, task):
        for idx, tasks in enumerate(mapping):
            if task in tasks:
                return idx
        print("task is not found")
        return -1

    # filter mappings scenarios where source/sink don't map to the same pe as the second/second to last tasks respectively
    # Since source and sink are dummies, it doesn't make a differrence if they are mapped  to other pes
    def filter_sink_source(all_taks_mappings, gen_config):
        full_potential_tasks_list = gen_config["full_potential_tasks_list"]
        MAX_TASK_CNT = gen_config["DB_MAX_TASK_CNT"]

        result = []
        for mapping in all_task_mappings:
            source_idx = get_tasks_pe_helper(mapping, full_potential_tasks_list[0])
            task_1_idx = get_tasks_pe_helper(mapping, full_potential_tasks_list[1])
            if source_idx == -1 or task_1_idx == -1:
                print("something went wrong")
                exit(0)
            if not (source_idx == task_1_idx):
                continue

            sink_idx = get_tasks_pe_helper(mapping, full_potential_tasks_list[-1])
            last_task_idx = get_tasks_pe_helper(mapping, full_potential_tasks_list[-2])
            if sink_idx == -1 or last_task_idx == -1:
                print("something went wrong")
                exit(0)
            if not (sink_idx == last_task_idx):
                continue

            result.append(mapping)

        if len(result) == 0:  # there are cases that because of the set up, the criteria can not be met
            return all_taks_mappings
        return result


    mapped_systems = []
    start = time.time()
    for idx, system in enumerate(filtered_system_list):
        if len(mapped_systems) > DB_MAX_SYSTEM_to_investigate:
            break

        all_task_mappings = list(binning(system.get_pe_cnt(), full_potential_tasks_list[0:MAX_TASK_CNT]))

        # Filtering
        # flter 1: filter mappings scenarios where source/sink don't map to the same pe as the second/second to last tasks respectively
        # Since source and sink are dummies, it doesn't make a differrence if they are mapped  to other pes
        #task_mapping_filtered_1 = all_task_mappings  # uncomment if you don't care about source/sink
        task_mapping_filtered_1 = filter_sink_source(all_task_mappings, gen_config)
        # filter_2:  filter the scenarios where tasks are mapped similarly to the same bus
        #task_mapping_filtered_2 = remove_permutation_per_bus_2(task_mapping_filtered_1, system.get_pe_set())
        task_mapping_filtered_2 = task_mapping_filtered_1

        use_balanced_mapping = True
        # just to filter out certain scenarios for debugging. not necessary
        if (use_balanced_mapping):
            task_mapping_filtered_3 = []
            # find the most balanced design
            task_mapping_mapping_std = {}
            for idx, task_mapping in enumerate(task_mapping_filtered_2):
                task_mapping_mapping_std[idx] = np.std([len(el) for el in task_mapping])
            sorted_x = collections.OrderedDict(sorted(task_mapping_mapping_std.items(), key=operator.itemgetter(1)))

            for idx in range(0, min(1, len(task_mapping_filtered_2))):
                blah = task_mapping_filtered_2[list(sorted_x.keys())[idx]]
                task_mapping_filtered_3.append(blah)
            task_mapping_filtered_2 = task_mapping_filtered_3

        #  populate
        for task_mapping in task_mapping_filtered_2:
            system_ = deepcopy(system)
            system_.set_pe_task_set(task_mapping)
            mapped_systems.append(system_)
            mapping_customization_variation_cnt_list.append((1,
                                                             pow(system_.get_pe_cnt(), DB_MAX_PE_CNT)*
                                                             pow(system_.get_mem_cnt(), DB_MAX_MEM_CNT)*
                                                             pow(system_.get_bus_cnt(), DB_MAX_BUS_CNT)))

    return mapped_systems,mapping_customization_variation_cnt_list


def exhaustive_system_generation(system_workers, gen_config):
    mapping_process_cnt = system_workers[0]
    mapping_process_id = system_workers[1]
    FARSI_gen_process_id = system_workers[1]
    #customization_process_cnt = system_workers[2]
    #customization_process_id = system_workers[3]

    # mapping assisted topology
    #task_cnt = MAX_TASK_CNT
    system_variations = 0

    # generate systems with various topologies
    system_topologies = generate_topologies(gen_config)

    # count all the valid mappings and customization (together)
    mapping_customization_variation_cnt_list, total_system_variation_cnt = count_mapping_customization(system_topologies, gen_config)

    # generate different mapping
    start = time.time()
    mapped_pe_systems, mapping_customization_variation_cnt_list = generate_pe_mapping(system_topologies, gen_config)  # map tasks to PEs
    print("pe mapping time" + str(time.time() - start))
    end = time.time()
    #mapped_systems_completed = mapped_pe_systems

    # parallelize
    process_id_work_dictionary = shard_work_equally(mapping_customization_variation_cnt_list, mapping_process_cnt)

    # map tasks to MEMs. comment out if you don't care about this
    start = time.time()
    mapped_systems_completed = generate_mem_mapping(mapped_pe_systems, process_id_work_dictionary, mapping_process_id)
    print("mem mapping time" + str(time.time() - start))

    # generate different customizations
    start = time.time()
    customized_systems = generate_customization(mapped_systems_completed, gen_config)
    print("customization time" + str(time.time() - start))

    print("total systems to explore for process id" + str(mapping_process_id) + "_" + str(FARSI_gen_process_id)+ "_" + str(len(customized_systems)))
    return customized_systems[: min(len(customized_systems), gen_config["DB_MAX_SYSTEM_to_investigate"])]


#for DS_type in ["naive", "DB_semantics"]:
#---------------------------
# sweepers
#---------------------------
def sweep_DS_info(gen_config):
    DB_MAX_PE_CNT = gen_config["DB_MAX_PE_CNT"]
    DB_MAX_MEM_CNT = gen_config["DB_MAX_MEM_CNT"]
    DB_MAX_BUS_CNT = gen_config["DB_MAX_BUS_CNT"]
    MAX_PAR_TASK_CNT = gen_config["MAX_PAR_TASK_CNT"]
    MAX_TASK_CNT = gen_config["MAX_TASK_CNT"]
    DB_MAX_PE_CNT_range = gen_config["DB_MAX_PE_CNT_range"]
    MAX_TASK_range = gen_config["MAX_TASK_CNT_range"]
    MAX_TASK_CNT_range = gen_config["MAX_TASK_CNT_range"]
    MAX_PAR_TASK_CNT_range = gen_config["MAX_PAR_TASK_CNT_range"]
    DS_output_mode= gen_config["DS_output_mode"]

    DS_type = gen_config["DS_type"]
    print("DS_type:" + DS_type)
    for DB_MAX_PE_CNT in DB_MAX_PE_CNT_range:

        # printing stuff
        print("-----------------------------------------")
        print("PB DB count:" + str(DB_MAX_PE_CNT))
        print("-----------------------------------------")
        print(" ,", end=" ")
        for MAX_PAR_TASK_CNT in MAX_PAR_TASK_CNT_range:
            print(str(MAX_PAR_TASK_CNT) +",", end =" ")
        print("\n")

        DB_MAX_BUS_CNT = DB_MAX_MEM_CNT = DB_MAX_PE_CNT
        for MAX_TASK_CNT in MAX_TASK_CNT_range:
            print(str(MAX_TASK_CNT) +",", end=" ")
            for MAX_PAR_TASK_CNT in MAX_PAR_TASK_CNT_range:
                print(str(get_DS_cnt(DS_type, gen_config, DS_output_mode)) +",", end =" ")
            print("\n")

def lists_of_lists_equal(lol1, lol2):
    if not (len(lol1) == len(lol2)):
        return False
    for lol1_el in lol1:
        if not(lol1_el in lol2):
            return False
    return True

def listify_tuples(list_):
    result = []
    for el in list_:
        if isinstance(el, tuple):
            result.extend(list(el))
        else:
            result.extend(el)
    return result

# mapping equal if the tasks under the PEs mapped to the same bus
# are equal
def mapping_equal(system_1, system_2, IP_set_per_bus):
    if not len(system_1) == len(system_2):
        return False

    IP_set_per_bus_acc = [0]
    for idx, IP_set_per_bus_el in enumerate(IP_set_per_bus):
        IP_set_per_bus_acc.append(IP_set_per_bus_acc[idx] + IP_set_per_bus_el)

    for idx in range(0, len(IP_set_per_bus_acc) - 1):
            idx_low = IP_set_per_bus_acc[idx]
            idx_up = IP_set_per_bus_acc[idx + 1]
            #system_1_listified = listify_tuples(system_1)
            #system_2_listified = listify_tuples(system_2)
            #if not lists_of_lists_equal(system_1_listified[idx_low:idx_up], system_2_listified[idx_low:idx_up]):
            if not lists_of_lists_equal(system_1[idx_low:idx_up], system_2[idx_low:idx_up]):
                return False

    return True

# since permutations of the blocks per buses do not generate new toopologies (since we haven't
# assigned a IP to them), we need to remove them
def remove_permutation_per_bus_2(PE_with_task_mapping_list, IP_set_per_bus):

    duplicated_systems_idx = []
    # iterate through and check the equality of each design
    for idx_x in range(0, len(PE_with_task_mapping_list)):
        if idx_x in PE_with_task_mapping_list:
            continue
        for idx_y in range(idx_x+1, len(PE_with_task_mapping_list)):
            if mapping_equal(PE_with_task_mapping_list[idx_x], PE_with_task_mapping_list[idx_y], IP_set_per_bus):
                duplicated_systems_idx.append(idx_y)

    non_duplicates = [PE_with_task_mapping_list[idx] for idx in range(0, len(PE_with_task_mapping_list))
                                      if not(idx in duplicated_systems_idx)]


    return non_duplicates

# ------------------
# some unit test. keep for unit testing
# ------------------
"""
system_1 = [[1,2,3], [4],[5]]
system_2 = [[1,3,2], [4]]
print(mapping_equal(system_1, system_2, [2,1]) == False)

system_1 = [[1,2,3], [4,5], [5]]
system_2 = [[4, 5], [1,2,3], [5]]
print(mapping_equal(system_1, system_2, [2,1]) == True)

system_1 = [[1,2,3], [4,5], [5], [6]]
system_2 = [[4, 5], [1,2,3], [6], [5]]
print(mapping_equal(system_1, system_2, [2,2]) == True)

system_1 = [[1,2,3], [4,5], [5], [6]]
system_2 = [[4, 5], [1,2,3], [6], [5]]
print(mapping_equal(system_1, system_2, [3,1]) == False)


system_1 = [[1,2,3], [4,5], [5], [6]]
system_2 = [[1, 2,3], [5], [6], [4,5]]
print(mapping_equal(system_1, system_2, [1,3]) == True)


system_1 = [[1,2,3], [4,5], [5], [6]]
system_2 = [[2,3], [5], [6], [4,5]]
print(mapping_equal(system_1, system_2, [1,3]) == False)


system_1 = [[1,2,3], [4,5], [5], [6]]
system_2 = [[2,3, 1], [5], [6], [4,5]]
print(mapping_equal(system_1, system_2, [1,3]) == False)

system_1 = [[1,2,3], [4,5]]
system_2 = [[4,5], [1,2,3]]
print(mapping_equal(system_1, system_2, [1,1]) == False)

system_1 = [[1,2,3], [4,5]]
system_2 = [[4,5], [1,2,3]]
print(mapping_equal(system_1, system_2, [2,0]) == True)
print("ok")
"""
"""
# binning tasks to PEs
total_PEs = 3
PE_with_task_mapping = list(binning(total_PEs, full_potential_tasks_list[:4]))
IP_set_per_bus = [1,1,1]
remove_permutation_per_bus_2(PE_with_task_mapping, IP_set_per_bus)
#a = bin.combinations()
print(results)
"""

#exhaustive_system_generation()
"""
def gen_test():
    for i in range(0,10):
        yield i

    return None

gen = gen_test()
for _ in gen:
    print(_)
"""
#for MAX_TASK_CNT in range(5, 8):
#    for MAX_PAR_TASK_CNT in range(1,3):
#        system_variation_count_2()

# all the combinations of balls and bins
# # https://www.careerbless.com/aptitude/qa/permutations_combinations_imp7.php

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np


def plot_design_space_size():
    pe_range = [10, 12, 14, 16, 18, 20]
    pe_count_design_space_size = {}
    knob_count = 4
    for pe_cnt in pe_range:
        pe_count_design_space_size[pe_cnt] = count_system_variation(pe_cnt, knob_count, "customization")


    #colors = {'Sim Time: Sub Sec':'lime', 'Sim Time: Sec':'darkgreen', 'Sim Time: Minute':'darkgreen', 'Sim Time: Hour':'olivedrab', 'Sim Time: Day':'darkgreen'}
    colors = {'Sim Time: Sub Sec':'lime', 'Sim Time: Sec':'darkgreen', 'Sim Time: Minute':'darkgreen', 'Sim Time: Hour':'white', 'Sim Time: Day':'white'}
    simulation_time = {"Sim Time: Sub Sec": .1*1/60, "Sim Time: Sec": 1/60, "Sim Time: Minute":2, 'Sim Time: hour': 60, 'Sim Time: Day': 60*24}
    #simulation_time = {'mili-sec':.001*1/60, 'hour': 60, 'day': 60*24}
    selected_simulation_time = {'Sim Time: Hour': 60, 'Sim Time: Day': 60*24}


    # budget

    font_size =25

    # cardinality
    fig, ax1 = plt.subplots()
    #ax1 = ax3.twinx()
    ax1.set_xlabel('Number of Processing Elements', fontsize=font_size)
    ax1.set_ylabel('Design Space Size', fontsize =font_size)
    ax1.plot(list(pe_count_design_space_size.keys()), list(pe_count_design_space_size.values()), label="Cardinality", color='red')
    #plt.legend() #loc="upper left")


    # exploration time
    ax2 = ax1.twinx()
    cntr = 0
    for k ,v in selected_simulation_time.items():
        ax2.plot(list(pe_count_design_space_size.keys()),
                 [(el * v)/ (365 * 24 * 60) for el in list(pe_count_design_space_size.values())], label=k, color=colors[k], linestyle=":")
        cntr +=1


    """
    k = 'Sim Time: Sub Sec'
    v = simulation_time[k]
    percentage = .0001
    ax2.plot(list(pe_count_design_space_size.keys()),
             [(el * v*percentage) / (30 * 24 * 60) for el in list(pe_count_design_space_size.values())], label=k + "+ selected points", color="gold",
             linestyle=":")
    """

    #ax2.plot(list(pe_count_design_space_size.keys()),
    #         [(el * v) / (30 * 24 * 60) for el in list(pe_count_design_space_size.values())], label=k + " + selected points", color="yellow", linestyle=":")
    ax2.set_ylabel('Exploration Time (Year)', fontsize=font_size)

    ax3 = ax1.twinx()
    #ax3.hlines(y=100, color='blue', linestyle='-', xmin=min(pe_range), xmax=20, label="Exploration Time Budget")
    ax3.hlines(y=100, color='white', linestyle='-', xmin=min(pe_range), xmax=20, label="Exploration Time Budget")

    #ax1.hlines(y=.00000005, color='r', linestyle='-', xmin=8, xmax=20)



    # ticks and such
    ax1.tick_params(axis='y', labelsize=font_size)
    ax1.tick_params(axis='x', labelsize=font_size)

    #ax2.tick_params(axis='y', which="both", bottom=False, top=False, right=False, left=False, labelbottom=False)
    #ax3.tick_params(axis='y', which="both", bottom=False, top=False, right=False, left=False, labelbottom=False)
    #ax3.tick_params(None)
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')

    ax1.set_xticklabels([10, 12, 15, 17, 20])
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])

    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    ax3.xaxis.set_ticks_position('none')
    ax3.yaxis.set_ticks_position('none')

    ax1.set_ylim((1, ax1.get_ybound()[1]))
    ax2.set_ylim((1, ax1.get_ybound()[1]))
    ax3.set_ylim((1, ax1.get_ybound()[1]))
    ax1.set_xlim((10, 20))
    ax2.set_xlim((10, 20))
    ax3.set_xlim((10, 20))





    # show
    #plt.legend() #loc="upper left")
    fig.tight_layout()
    plt.show()
    fig.savefig("exploration_time.png")

    print("ok")
# print("number of system variations: " + str(float(system_variations)))
# print("exhaustive simulation time (hours): " + str(float(system_variations)/(20*3600)))

def count_system_variation(task_cnt, knob_count, mode="all"):
    MAX_TASK_CNT = task_cnt

    # assuming that we have 5 different tasks and hence (can have up to 5 different blocks).
    # we'd like to know how many different migration/allocation combinations are out there.
    # Assumptions:
    #           PE's are identical.
    #           Buses are identical
    #           memory is ignored for now

    MAX_PE_CNT = MAX_TASK_CNT
    task_cnt = MAX_TASK_CNT
    system_variations = 0
    num_of_knobs = knob_count

    topology_dict = [1,
                     1,
                     4,
                     38,
                     728,
                     26704,
                     1866256,
                     251548592,
                     66296291072,
                     34496488594816,
                     35641657548953344,
                     73354596206766622208,
                     301272202649664088951808,
                     2471648811030443735290891264,
                     40527680937730480234609755344896,
                     1328578958335783201008338986845427712,
                     87089689052447182841791388989051400978432,
                     11416413520434522308788674285713247919244640256,
                     2992938411601818037370034280152893935458466172698624,
                     1569215570739406346256547210377768575765884983264804405248,
                     1645471602537064877722485517800176164374001516327306287561310208]

    for pe_cnt in range(1, MAX_PE_CNT):

        topology = topology_dict[pe_cnt-1]
        # then calculate mapping (migrate)
        mapping = calc_stirling(task_cnt, pe_cnt)*factorial(pe_cnt)
        # then calculate customization (swap)
        swap = math.pow(num_of_knobs, (pe_cnt))

        if mode == "all":
            system_variations += topology*mapping*swap
        if mode == "customization":
            system_variations += swap
        if mode == "mapping":
            system_variations += mapping
        if mode == "topology":
            system_variations += topology





    return system_variations
    #print("{:e}".format(system_variations))

pe_cnt = 20
knob_count = 4
ds_size = {}
ds_size_digits = {}
for design_stage in ["topology", "mapping", "customization", "all"]:
    ds_size[design_stage] = count_system_variation(pe_cnt, knob_count, design_stage)
    ds_size_digits[design_stage] = math.log10(count_system_variation(pe_cnt, knob_count, design_stage))



#plot_design_space_size()

