#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

# simulation mode
from collections import defaultdict
import home_settings
from specs.LW_cl import  *
import os
import settings.config_cacti as CC
termination_mode = "workload_completion"  # when to terminate the exploration
assert termination_mode in ["workload_completion", "time_budget_reached"], "termination_mode:" +\
                                                                           termination_mode + " not defined"
# json inputs
design_input_folder = "model_utils/specs"

simulation_method = "performance"   # whether to performance simulator or power simulator
#simulation_method = "power_knobs"

# --------------------
# DSE params
# --------------------
# algorithm
dse_type = "hill_climbing"   # [exhaustive, hill_climbing]  # type of the design space exploration
dse_collection_mode = "serial"  # [serial, parallel] # only relevant for exhaustive for now
                                         # python parallel: python spawns processes  (used when running within a server)
                                         # bash_parallel: you need to manually run the processes (used when you want to
                                         #                                                    run across multiple servers)

# exhaustive analysis output folder names
exhaustive_output_file_prefix = stage_1_input_exhaustive_data = 'exhaustive_for_pid'
stage_2_input_exhaustive_data = 'stage_2_for_pid'
stage_2_output_exhaustive_data = 'stage_2_consolidated'
exhaustive_target_data = "target_budget"
exhaustive_result_dir = "exhaustive_search"
FARSI_result_dir = "FARSI_search"
FARSI_outputfile_prefix_verbose = "FARSI_search"
FARSI_outputfile_prefix_minimal = "FARSI_search_minimal"
FARSI_outputfile_prefix_error_integration = "FARSI_error_integration_study"  # compile the data for error_integration vs exact IP library search
FARSI_cost_correlation_study_prefix = "FARSI_cost_correlation_study"
FARSI_input_error_output_cost_sensitivity_study_prefix = "FARSI_input_error_output_cost_sensitivity_study"
FARSI_input_error_input_cost_sensitivity_study_prefix = "FARSI_input_error_input_cost_sensitivity_study"
FARSI_input_error_output_PPA_sensitivity_study_prefix = "FARSI_input_error_output_PPA_sensitivity_study"
FARSI_simple_run_prefix = "FARSI_simple_run"
FARSI_simple_sim_run_study_prefix = "simple_sim_run"


parallel_processes_count = 1  # number of processes to spawn to search the design space. Only used for exhaustive search
process_id = 0  # used for parallel execution. Each process is assigned an id, through main

warning_mode = "always"
num_clusters = 2  # how many clusters to create everytime we split
TOTAL_RUN_THRESHOLD = 5000 # acceptable iterations count without improvement before termination
DES_STAG_THRESHOLD = 50  # acceptable iterations count without improvement before termination

neigh_gen_mode = "some" # neighbouring design points generation mode ("all" "random_one", ...)
num_neighs_to_try = 3 # how many neighs to try around a dp
neigh_sel_mode = "best"  # neighbouring design selection mode (best, sometimes, best ...)
dp_rank_obj = "latency"  # design point ranking object function(best, sometimes, best ...)

sel_next_dp = "all_metrics"  # how to select the next desigh, ["all_metrics", "one_"metric"]


# selection algorithm (picking the best neighbour)
neigh_sel_algorithm = "annealing"
SA_breadth = 1 # breath of the neighbour search
SA_depth = 15 # depth of the neighbour search
annealing_max_temp = 500
annealing_temp_dec = 50
annealing_dampening_coef = 10  # how much to dampen the metric that has  met the design objectives
                               # in annealing_energy calculation

metric_sel_dis_mode = "eliminate"  # {"eliminate", "dampen", "simple"} eliminate, eliminates the metric metric from
                                   # distance calculation and dampen , dampens it (using annealing_dampening_coef)

# scheduling policy
scheduling_policy = "FRFS"  # first read, first serve

#  migration policy
#migrant_clustering_policy = "tasks_dependency" # the policy that is used for clustering the migrants ["random", "tasks_dependency"]
migrant_clustering_policy = "selected_kernel" # the policy that is used for clustering the migrants ["random", "tasks_dependency"]
migration_policy = "random"  # the policy to use in picking the clustered tasks to move

# objectives
objectives = ["latency"]  # [power, area, latency] are the options
#objective_function_type = "pareto"  # ["pareto", "weighted"] if weighted, we need to provide the weights

sorting_SOC_metric = "power"
all_metrics = ["latency",  "power", "area", "energy", "cost"]    # all the metrics for the evaluation of SOC
budgetted_metrics = ["latency",  "power", "area"]
other_metrics = ["cost"]

budget_dict = {}
budget_dict["glass"] = {}
budget_dict["glass"]["power"] = .05
budget_dict["glass"]["area"] = .000005

home_dir = home_settings.home_dir
#home_dir = os.getcwd()+"/../../"

# ------------------------
# verify parameters
# ------------------------
verify_home_dir = home_dir

# ------------------------
# move parameters
# ------------------------
# negative means smaller the better. We only support minimization
# problems (i.e, the smaller the better)
metric_improvement_dir = {}
metric_improvement_dir["latency"] = -1  # direction of improvement is reduction, and hence -1
metric_improvement_dir["power"] = -1  # direction of improvement is reduction, and hence -1
metric_improvement_dir["energy"] = -1 # direction of improvement is reduction, and hence -1
metric_improvement_dir["area"] = -1  # direction of improvement is reduction, and hence -1
metric_improvement_dir["cost"] = -1 # direction of improvement is reduction, and hence -1
move_s_krnel_selection = ["bottleneck"]   # options are :bottleneck, improvement_ease


for metric in all_metrics:
    if (metric not in metric_improvement_dir.keys()) or\
            not(metric_improvement_dir[metric] == -1):  # is not a minimization problem
                print("---Error:can only support metrics that require minimization. You need to change the metric selection in"
                      "navigation heuristic if you want otherwise in ")
                exit(0)

heuristic_type = "FARSI" # {moos, FARSI, SA}
moos_greedy_mode = "phv"
MOOS_GREEDY_CTR_RUN = 10
DESIGN_COLLECTED_PER_GREEDY = 20

#objective_function = 0  #
#objective_budget = .000000001
metric_trans_dict = {"latency": ["split", "swap", "migrate", "split_swap"], "power": ["split", "swap", "migrate", "split_swap"],
                      "area": ["split", "swap", "migrate", "split_swap"]}

cleaning_threshold = 2000000000000000000000  # how often to activate cleaning
cleaning_consecutive_iterations = 1  # how many consecutive iterations to clean

move_metric_ranking_mode = "exact"  # exact, prob.  If exact, metrics are ranked (and hence selected) based on
                                    # their distance to the goal. If prob, we sample probabilistically based on the
                                    # distance

move_krnel_ranking_mode = "exact"  # exact, prob.  If exact, kernels are ranked (and hence selected) based on
                                    # their distance to the goal. If prob, we sample probabilistically based on the
                                    # distance

move_blck_ranking_mode = "exact"  # exact, prob.  If exact, blocks are ranked (and hence selected) based on
                                    # their distance to the goal. If prob, we sample probabilistically based on the
                                    # distance

max_krnel_stagnation_ctr = 2
fitted_budget_ctr_threshold = 1  # how many times fitting the budget before terminating

recently_cached_designs_queue_size = 20
max_recently_seen_design_ctr = 2
assert(recently_cached_designs_queue_size > max_recently_seen_design_ctr)
# --------------------
# DEBUGGING
# --------------------
NO_VIS = False # if set, no visualization is used. This speeds up everything
DEBUG_SANITY  = True # run sanity check on the design
DEBUG_FIX = False  # non randomize the flow (by not touching the seed)
VIS_GR_PER_GEN =  False and not NO_VIS # visualize the graph per design point generation
VIS_SIM_PER_GEN = False and not NO_VIS # if true, we visualize the simulation progression
VIS_GR_PER_ITR =  True and not NO_VIS # visualize the graph exploration per iteration
VIS_PROFILE = True and not NO_VIS # visualize the profiling data
VIS_FINAL_RES = False and not NO_VIS # see the final results
VIS_ALL = False  and not NO_VIS # visualize everything
REPORT = True  # report the stats (to the screen); draw plots.
DATA_DELIVEYRY = "absolute" #[obfuscate, absolute]"
DEBUG_MISC = False  # scenarios haven't covered above
pa_conversion_file = "pa_conversion.txt"
WARN = False
data_folder = "data"
PA_output_folder = data_folder+"/"+"PA_output"
sim_progress_folder = data_folder+"/"+"sim_progress"
RUN_VERIFICATION_PER_GEN = False # every new desi, generate the verification data
RUN_VERIFICATION_PER_NEW_CONFIG = False
RUN_VERIFICATION_PER_IMPROVMENT = False and not (RUN_VERIFICATION_PER_GEN or RUN_VERIFICATION_PER_NEW_CONFIG) # every improvement, generate verification
                                                                         # don't want to double generate, hence the second
                                                                         # predicate clause
RUN_VERIFICATION_AT_ALL = RUN_VERIFICATION_PER_IMPROVMENT or RUN_VERIFICATION_PER_NEW_CONFIG or RUN_VERIFICATION_PER_GEN

VIS_SIM_PROG = RUN_VERIFICATION_PER_GEN or RUN_VERIFICATION_PER_IMPROVMENT or RUN_VERIFICATION_PER_NEW_CONFIG  # visualize the simulation progression

verification_result_file = "verification_result_file.csv"
# MOVES

FARSI_memory_consumption = "high"  # [low, high] if low is selected, we deactivate certain knobs to avoid using memory excessively
FARSI_performance = "fast"  # ["slow", "fast"]  # if set to fast, we don't visualize as often and use certain fast versions
                            # of functions to accomplish the tasks
if FARSI_performance == "fast":
    vis_reg_ctr_threshold = 60
else:
    vis_reg_ctr_threshold = 1

DEBUG_MOVE =  True and not NO_VIS # if true, we print/collect relevant info about moves
regulate_move_tracking = (FARSI_memory_consumption == "low") # if true, we don't track and hence graph every move. This helps preventing memory pressure (and avoid getting killed by the OS)
#vis_move_trail_ctr_threshold = 20 # how often sample the moves (only applies if regulat_move_tracking enabled)

cache_seen_designs = False and not(FARSI_memory_consumption == "low") # if True, we cache the designs that we have seen. This way we wont simulate them unnecessarily.
                          # This should be set to false if memory is an issue

VIS_MOVE_TRAIL = DEBUG_MOVE and not NO_VIS and False
eval_mode ="statistical"  # not statistical evaluation ["singular, statistical]. Note that singular is deprecated now
statistical_reduction_mode = "avg"
hw_sampling = {"mode":"exact", "population_size":1, "reduction":"avg"}   # mode:["error_integration", "exact"]  # error integration means that our IP library has an error and needs to be taken into account
                                               # exact, means that (even if IP library has an error), treat the (most likely) value as accurate value

check_pointing_allowed = True
check_point_list = ["ex", "db", "counters"] #choose from ["sim","ex", "db", "counters"]


use_slack_management_estimation = False and not (RUN_VERIFICATION_PER_GEN or RUN_VERIFICATION_PER_IMPROVMENT or RUN_VERIFICATION_PER_NEW_CONFIG)# if run verification, we can apply slack, otherwise we get the wrong numbers
jitter_population_size= 1  # not statistical evaluation
if hw_sampling["mode"] == "exact":
    hw_sampling["population_size"] = 1

#dice_factor_list = range(1, 150, 50)
#dice_factor_list = [1]
sw_model = "gables_inspired_exact"  # [gables_inspired_exact, gables_inspired, sequential] the diff is that exact replicates the PEs to solve the PA DRVR preemption issue
#sw_model = "sequential"  # read, execute write done in this order instead of simultenously
#if not sw_model == "gables_inspired":
#    dice_factor_list = [1]

if VIS_GR_PER_GEN: VIS_GR_PER_ITR = True

if VIS_ALL:
    DEBUG = True; DEBUG_FIX = True; VIS_GR_PER_GEN = True; VIS_GR_PER_ITR = True; VIS_PROFILE = True
    VIS_FINAL_RES = True;

# visualization
hw_graphing_mode = "block_task" # block, block_extra, block_task
stats_output = "stats"

# clustering
ic_mig_clustering = "data_sharing" # how to pick the pe, mem tuples choose between ["data_sharing", random]
tasks_clustering_data_sharing_method = "task_dep"

# area
statically_sized_blocks = ["gpp"]  # these blocks size is predeteremined (as opposed to memory, buses and ips that
                                   # require dynamic size based on the mapping
zero_sized_blocks = ["ic"]         # blocks to ignore for now  # TODO: figure ic size later



DMA_mode = "serialized_read_write"  # [serialized_read_write, parallel_read_write]
#DMA_mode = "parallelized_read_write"  # [serialized_read_write, parallelized_read_write]

# power  collection period (how often to divide energy). it's measured in seconds
#PCP = .0001
PCP = .01

budget_fitting_coeff = .9999 # used to make sure the slack values are uses in a way to bring the power and latency just beneath budget

# soource, sink
#souurce_memory_work = 81920 # in bytes
proj_name = "simple_multiple_hops"
proj_name = "SLAM"

# PA (platform) conversion files
hw_yaml = "hw.yml"
pa_des_top_name = "my_top"
pa_space_distance = 1
pa_push_distance = 4
max_budget_coeff = 50

# check pointing and reading from checkpoins
latest_visualization = os.path.join(home_dir, 'data_collection/data/latest_visualization')
check_point_folder = os.path.join(home_dir, 'data_collection/data/check_points')
replay_folder_base = os.path.join(home_dir, 'data_collection/data/replayer')
database_csv_folder = os.path.join(home_dir, 'specs/database_csvs/')  # where all the library input are located

axis_unit = {"area": "mm2", "power": "mW", "latency": "s"}
database_data_dir = os.path.join(home_dir, "specs", "database_data")
transaction_base_simulation = False   # do not set to true. It doesn't work


# CACTI
use_cacti = True and not RUN_VERIFICATION_AT_ALL # if True, use cacti. You have to have cacti installed.j
#use_cacti = True
cact_bin_addr = CC.cact_bin_addr
cacti_param_addr = CC.cacti_param_addr
cacti_data_log_file = CC.cacti_data_log_file
cacti_input_col_order = ["mem_subtype", "mem_size"]
cacti_output_col_order = ["read_energy_per_byte", "write_energy_per_byte", "area"]
cacti_min_memory_size_in_bytes = 2048 # bellow this value cacti errors out. We can play with burst size and page size to fix this though

#ACC_coeff = 128  # comparing to what we have parsed, how much to modify. This is just for some exploration purposes
	       # It should almost always set to 1


transformation_selection_mode = "random"  # choose from {random, arch-aware}

all_available_transformations = ["migrate", "swap", "split", "split_swap"]#, "transfer", "routing"]
if RUN_VERIFICATION_AT_ALL:
    all_available_transformations = ["migrate", "swap", "split", "split_swap"]

min_mem_size = {"sram": 256000, "dram":256000}

dram_stacked = True
parallelism_analysis = "dynamic" # choose from ["dynamic", "static"]  # at the moment static is not working, something to do with the task and task sync
                                 # and read to being present after unloading


heuristic_scaling_study = True
print_info_regularly = False

out_of_memory_percentage = 93
default_cmd_queue_size = 16
default_data_queue_size = 16
#default_burst_size = 128
default_burst_size = 256
#default_cmd_queue_size = 16
#default_data_queue_size = 16
#default_burst_size = 64


memory_conscious = True
