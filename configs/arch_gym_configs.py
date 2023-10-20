import os
import sys
import psutil

settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
proj_root_path = os.path.abspath(settings_dir_path + "/..")

os.sys.path.insert(0, proj_root_path)
os.sys.path.insert(0, proj_root_path + "/arch_gym")
os.sys.path.insert(0, proj_root_path + "/arch_gym/sims")
os.sys.path.insert(0, proj_root_path + "/arch_gym/sims/Timeloop")
os.sys.path.insert(0, proj_root_path + "/arch_gym/sims/Sniper")
os.sys.path.insert(0, proj_root_path + "/arch_gym/sims/DRAM")

os.sys.path.insert(0,proj_root_path+ "/arch_gym/envs")
'''
def check_paths(paths):
for path in paths:
if not os.path.exists(path):
# print in red color if there is an error
print("\033[91m", end="")
print("Path: {} does not exist".format(path))
else:
# print in green color if there is no error
print("\033[92m", end="")
print("Path: {} exists".format(path))
'''

###############################
#  Algorithm Configurations   #
###############################
# ACO
aco_config = "default_astrasim.yaml"
aco_batch_mode = False
aco_base_path = proj_root_path

ant_count = [2, 4, 8, 16, 32, 64] # 6
evaporation = [0.1, 0.25, 0.5, 0.75, 1.0] # 5
greediness = [0.0, 0.25, 0.5, 0.75, 1.0] # 5
depth= [2, 4, 8, 16] # 4

# GA
ga_batch_mode = False
num_agents = [2, 4, 8, 16, 32, 64]
num_iter_ga = [8, 16, 32, 64]
prob_mut = [0.01, 0.05, 0.001]

# BO
num_iter_bo = [16, 32, 64]
rand_state_bo = [1, 2, 3, 4, 5]

# Random Walk
num_steps = [10000, 20000]

# Reinforcement Learning PPO
rl_agent = False

########################
#  Target Spec Sniper  #
########################

# Gainestown latency for 600
# 30% improvemnent in latency
# 15% improvement in area
# 25% improvement in power

target_latency  = 23160713.1
target_power   = 30
target_area   = 81

################
#    DRAMSys   #
################

dram_mem_controller_config = os.path.join(proj_root_path, "sims/DRAM/DRAMSys/library/resources/configs/mcconfigs")
dram_mem_controller_config_file = os.path.join(dram_mem_controller_config, "policy.json")
binary_name = "DRAMSys"
exe_path = os.path.join(proj_root_path, "sims/DRAM/binary/DRAMSys")
sim_config = os.path.join(proj_root_path, "sims/DRAM/DRAMSys/library/simulations/ddr3-example.json")
experiment_name = "random_walk.csv"
logdir = os.path.join(proj_root_path, "logs")
dramsys_envlogger_path = os.path.join(proj_root_path, "sims/DRAM/envlogger")
target_power = 1 # mw
target_latency = 0.1 # ns
# Make sure these files exists
dram_sys_paths = []
dram_sys_paths.append(dram_mem_controller_config)
dram_sys_paths.append(exe_path)
dram_sys_paths.append(sim_config)
dram_sys_paths.append(dram_mem_controller_config_file)
dram_sys_paths.append(logdir)

dram_sys_workload = ['stream.stl', 'random.stl', 'cloud-1.stl', 'cloud-2.stl']
#check_paths(dram_sys_paths)
################
#    Sniper    #
################

sniper_config = os.path.join(proj_root_path, "sims/Sniper/arch_gym_x86.cfg")
sniper_binary_name = "simulate_benchmark.py"
sniper_binary_path = os.path.join(proj_root_path, "sims/Sniper")
sniper_logdir = os.path.join(proj_root_path, "sims/Sniper/logs")
sniper_workload = "600"
sniper_numcores = str(psutil.cpu_count())
sniper_metric_log = "sniper_metric_log.csv"
spec_workload = "602"
sniper_envlogger_path = os.path.join(proj_root_path, "sims/Sniper/envlogger")
# Todo: Set this to a random string when not using Sniper
sniper_mode = "batch"
dummy_power_file = os.path.join(proj_root_path, "sims/Sniper/")

# Make sure these files exists
sniper_sim_paths = []
sniper_sim_paths.append(sniper_config)
sniper_sim_paths.append(sniper_binary_path)
sniper_sim_paths.append(sniper_logdir)
sniper_sim_paths.append(dummy_power_file)
#check_paths(sniper_sim_paths)

################
#    Timeloop  #
################

timeloop_binary_name = "simulate_timeloop.py"
timeloop_binary_path = os.path.join(proj_root_path, "sims/Timeloop/")
timeloop_parameters = os.path.join(proj_root_path, "sims/Timeloop/parameters.ini")
timeloop_scriptdir = os.path.join(proj_root_path, "sims/Timeloop/script")
timeloop_outputdir = os.path.join(proj_root_path, "sims/Timeloop/output")
timeloop_archdir = os.path.join(proj_root_path, "sims/Timeloop/arch")
timeloop_mapperdir = os.path.join(proj_root_path, "sims/Timeloop/mapper")
timeloop_workloaddir = os.path.join(proj_root_path, "sims/Timeloop/layer_shapes/AlexNet")
timeloop_numcores = str(psutil.cpu_count())

# Make sure these files exists
timeloop_sim_paths = []
timeloop_sim_paths.append(timeloop_binary_path)
timeloop_sim_paths.append(timeloop_scriptdir)
timeloop_sim_paths.append(timeloop_outputdir)
timeloop_sim_paths.append(timeloop_archdir)
timeloop_sim_paths.append(timeloop_mapperdir)
timeloop_sim_paths.append(timeloop_workloaddir)
#check_paths(timeloop_sim_paths)

##########################
#  Target Spec Timeloop  #
##########################

# 30% improvement in energy
# 15% improvement in area
# 20% improvement in cycles

target_energy_improv = 0.3
target_area_improv   = 0.15
target_cycle_improv  = 0.2

target_energy = 29206 * (1 - target_energy_improv)
target_area   = 2.03 * (1 - target_area_improv)
target_cycles = 7885704 * (1 - target_cycle_improv)

##########################
#  Target Spec Mastero   #
##########################
mastero_model_path = os.path.join(proj_root_path, "sims/gamma/data/model")
exe_file = os.path.join(proj_root_path, "sims/gamma/cost_model/maestro")
aco_config_file = os.path.join(proj_root_path, "settings/default_maestro.yaml")

# switch back to default color
print("\033[0m", end="")

