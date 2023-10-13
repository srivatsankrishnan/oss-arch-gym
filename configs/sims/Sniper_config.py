import os
import sys
import psutil

# Get relative paths to current file & dir to construct oss-arch-gym project repo path
curr_config_file_path = os.path.realpath(__file__)
curr_config_dir_path = os.path.dirname(curr_config_file_path)
proj_root_path = os.path.abspath(curr_config_dir_path + "/../..")

# Add python modules from other dirs 
os.sys.path.insert(0, proj_root_path)
os.sys.path.insert(0, proj_root_path  + "/arch_gym")
os.sys.path.insert(0, proj_root_path  + "/arch_gym/sims/Sniper")
os.sys.path.insert(0, proj_root_path  + "/arch_gym/envs")

#############################
#   Sniper Configurations   #
#############################
sniper_config = os.path.join(proj_root_path, "sims/Sniper/arch_gym_x86.cfg")
sniper_binary_name = "simulate_benchmark.py"
sniper_binary_path = os.path.join(proj_root_path, "sims/Sniper")
sniper_logdir = os.path.join(proj_root_path, "sims/Sniper/logs")
sniper_workload = "600"
sniper_numcores = str(psutil.cpu_count())
sniper_metric_log = "sniper_metric_log.csv"
spec_workload = "602"
sniper_envlogger_path = os.path.join(proj_root_path, "sims/Sniper/envlogger")
sniper_mode = "batch"
dummy_power_file = os.path.join(proj_root_path, "sims/Sniper/")

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
