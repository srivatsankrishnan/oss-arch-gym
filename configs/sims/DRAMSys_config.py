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
os.sys.path.insert(0, proj_root_path  + "/arch_gym/sims/DRAM")
os.sys.path.insert(0, proj_root_path  + "/arch_gym/envs")

##############################
#   DRAMSys Configurations   #
##############################
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
dram_sys_workload = ['stream.stl', 'random.stl', 'cloud-1.stl', 'cloud-2.stl']
