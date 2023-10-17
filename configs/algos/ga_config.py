
import os
import sys
import psutil

# Get relative paths to current file & dir to construct oss-arch-gym project repo path
curr_config_file_path = os.path.realpath(__file__)
curr_config_dir_path = os.path.dirname(curr_config_file_path)
proj_root_path = os.path.abspath(curr_config_dir_path + "/../..")

# Add python modules from other dirs 
os.sys.path.insert(0, proj_root_path)
os.sys.path.insert(0, proj_root_path + "/arch_gym")
os.sys.path.insert(0, proj_root_path + "/arch_gym/envs")

########################
#   GA Configuration   #
########################
ga_batch_mode = False
num_agents = [2, 4, 8, 16, 32, 64]
num_iter_ga = [8, 16, 32, 64]
prob_mut = [0.01, 0.05, 0.001]
