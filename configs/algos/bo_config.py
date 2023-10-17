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

##################################
#  BayesOpt(BO)  Configuration   #
##################################
num_iter_bo = [16, 32, 64]
rand_state_bo = [1, 2, 3, 4, 5]
