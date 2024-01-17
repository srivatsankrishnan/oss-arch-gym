
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
#  ACO Configuration   #
########################
aco_config = "default_astrasim.yaml"
aco_batch_mode = False
aco_base_path = proj_root_path

ant_count = [2, 4, 8, 16, 32, 64] # 6
evaporation = [0.1, 0.25, 0.5, 0.75, 1.0] # 5
greediness = [0.0, 0.25, 0.5, 0.75, 1.0] # 5
depth= [2, 4, 8, 16] # 4
