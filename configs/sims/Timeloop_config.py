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
os.sys.path.insert(0, proj_root_path  + "/arch_gym/sims/Timeloop")
os.sys.path.insert(0, proj_root_path  + "/arch_gym/envs")

###############################
#   Timeloop Configurations   #
###############################

timeloop_binary_name = "simulate_timeloop.py"
timeloop_binary_path = os.path.join(proj_root_path, "sims/Timeloop/")
timeloop_parameters  = os.path.join(proj_root_path, "sims/Timeloop/parameters.ini")
timeloop_scriptdir   = os.path.join(proj_root_path, "sims/Timeloop/script")
timeloop_outputdir   = os.path.join(proj_root_path, "sims/Timeloop/output")
timeloop_archdir     = os.path.join(proj_root_path, "sims/Timeloop/arch")
timeloop_mapperdir   = os.path.join(proj_root_path, "sims/Timeloop/mapper")
timeloop_workloaddir = os.path.join(proj_root_path, "sims/Timeloop/layer_shapes/AlexNet")
timeloop_numcores    = str(psutil.cpu_count())

##########################
#  Target Spec Timeloop  #
##########################

# 30% improvement in energy
# 15% improvement in area
# 20% improvement in cycles

target_energy_improv = 0.3
target_area_improv   = 0.15
target_cycle_improv  = 0.2

target_energy = 29206   * (1 - target_energy_improv)
target_area   = 2.03    * (1 - target_area_improv)
target_cycles = 7885704 * (1 - target_cycle_improv)

##########################
#   ACO  Configurations  #
##########################
# ACO
aco_config = "default_astrasim.yaml"