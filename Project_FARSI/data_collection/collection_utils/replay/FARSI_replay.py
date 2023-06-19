#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import os
import sys
sys.path.append(os.path.abspath('./../'))
import home_settings
from DSE_utils.design_space_exploration_handler import  *
from specs.database_input import *
from replayer import *

# This files shows examples of how to setup the replayer

# ------------------------------
# Functionality:
#       list the subdirectories of a directory
# Variables:
#       result_base_dir: parent directory
#       prefix: prefix to add to the name of the subdirectories (to generate the address)
# ------------------------------
def list_dirs(result_base_dir, prefix):
    dirs = os.listdir(result_base_dir)
    result = []
    for dir in dirs:
        dir_full_addr = os.path.join(result_base_dir, dir)   # add path to each file
        if not os.path.isdir(dir_full_addr):
            continue
        result.append(os.path.join(prefix,dir, os.listdir(dir_full_addr)[0]))
    return result

replayer_obj = Replayer()


mode = "individual"   # individual, entire_folder


# individual replay
# TODO: clean this up, and add it as an option
"""
# individual replay
#des_folder_name = "/two_three_mem/data/04-29_14-08_22_5/04-29_14-08_22_5"
#des_folder_name = "/two_three_mem/data/04-29_14-07_00_0"
#des_folder_name = "/replay/two_bus/data/04-29_22-14_01_3/04-29_22-14_01_3"
#des_folder_name = "/two_bus/data/04-29_22-14_01_3/04-29_22-14_01_3"
#des_folder_name = "/05-08_18-43_07_13/05-08_18-43_07_13_13"
des_folder_name = "bus_slave_to_master_connection_bug/05-08_20-30_41_40/05-08_20-30_41_40_40"
des_folder_list = [des_folder_name]
"""

# batch replay
prefix = "DP_Stats_testing_Data/"
folder_to_look_into =config.replay_folder_base
des_folder_list = list_dirs(folder_to_look_into, prefix)

# iterate through the folder and call replay on it
for des_folder_name in des_folder_list:
    replayer_obj.replay(des_folder_name)
    replayer_obj.gen_pa()
