#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.append(os.path.abspath('./../'))
import home_settings
from top.main_FARSI import run_FARSI
from settings import config
import os
import itertools
# main function
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from visualization_utils.vis_hardware import *
import numpy as np
from specs.LW_cl import *
from specs.database_input import  *
import math
import matplotlib.colors as colors
#import pandas
import matplotlib.colors as mcolors
import pandas as pd
import argparse, sys
from specs import database_input


import FARSI_what_ifs


if __name__ == "__main__":
    # set the number of workers to be used (parallelism applied)
    current_process_id = 0
    total_process_cnt = 1
    system_workers = (current_process_id, total_process_cnt)

    # set the study type 
    study_type = "cost_PPA"
    study_subtype = "run"
    assert study_type in ["cost_PPA", "simple_run", "input_error_output_cost_sensitivity", "input_error_input_cost_sensitivity"]
    assert study_subtype in ["run", "plot_3d_distance"]

    # set result folder according to the time and the study type
    result_home_dir = os.path.join(config.home_dir, "data_collection/data/" + study_type)
    date_time = datetime.now().strftime('%m-%d_%H-%M_%S')
    result_folder = os.path.join(result_home_dir,
                                 date_time)

    # set the workload
    workloads = {"audio_decoder"}  # select from {"audio_decoder", "edge_detection", "hpvm_cava"}

    # set software hardware database population (refer to database.py for more information}
    sw_hw_database_population = {"db_mode": "parse", "hw_graph_mode": "generated_from_scratch", "workloads": workloads,
                                 "misc_knobs":{}}

    # run FARSI (a simple exploration study)
    FARSI_what_ifs.simple_run(result_folder, sw_hw_database_population, system_workers)