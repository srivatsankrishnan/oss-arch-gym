#!/usr/bin/env python3

import numpy as np
import pandas as pd
import time
import os
import sys

os.sys.path.insert(0, os.path.abspath('../../'))
print(os.sys.path)
from arch_gym.envs.envHelpers import helpers

import configparser
import yaml
import json
import subprocess
from absl import flags
from absl import app

flags.DEFINE_integer('ant_count', 2, 'Number of Ants.')
flags.DEFINE_float('greediness', 0.5, 'How greedy you want the ants to be?.')
flags.DEFINE_float('decay', 0.1, 'Decay rate for pheromone.')
flags.DEFINE_float('evaporation', 0.25, 'Evaporation value for pheromone.')
flags.DEFINE_float('start', 0.1, 'Start value for pheromone.')
flags.DEFINE_string('traject_dir', './all_logs/aco_trajectories', 'Directory to trajectory data.')
flags.DEFINE_string('aco_log_dir', './all_logs/aco_logs', 'Directory to store logs.')
flags.DEFINE_string('workload', 'stream.stl', 'Which workload to run')
flags.DEFINE_string('exp_config_file', 'exp_config.ini', 'Experiment config file.')
flags.DEFINE_integer('depth', 10, 'Depth of the network.')
flags.DEFINE_string('summary_dir', '.', 'Directory to store summaries.')
flags.DEFINE_string('reward_formulation', 'cycles', 'Reward formulation to use.')
flags.DEFINE_bool('use_envlogger', True, 'Use EnvLogger to log environment data.')
flags.DEFINE_string('knobs', 'astrasim_220_example/knobs.py', "path to knobs spec file")
flags.DEFINE_string('network', 'astrasim_220_example/network_input.yml', "path to network input file")
flags.DEFINE_string('system', 'astrasim_220_example/system_input.json', "path to system input file")
flags.DEFINE_string('workload_file', 'astrasim_220_example/workload_cfg.json', "path to workload input file")
flags.DEFINE_bool('congestion_aware', True, "astra-sim congestion aware or not")
# FLAGS.workload_file = astrasim_220_example/workload_cfg.json if GENERATE_WORKLOAD = True
# FLAGS.workload_file = astrasim_220_example/workload-et/generated if GENERATE_WORKLOAD = False

FLAGS = flags.FLAGS

# define AstraSim version
VERSION = 2
astraSim_helper = helpers()

def main(_):
    # Dummy "training" input for POC
    x_train = np.array([1,2,3,4,5])
    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_train))
    
    # Experiment name (combine ant_count, greediness and evaporation
    exp_name = str(FLAGS.workload)+"_ant_count_" + str(FLAGS.ant_count) + "_greediness_" + str(FLAGS.greediness) + "_evaporation_" + str(FLAGS.evaporation) + "_depth_" + str(FLAGS.depth)
    
    # create trajectory directory (current dir + traject_dir)
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)
    
    # create log directory (current dir + log_dir)
    log_dir = os.path.join(FLAGS.summary_dir, FLAGS.aco_log_dir, FLAGS.reward_formulation, exp_name)

    settings_file_path = os.path.realpath(__file__)
    settings_dir_path = os.path.dirname(settings_file_path)
    proj_root_path = os.path.abspath(settings_dir_path)

    astrasim_archgym = os.path.join(proj_root_path, "astrasim-archgym")
    archgen_v1_knobs = os.path.join(astrasim_archgym, "dse/archgen_v1_knobs")
    knobs_spec = os.path.join(proj_root_path, FLAGS.knobs)
    networks_folder = os.path.join(astrasim_archgym, "themis/inputs/network")

    system_knob, network_knob, workload_knob = astraSim_helper.parse_knobs_astrasim(knobs_spec)
    dicts = [(system_knob, 'system'), (network_knob, 'network'), (workload_knob, 'workload')]

    if VERSION == 1:
        network_file = os.path.join(networks_folder, "analytical/4d_ring_fc_ring_switch.json")
    else:
        network_file = os.path.join(proj_root_path, FLAGS.network)

    action_dict = {}
    action_dict['network'] = astraSim_helper.parse_network_astrasim(network_file, action_dict, VERSION)

    if VERSION == 1:
        dimension_count = action_dict['network']["dimensions-count"]
    else:
        dimension_count = len(action_dict['network']["topology"])

    if "dimensions-count" in network_knob:
        dimensions = sorted(list(network_knob["dimensions-count"][0]))
    else:
        dimensions = [dimension_count]

    # iterate through dimensions in dimensions-count
    for dimension in dimensions:

        root_path = os.path.join(proj_root_path, '../../settings')
        yaml_path = os.path.join(root_path, "default_astrasim.yaml")

        print("PATH: ", yaml_path)
        # set to everything in yaml file's default (last iteration)
        data = yaml.load(open(yaml_path), Loader=yaml.Loader)
        # write knobs to yaml file
        # Rewrite the flags
        data['DeepSwarm']['aco']["ant_count"] = FLAGS.ant_count
        data['DeepSwarm']['aco']["greediness"] = FLAGS.greediness
        data['DeepSwarm']['aco']["pheromone"]["decay"] = FLAGS.decay
        data['DeepSwarm']['aco']["pheromone"]["evaporation"] = FLAGS.evaporation
        data['DeepSwarm']['aco']["pheromone"]["start"] = FLAGS.start
        data['DeepSwarm']["max_depth"] = FLAGS.depth
        # Rewrite the attributes AND CORRECT THE DIMENSIONS
        data['Nodes']['ArchParamsNode']['attributes'] = {}
        for dict_type, dict_name in dicts:
            for knob in dict_type:
                if knob == "dimensions-count":
                    continue
                # from hyphen to CamelCase
                knob_converted = astraSim_helper.convert_knob_ga_astrasim(knob)
                if isinstance(dict_type[knob][0], set):
                    if dict_type[knob][1] == "FALSE":
                        for i in range(1, dimension + 1):
                            knob_dimension = knob_converted + str(i)
                            list_sorted = sorted(list(dict_type[knob][0]))
                            data['Nodes']['ArchParamsNode']['attributes'][knob_dimension] = [i for i in list_sorted]
                    else:
                        list_sorted = sorted(list(dict_type[knob][0]))
                        data['Nodes']['ArchParamsNode']['attributes'][knob_converted] = [i for i in list_sorted]
                else:
                    if dict_type[knob][1] == "FALSE":
                        for i in range(1, dimension + 1):
                            knob_dimension = knob_converted + str(i)
                            data['Nodes']['ArchParamsNode']['attributes'][knob_dimension] = [i for i in range(dict_type[knob][0][0], dict_type[knob][0][1] + 1)]
                    else:
                        data['Nodes']['ArchParamsNode']['attributes'][knob_converted] = [i for i in range(dict_type[knob][0][0], dict_type[knob][0][1] + 1)]
        
        print("YAML DATA: ", data)

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
            f.flush()

        time.sleep(2)

        train_aco_helper = os.path.join(proj_root_path, "trainACOAstraSimHelper.py")

        command = ['python', train_aco_helper, f'--dimension={dimension}', f'--reward_formulation={FLAGS.reward_formulation}',
                    f'--use_envlogger={FLAGS.use_envlogger}', f'--knobs={FLAGS.knobs}', f'--network={FLAGS.network}', 
                    f'--system={FLAGS.system}', f'--workload_file={FLAGS.workload_file}', f'congestion_aware={FLAGS.congestion_aware}',
                    f'--ant_count={FLAGS.ant_count}', f'--greediness={FLAGS.greediness}', f'--decay={FLAGS.decay}',
                    f'--evaporation={FLAGS.evaporation}', f'--start={FLAGS.start}', f'--depth={FLAGS.depth}',
                    f'--traject_dir={FLAGS.traject_dir}', f'--aco_log_dir={FLAGS.aco_log_dir}', f'--summary_dir={FLAGS.summary_dir}',
                    f'--workload={FLAGS.workload}'
                    ]
        subprocess.run(command)

    # for d in dimensions:


    

if __name__ == '__main__':
   app.run(main)