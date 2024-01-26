#!/usr/bin/env python3

import numpy as np
import pandas as pd
import time
import os
import sys

os.sys.path.insert(0, os.path.abspath('../../'))
print(os.sys.path)
from aco.DeepSwarm.deepswarm.backends import Dataset, AstraSimBackend
from aco.DeepSwarm.deepswarm.deepswarm import DeepSwarm
from arch_gym.envs.envHelpers import helpers

import configparser
from absl import flags
from absl import app

flags.DEFINE_float('evaporation', 0.1, 'Evaporation value for pheromone.')
flags.DEFINE_integer('ant_count', 2, 'Number of Ants.')
flags.DEFINE_float('greediness', 0.5, 'How greedy you want the ants to be?.')
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
flags.DEFINE_bool('congestion_aware', False, "astra-sim congestion aware or not")
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

    dataset = Dataset(training_examples=x_train, training_labels=None, testing_examples=x_train, testing_labels=None)
    
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

    if VERSION == 1:
        network_file = os.path.join(networks_folder, "analytical/4d_ring_fc_ring_switch.json")
    else:
        network_file = os.path.join(proj_root_path, FLAGS.network)

    action_dict = {}
    action_dict['network'] = astraSim_helper.parse_network_astrasim(network_file, action_dict, VERSION)

    if VERSION == 1:
        dimension = action_dict['network']["dimensions-count"]
    else:
        dimension = len(action_dict['network']["topology"])

    if "dimensions-count" in network_knob:
        dimensions = sorted(list(network_knob["dimensions-count"][0]))
    else:
        dimensions = [dimension]
    
    for d in dimensions:
        backend = AstraSimBackend(dataset, 
                                exp_name=exp_name,
                                log_dir=log_dir,
                                traject_dir=traject_dir,
                                reward_formulation= FLAGS.reward_formulation,
                                use_envlogger=FLAGS.use_envlogger,
                                VERSION=VERSION,
                                knobs_spec=FLAGS.knobs,
                                network=FLAGS.network,
                                system=FLAGS.system,
                                workload=FLAGS.workload_file,
                                congestion_aware=FLAGS.congestion_aware,
                                dimension=d)
        deepswarm = DeepSwarm(backend=backend)
        
        topology = deepswarm.find_topology()
    
    

if __name__ == '__main__':
   app.run(main)