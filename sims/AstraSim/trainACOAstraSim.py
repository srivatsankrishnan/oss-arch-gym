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

import configparser
from absl import flags
from absl import app

flags.DEFINE_float('evaporation', 0.1, 'Evaporation value for pheromone.')
flags.DEFINE_integer('ant_count', 2, 'Number of Ants.')
flags.DEFINE_float('greediness', 0.5, 'How greedy you want the ants to be?.')
flags.DEFINE_string('traject_dir', 'aco_trajectories', 'Directory to trajectory data.')
flags.DEFINE_string('aco_log_dir', 'aco_logs', 'Directory to store logs.')
flags.DEFINE_string('workload', 'stream.stl', 'Which workload to run')
flags.DEFINE_string('exp_config_file', 'exp_config.ini', 'Experiment config file.')
flags.DEFINE_integer('depth', 1, 'Depth of the network.')
flags.DEFINE_string('summary_dir', '.', 'Directory to store summaries.')
flags.DEFINE_string('reward_formulation', 'power', 'Reward formulation to use.')
flags.DEFINE_bool('use_envlogger', True, 'Use EnvLogger to log environment data.')
FLAGS = flags.FLAGS


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

    print("Trajectory directory:", traject_dir)
    print("Log directory:", log_dir)
    print("Experiment name:", exp_name)
    
    backend = AstraSimBackend(dataset, 
                            exp_name=exp_name,
                            log_dir=log_dir,
                            traject_dir=traject_dir,
                            reward_formulation= FLAGS.reward_formulation,
                            use_envlogger=FLAGS.use_envlogger)
    deepswarm = DeepSwarm(backend=backend)
    
    topology = deepswarm.find_topology()
    
    

if __name__ == '__main__':
   app.run(main)