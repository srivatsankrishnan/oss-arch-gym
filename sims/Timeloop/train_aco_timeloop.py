import os
os.sys.path.insert(0, os.path.abspath('../../'))
print(os.sys.path)
from aco.DeepSwarm.deepswarm.backends import Dataset, TimeloopBackend
from aco.DeepSwarm.deepswarm.deepswarm import DeepSwarm

import argparse
import configparser
import numpy as np
import pandas as pd
import sys
import os
import configs.arch_gym_configs as arch_gym_configs

from absl import flags
from absl import app

# get the base directory from the file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR: ", BASE_DIR)

flags.DEFINE_string("script", str(BASE_DIR) + "/script", "Path to the script")
flags.DEFINE_string("output", str(BASE_DIR) + "/output", "Path to the output")
flags.DEFINE_string("arch", str(BASE_DIR) + "/arch", "Path to the arch")
flags.DEFINE_string("mapper", str(BASE_DIR) + "/mapper", "Path to the mapper")
flags.DEFINE_string("workload", str(BASE_DIR) + "/layer_shapes/AlexNet", "Path to the workload")
flags.DEFINE_string("summary_dir", ".", "Path to the log")
flags.DEFINE_string('traject_dir', 'aco_trajectories', 'Directory to trajectory data.')
flags.DEFINE_string('aco_log_dir', 'aco_logs', 'Directory to store logs.')

flags.DEFINE_string("params_file", str(BASE_DIR) + "/parameters.ini", "Path to the parameters file")

flags.DEFINE_string("runtime", "docker", "Runtime to use: docker or singularity")

flags.DEFINE_bool('use_envlogger', False, 'Whether to use envlogger.')

flags.DEFINE_float('target_energy', 20444.2, 'Target energy value.')
flags.DEFINE_float('target_area', 1.7255, 'Target area value.')
flags.DEFINE_float('target_cycles', 6308563, 'Target cycles value.')

flags.DEFINE_float('evaporation', 0.1, 'Evaporation value for pheromone.')
flags.DEFINE_integer('ant_count', 16, 'Number of Ants.')
flags.DEFINE_float('greediness', 0.5, 'How greedy you want the ants to be?.')
flags.DEFINE_integer('depth', 4, 'Depth of the network.')
flags.DEFINE_string('reward_formulation', 'latency', 'Reward formulation to use.')
FLAGS = flags.FLAGS


def main(_):
    
    # experiment name
    if "AlexNet" in FLAGS.workload:
        workload = "AlexNet"
    elif "resnet" in FLAGS.workload:
        workload = "ResNet"
    elif "mobilenet" in FLAGS.workload:
        workload = "mobilenet"

    # Experiment name (combine ant_count, greediness and evaporation
    exp_name = str(workload)+"_ant_count_" + str(FLAGS.ant_count) + "_greediness_" + str(FLAGS.greediness) + "_evaporation_" + str(FLAGS.evaporation) + "_depth_" + str(FLAGS.depth)
    
    # create trajectory directory (current dir + traject_dir)
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)
    
    # create log directory (current dir + log_dir)
    log_dir = os.path.join(FLAGS.summary_dir, FLAGS.aco_log_dir, FLAGS.reward_formulation, exp_name)

    print("Trajectory directory:", traject_dir)
    print("Log directory:", log_dir)
    print("Experiment name:", exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    target_val = [FLAGS.target_area, FLAGS.target_energy, FLAGS.target_cycles]

    x_train = np.array([1, 2, 3, 4, 5])

    dataset = Dataset(training_examples=x_train, training_labels=None,
                      testing_examples=x_train, testing_labels=None)
    backend = TimeloopBackend(dataset, script_dir=FLAGS.script, output_dir=FLAGS.output, arch_dir=FLAGS.arch,
                              mapper_dir = FLAGS.mapper, workload_dir=FLAGS.workload,
                              target_val=target_val, log_dir=log_dir, reward_formulation=FLAGS.reward_formulation,
                              use_envlogger=FLAGS.use_envlogger, exp_name=exp_name)
    
    deepswarm = DeepSwarm(backend=backend)

    topology = deepswarm.find_topology()

if __name__ == '__main__':
   app.run(main)