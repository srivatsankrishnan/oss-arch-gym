#!/usr/bin/env python3

import configparser
import os
import sys
import argparse
import envlogger

import numpy as np
import pandas as pd

# from timeloop_wrapper import TimeloopWrapper
from absl import logging
from timeloop_wrapper_singularity import TimeloopWrapper
from process_params import TimeloopConfigParams
os.sys.path.insert(0, os.path.abspath('../../'))
from arch_gym.envs.TimeloopEnv import TimeloopEnv
from arch_gym.envs.timeloop_acme_wrapper import make_timeloop_env
from arch_gym.envs.envHelpers import helpers


from absl import app
from absl import flags
from absl import logging

# get the base directory from the file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR: ", BASE_DIR)

flags.DEFINE_string("script", str(BASE_DIR) + "/script", "Path to the script")
flags.DEFINE_string("output", str(BASE_DIR) + "/output", "Path to the output")
flags.DEFINE_string("arch", str(BASE_DIR) + "/arch", "Path to the arch")
flags.DEFINE_string("mapper", str(BASE_DIR) + "/mapper", "Path to the mapper")
flags.DEFINE_string("workload", str(BASE_DIR) + "/layer_shapes/AlexNet", "Path to the workload")
flags.DEFINE_string("summary_dir", ".", "Path to the log")
flags.DEFINE_string("runtime", "docker", "Runtime to use: docker or singularity")
flags.DEFINE_string("params_file", str(BASE_DIR) + "/parameters.ini", "Path to the parameters file")


flags.DEFINE_integer("num_episodes", 1, "Maximum number of episodes")
flags.DEFINE_integer("num_steps", 100, "Maximum number of steps per episode")
flags.DEFINE_float("target_energy", 20444.2, "Target energy")
flags.DEFINE_float("target_area", 1.7255, "Target area")
flags.DEFINE_float("target_cycles", 6308563, "Target cycles")
flags.DEFINE_integer("-save_actions", 0, "Save actions")
flags.DEFINE_string("reward_formulation", "energy", "Reward formulation")
flags.DEFINE_boolean("use_envlogger", False, "Use envlogger")
FLAGS = flags.FLAGS

class TimeloopRandomWalk():
    def __init__(self, script_dir, output_dir, arch_dir, mapper_dir, workload_dir, target_val, params_file, reward_formulation):
        '''Initializes parameters'''

        self.config_params_obj = TimeloopConfigParams(params_file)
        self.arch_params = self.config_params_obj.get_all_params()

        self.helper = helpers()

        # Initialize timeloop wrapper
        self.timeloop = TimeloopWrapper()

        # Initialize the environment
        self.env = TimeloopEnv(script_dir=script_dir, output_dir=output_dir,
                               arch_dir=arch_dir, mapper_dir=mapper_dir,
                               workload_dir=workload_dir, target_val=target_val, reward_formulation=reward_formulation)

        self.env_wrapper = make_timeloop_env(env=self.env)

    def take_random_action(self):
        '''Takes a random action on the given arch parameters'''

        action_dict = self.helper.decode_timeloop_action(
            self.env.action_space.sample() + 1)

        print("action_dict: ", action_dict)
        return action_dict

def wrap_in_envlogger(env, envlogger_dir):
    if FLAGS.use_envlogger:
        logging.info('Wrapping environment with EnvironmentLogger...')

        env = envlogger.EnvLogger(env,
                             data_directory=FLAGS.log,
                             max_episodes_per_file=1000,
                             metadata={
                                 'agent_type': 'random',
                                 'env_type': type(env).__name__,
                                 'num_episodes': FLAGS.num_episodes,
                             }) 

        logging.info('Done wrapping environment with EnvironmentLogger.')
        return env
    else:
        return env
def log_fitness_to_csv(filename, fitness_dict):
        df = pd.DataFrame([fitness_dict['reward']])
        csvfile = os.path.join(filename, "fitness.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

        # append to csv
        df = pd.DataFrame([fitness_dict])
        csvfile = os.path.join(filename, "trajectory.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

def main(_):
    target_val = np.array(
        [FLAGS.target_energy, FLAGS.target_area, FLAGS.target_cycles])

    # Initialize the random walker
    randomwalker = TimeloopRandomWalk(FLAGS.script, FLAGS.output, FLAGS.arch,
                                        FLAGS.mapper, FLAGS.workload, target_val, FLAGS.params_file, FLAGS.reward_formulation)

    fitness_hist = {}
    
    # experiment name
    if "AlexNet" in FLAGS.workload:
        workload = "AlexNet"
    elif "resnet" in FLAGS.workload:
        workload = "ResNet"
    elif "mobilenet" in FLAGS.workload:
        workload = "mobilenet"
    
    exp_name = str(workload)+"_num_steps_" + str(FLAGS.num_steps) + "_num_episodes_" + str(FLAGS.num_episodes)

    # append logs to base path
    log_path = os.path.join(FLAGS.summary_dir, 'random_walker_logs', FLAGS.reward_formulation, exp_name)

    
    # check if log_path exists else create it
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    env = randomwalker.env_wrapper
    env = wrap_in_envlogger(env, log_path)

    for i in range(FLAGS.num_episodes):
        logging.info('Episode %r', i)
        for step in range(FLAGS.num_steps):
            env.reset()
            # generate random actions
            action = randomwalker.env.action_space.sample() + 1
            print("action: ", len(action))
            
            _, reward, _, info = env.step(action)

            action_dict = randomwalker.helper.decode_timeloop_action(action)

            fitness_hist['reward'] = reward
            fitness_hist['action'] = action_dict
            fitness_hist['obs'] = info

            log_fitness_to_csv(log_path, fitness_hist)
if __name__ == "__main__":
    app.run(main)

