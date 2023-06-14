import os
import sys

import _pickle as cPickle
import datetime
settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
os.sys.path.insert(0, settings_dir_path)

os.sys.path.insert(0, settings_dir_path + '/../../')
os.sys.path.insert(0, settings_dir_path + '/../../Project_FARSI')
os.sys.path.insert(0, settings_dir_path + '/../../Project_FARSI/data_collection/collection_utils')

from Project_FARSI import *
import home_settings
from top.main_FARSI import run_FARSI_only_simulation
from top.main_FARSI import run_FARSI
from top.main_FARSI import run_FARSI
from top.main_FARSI import set_up_FARSI_with_arch_gym
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
import data_collection.collection_utils.what_ifs.FARSI_what_ifs as wf


from absl import flags
from absl import app
from absl import logging

os.sys.path.insert(0, os.path.abspath('../../'))
from configs import arch_gym_configs
from arch_gym.envs.envHelpers import helpers
from arch_gym.envs import FARSI_sim_wrapper
import envlogger
import numpy as np
import pandas as pd

flags.DEFINE_string('workload', 'edge_detection', 'Which FARSI workload to run?')
flags.DEFINE_integer('num_steps', 100, 'Number of training steps.')
flags.DEFINE_integer('num_episodes', 2, 'Number of training episodes.')
flags.DEFINE_string('traject_dir', 'random_walker_trajectories', 'Directory to save the dataset.')
flags.DEFINE_string('summary_dir', '.', 'Directory to save the summary.')
flags.DEFINE_string('reward_formulation', 'power', 'Which reward formulation to use?')
flags.DEFINE_bool('use_envlogger', False, 'Whether to use envlogger.')
FLAGS = flags.FLAGS

def log_fitness_to_csv(filename, fitness_dict):

        # create filename directory if it doesn't exist
        if not os.path.exists(filename):
            os.makedirs(filename)

        df = pd.DataFrame([fitness_dict['reward']])
        csvfile = os.path.join(filename, "fitness.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

        # append to csv
        df = pd.DataFrame([fitness_dict])
        csvfile = os.path.join(filename, "trajectory.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')


def wrap_in_envlogger(env, envlogger_dir):
    metadata = {
        'agent_type': 'RandomWalker',
        'num_steps': FLAGS.num_steps,
        'env_type': type(env).__name__,
    }
    if FLAGS.use_envlogger:
        logging.info('Wrapping environment with EnvironmentLogger...')
        env = envlogger.EnvLogger(env,
                                  data_directory=envlogger_dir,
                                  max_episodes_per_file=1000,
                                  metadata=metadata)
        logging.info('Done wrapping environment with EnvironmentLogger.')
        return env
    else:
        return env


def main(_):

    # experiment name 
    exp_name = str(FLAGS.workload)+"_num_steps_" + str(FLAGS.num_steps) + "_num_episodes_" + str(FLAGS.num_episodes)

    reward_formulation_list = FLAGS.reward_formulation.split(" ")

    # join the list of reward formulations with a _
    reward_formulation = "_".join(reward_formulation_list)

    # append logs to base path
    log_path = os.path.join(FLAGS.summary_dir, 'random_walker_logs', reward_formulation, exp_name)

    # get the current working directory and append the exp name
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, reward_formulation, exp_name)

    # check if log_path exists else create it
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if FLAGS.use_envlogger:
        if not os.path.exists(traject_dir):
            os.makedirs(traject_dir)

    env = FARSI_sim_wrapper.make_FARSI_sim_env(reward_formulation = FLAGS.reward_formulation, workload=FLAGS.workload)
    env = wrap_in_envlogger(env, traject_dir)
    FARSI_sim_helper = helpers()
    design_space_mode = "limited"  # ["limited", "comprehensive"]
    SOC_design_space = FARSI_sim_helper.gen_SOC_design_space(env, design_space_mode)
    encoding_dictionary = FARSI_sim_helper.gen_SOC_encoding(env, SOC_design_space)
    
    fitness_hist = {}
    
    for i in range(FLAGS.num_episodes):
        logging.info('Episode %r', i)
        for step in range(FLAGS.num_steps):
            # generate actions randomly
            # completely random
            check_system = True
            action_encoded = FARSI_sim_helper.random_walk_FARSI_array_style(env, encoding_dictionary, check_system)

            # serialize to convert to string/dictionary
            action= FARSI_sim_helper.action_decoder_FARSI(action_encoded, encoding_dictionary)

            # pass in the move object into the step
            _, reward, _, info = env.step(action)
            
            action_dict_for_logging={}
            for key in action.keys():
                if "encoding" not in key:
                    action_dict_for_logging[key] = action[key]
            
            if reward is not None: # reward is None on resetting of env and do not need to log this
                fitness_hist["action"] = action_dict_for_logging
                fitness_hist["reward"] = reward.item()
                fitness_hist["obs"] = [metric.item() for metric in info]
                log_fitness_to_csv(log_path, fitness_hist)

 
if __name__ == '__main__':
   app.run(main)

