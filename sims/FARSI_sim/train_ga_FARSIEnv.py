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
import itertools
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
from sko.GA import GA
import matplotlib.colors as mcolors
import pandas as pd
import argparse, sys
import data_collection.collection_utils.what_ifs.FARSI_what_ifs as wf


from absl import flags
from absl import app
from absl import logging

os.sys.path.insert(0, os.path.abspath('../../'))
#from configs import arch_gym_configs
from arch_gym.envs.envHelpers import helpers
from arch_gym.envs import FARSI_sim_wrapper
import envlogger
import numpy as np
import pandas as pd

flags.DEFINE_bool('use_envlogger', False, 'Whether to use envlogger.')
flags.DEFINE_string('reward_formulation', 'power', 'Reward formulation to use')
flags.DEFINE_string('workload', 'edge_detection', 'Which Workload to run')
flags.DEFINE_integer('num_iter', 100, 'Number of training steps.')
flags.DEFINE_integer('num_agents', 32, 'Number of agents.')
flags.DEFINE_float('prob_mutation', 0.1, 'Probability of mutation.')
flags.DEFINE_string('traject_dir','ga_trajectories', 'Directory to save the dataset.')
flags.DEFINE_string('summary_dir', '.', 'Directory to save the summary.')

FLAGS = flags.FLAGS

def wrap_in_envlogger(env, envlogger_dir):
    metadata = {
        'agent_type': 'GA',
        'num_steps': FLAGS.num_iter,
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


def generate_run_directories():
    reward_formulation_list = FLAGS.reward_formulation.split(" ")

    # join the list of reward formulations with a _
    reward_formulation = "_".join(reward_formulation_list)
    
    # Construct the exp name from seed and num_iter
    exp_name = FLAGS.workload + "_num_iter_" + str(FLAGS.num_iter) + "_num_agents_" + str(FLAGS.num_agents) + "_prob_mut_" + str(FLAGS.prob_mutation)
  
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, reward_formulation, exp_name)
    
    # log directories for storing exp csvs
    exp_log_dir = os.path.join(FLAGS.summary_dir,"ga_logs", reward_formulation, exp_name)

    return traject_dir, exp_log_dir
    
def FARSI_optimization_function(p):
    '''
    This function is used to optimize the DRAM parameters. The default objective is to minimize. If you have a reward/fitness formulation
    that is to be maximized, you can simply return -1 * your_reward.
    '''
    rewards = []
    print("Agents Action", p)
    # instantiate the environment and the helpers

    env = FARSI_sim_wrapper.make_FARSI_sim_env(reward_formulation = FLAGS.reward_formulation, workload=FLAGS.workload)
    FARSI_sim_helper = helpers()
    design_space_mode = "limited"  # ["limited", "comprehensive"]
    SOC_design_space = FARSI_sim_helper.gen_SOC_design_space(env, design_space_mode)
    encoding_dictionary = FARSI_sim_helper.gen_SOC_encoding(env, SOC_design_space)

    traject_dir, exp_log_dir = generate_run_directories()
    
    env = wrap_in_envlogger(env, FLAGS.summary_dir)
    
    if FLAGS.use_envlogger:
        if not os.path.exists(traject_dir):
            os.makedirs(traject_dir)
    env.reset()

    # decode the actions
    action_dict = FARSI_sim_helper.action_decoder_FARSI(p, encoding_dictionary)
    
    # take a step in the environment
    _, reward, _, info = env.step(action_dict)
    
    action_dict_for_logging={}
    for key in action_dict.keys():
        if "encoding" not in key:
            action_dict_for_logging[key] = action_dict[key]
    
    fitness_dict = {}
    fitness_dict["action"] = action_dict_for_logging
    fitness_dict["reward"] = reward.item()
    fitness_dict["obs"] = [metric.item() for metric in info]

    # Convert dictionary to dataframe
    fitness_df = pd.DataFrame([fitness_dict], columns=["action", "reward", "obs"])

    # check if exp_log_dir exists
    if not os.path.exists(exp_log_dir):
        os.makedirs(exp_log_dir)

    # write it to csv file append mode
    fitness_df.to_csv(os.path.join(exp_log_dir, "fitness.csv"), mode='a', header=False, index=False)
    rewards.append(reward)
    
    return -1 * reward
    

def main(_):
    dummy_env = FARSI_sim_wrapper.make_FARSI_sim_env(reward_formulation = FLAGS.reward_formulation, workload=FLAGS.workload)
    FARSI_sim_helper = helpers()
    design_space_mode = "limited"  # ["limited", "comprehensive"]
    SOC_design_space = FARSI_sim_helper.gen_SOC_design_space(dummy_env, design_space_mode)
    encoding_dictionary = FARSI_sim_helper.gen_SOC_encoding(dummy_env, SOC_design_space)
    
    if FLAGS.workload == "audio_decoder":
        n_dim = 54
    elif FLAGS.workload == "hpvm_cava":
        n_dim = 36
    elif FLAGS.workload == "edge_detection":
        n_dim = 34

    ga = GA(
        func=FARSI_optimization_function, 
        n_dim=n_dim, 
        size_pop=FLAGS.num_agents,
        max_iter=FLAGS.num_iter,
        prob_mut=FLAGS.prob_mutation,
        lb=encoding_dictionary["encoding_flattened_lb"],
        ub=encoding_dictionary["encoding_flattened_ub"],
        precision=[1.0 for config in range(len(encoding_dictionary["encoding_flattened_lb"]))]
    )

    best_x, best_y = ga.run()

    # get directory names
    _, exp_log_dir = generate_run_directories()

    # check if exp_log_dir exists
    if not os.path.exists(exp_log_dir):
        os.makedirs(exp_log_dir)
    
    Y_history = pd.DataFrame(ga.all_history_Y)
    Y_history.to_csv(os.path.join(exp_log_dir, "Y_history.csv"))

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.savefig(os.path.join(exp_log_dir, "Y_history.png"))

if __name__ == '__main__':
   app.run(main)