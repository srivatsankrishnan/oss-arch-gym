import os
import sys

import json
import numpy as np

from absl import flags
from absl import app
from absl import logging
os.sys.path.insert(0, os.path.abspath('../../../../'))
os.sys.path.insert(0, os.path.abspath('../../../../arch_gym'))
from sko.GA import GA
from configs import arch_gym_configs
from arch_gym.envs.envHelpers import helpers
from arch_gym.envs import maestero_wrapper
import envlogger

from configs import arch_gym_configs

import pandas as pd
import matplotlib.pyplot as plt

flags.DEFINE_integer('num_steps', 10, 'Number of training steps.')
flags.DEFINE_integer('num_agents', 4, 'Number of agents.')
flags.DEFINE_float('prob_mutation', 0.1, 'Probability of mutation.')
flags.DEFINE_string('workload','resnet18', 'ML model name')
flags.DEFINE_integer('layer_id', 2, 'Layer id')
flags.DEFINE_string('summary_dir', 'test', 'Directory to save the summary.')
flags.DEFINE_string('reward_formulation', 'latency', 'Reward formulation to use')
flags.DEFINE_string('traject_dir','ga_trajectories', 'Directory to save the dataset.')
flags.DEFINE_bool('use_envlogger', False, 'Whether to use envlogger.')

FLAGS = flags.FLAGS

def generate_run_directories():
    # Construct the exp name from seed and num_iter
    exp_name = FLAGS.workload + "_num_iter_" + str(FLAGS.num_steps) + "_num_agents_" + str(FLAGS.num_agents) + "_prob_mut_" + str(FLAGS.prob_mutation)
  
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)
    
    # log directories for storing exp csvs
    exp_log_dir = os.path.join(FLAGS.summary_dir,"ga_logs",FLAGS.reward_formulation, exp_name)

    return traject_dir, exp_log_dir


def log_fitness_to_csv(filename, fitness_dict):
        df = pd.DataFrame([fitness_dict['reward']])
        csvfile = os.path.join(filename, "fitness.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

        df = pd.DataFrame([fitness_dict['action']])
        csvfile = os.path.join(filename, "actions.csv")
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

def Mastero_optimization_function(p):
    
    env = maestero_wrapper.make_maestro_env(rl_form='random_walker')
    maestro_helpers = helpers()
    fitness_hist = {}

    traject_dir, exp_log_dir = generate_run_directories()
    
    # check if log_path exists else create it
    if not os.path.exists(exp_log_dir):
        os.makedirs(exp_log_dir)

    if FLAGS.use_envlogger:
        if not os.path.exists(traject_dir):
            os.makedirs(traject_dir)
    env = wrap_in_envlogger(env, traject_dir)
    
  
    # reset the environment
    env.reset()

    # take a step
    _, reward, c, info = env.step(p)
    
    fitness_dict = {}
    fitness_dict["action"] =p
    fitness_dict["reward"] = reward
    fitness_dict["obs"] = info

    # Convert dictionary to dataframe
    fitness_df = pd.DataFrame([fitness_dict], columns=["action", "reward", "obs"])

    # check if exp_log_dir exists
    if not os.path.exists(exp_log_dir):
        os.makedirs(exp_log_dir)

    # write it to csv file append mode
    fitness_df.to_csv(os.path.join(exp_log_dir, "fitness.csv"), mode='a', header=False, index=False)
    
    
    return -1 * reward
    

def main(_):

    workload = FLAGS.workload
    layer_id = FLAGS.layer_id

    maestro_helpers = helpers()
    dimension,_ = maestro_helpers.get_dimensions(workload, layer_id)
    
    # encoding format
    # seed, parallelization, l2-S, l2-R, l2-K, l2-C, l2-X, l2-Y, parallelization, l1-S
    # l1-R, l1-K, l1-C, l1-X, l1-Y, seed2, num_pes 
    ga = GA(
        func=Mastero_optimization_function, 
        n_dim=17, 
        size_pop=FLAGS.num_agents,
        max_iter=FLAGS.num_steps,
        prob_mut=FLAGS.prob_mutation,
        lb=[0, 0,  dimension['S']-1, dimension['R']-1, 1, 1, 1, 1, 0, dimension['S']-1, dimension['R']-1, 1, 1, 1, 1, 0,1],
        ub=[(2**32)-1, 3, dimension['S'], dimension['R'],
            dimension['K'], dimension['C'], dimension['X'],
            dimension['Y'], 3, dimension['S'], dimension['R'],
            dimension['K'], dimension['C'], dimension['X'], dimension['Y'], (2**32)-1, 1024],
        precision=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
    )
    
    best_x, best_y = ga.run()
    
    
    # get directory names
    _, exp_log_dir = generate_run_directories()
    
    # check if exp_log_dir exists
    if not os.path.exists(exp_log_dir):
        os.makedirs(exp_log_dir)
    
    print(ga.all_history_Y)
    # Convert each array to a list and concatenate the resulting lists
    arr = [a.squeeze().tolist() for a in ga.all_history_Y]

    print(arr)
    
    Y_history = pd.DataFrame(arr)
    Y_history.to_csv(os.path.join(exp_log_dir, "Y_history.csv"))

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.savefig(os.path.join(exp_log_dir, "Y_history.png"))
    
    # save the best_x and best_y to a csv file
    best_x = pd.DataFrame(best_x)
    best_x.to_csv(os.path.join(exp_log_dir, "best_x.csv"))

    best_y = pd.DataFrame(best_y)
    best_y.to_csv(os.path.join(exp_log_dir, "best_y.csv"))


if __name__ == '__main__':
   app.run(main)
