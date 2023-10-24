
from absl import flags
from absl import app

from absl import logging
import sys
import os.path
import pandas as pd
os.sys.path.insert(0, os.path.abspath('../../'))
from arch_gym.envs import customenv_wrapper
import envlogger
import numpy as np

from arch_gym.envs.custom_env import CustomEnv

flags.DEFINE_integer('num_steps', 50, 'Number of training steps.')
flags.DEFINE_integer('num_episodes', 1, 'Number of training episodes.')
flags.DEFINE_string('traject_dir', 
                    'random_walker_trajectories', 
            'Directory to save the dataset.')
flags.DEFINE_bool('use_envlogger', False, 'Use envlogger to log the data.')  
flags.DEFINE_string('summary_dir', '.', 'Directory to save the summary.')
flags.DEFINE_string('reward_formulation', 'power', 'Which reward formulation to use?')

FLAGS = flags.FLAGS

def log_fitness_to_csv(filename, fitness_dict):
    """Logs fitness history to csv file

    Args:
        filename (str): path to the csv file
        fitness_dict (dict): dictionary containing the fitness history
    """
    df = pd.DataFrame([fitness_dict['reward']])
    csvfile = os.path.join(filename, "fitness.csv")
    df.to_csv(csvfile, index=False, header=False, mode='a')

    # append to csv
    df = pd.DataFrame([fitness_dict])
    csvfile = os.path.join(filename, "trajectory.csv")
    df.to_csv(csvfile, index=False, header=False, mode='a')

def wrap_in_envlogger(env, envlogger_dir):
    """Wraps the environment in envlogger

    Args:
        env (gym.Env): gym environment
        envlogger_dir (str): path to the directory where the data will be logged
    """
    metadata = {
        'agent_type': 'RANDOM_WALKER',
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
    env = customenv_wrapper.make_custom_env(max_steps=FLAGS.num_steps)
    fitness_hist = {}

    # experiment name 
    exp_name = "num_steps_" + str(FLAGS.num_steps) + "_num_episodes_" + str(FLAGS.num_episodes)

    # append logs to base path
    log_path = os.path.join(FLAGS.summary_dir, 'random_walker_logs', exp_name)

    # get the current working directory and append the exp name
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, exp_name)

    # check if log_path exists else create it
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if FLAGS.use_envlogger:
        if not os.path.exists(traject_dir):
            os.makedirs(traject_dir)
    env = wrap_in_envlogger(env, traject_dir)
    
    for i in range(FLAGS.num_episodes):
        logging.info('Episode %r', i)
        for step in range(FLAGS.num_steps):
            # generate random actions
            action = env.action_space.sample()
            print(action)

            done, reward, info, obs = env.step(action)
            fitness_hist['reward'] = reward
            fitness_hist['action'] = action
            fitness_hist['obs'] = obs
            log_fitness_to_csv(log_path, fitness_hist)
    


if __name__ == '__main__':
    app.run(main)