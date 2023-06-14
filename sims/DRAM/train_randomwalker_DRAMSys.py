import os
import sys

from absl import flags
from absl import app
from absl import logging

os.sys.path.insert(0, os.path.abspath('../../'))
from configs import arch_gym_configs
from arch_gym.envs.envHelpers import helpers
from arch_gym.envs import dramsys_wrapper
import envlogger
import numpy as np
import pandas as pd

flags.DEFINE_string('workload', 'stream.stl', 'Which DRAMSys workload to run?')
flags.DEFINE_integer('num_steps', 100, 'Number of training steps.')
flags.DEFINE_integer('num_episodes', 2, 'Number of training episodes.')
flags.DEFINE_string('traject_dir', 
                    'random_walker_trajectories', 
            'Directory to save the dataset.')
flags.DEFINE_bool('use_envlogger', False, 'Use envlogger to log the data.')  
flags.DEFINE_string('summary_dir', '.', 'Directory to save the summary.')
flags.DEFINE_string('reward_formulation', 'power', 'Which reward formulation to use?')
FLAGS = flags.FLAGS

def log_fitness_to_csv(filename, fitness_dict):
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
    env = dramsys_wrapper.make_dramsys_env()
    
    dram_helper = helpers()
    
    fitness_hist = {}

    # experiment name 
    exp_name = str(FLAGS.workload)+"_num_steps_" + str(FLAGS.num_steps) + "_num_episodes_" + str(FLAGS.num_episodes)

    # append logs to base path
    log_path = os.path.join(FLAGS.summary_dir, 'random_walker_logs', FLAGS.reward_formulation, exp_name)

    # get the current working directory and append the exp name
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)

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
            action = np.asarray(dram_helper.random_walk())

            # decode the actions
            action_dict = dram_helper.action_decoder_ga(action)

            _, reward, c, info = env.step(action_dict)
            fitness_hist['reward'] = reward
            fitness_hist['action'] = action_dict
            fitness_hist['obs'] = info
            log_fitness_to_csv(log_path, fitness_hist)
    
if __name__ == '__main__':
   app.run(main)

