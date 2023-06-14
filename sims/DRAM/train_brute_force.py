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
flags.DEFINE_integer('num_steps', 10000, 'Number of training steps.')
flags.DEFINE_integer('num_episodes', 1, 'Number of training episodes.')
flags.DEFINE_string('traject_dir', 
                    'random_walker_trajectories', 
            'Directory to save the dataset.')
FLAGS = flags.FLAGS

def log_fitness_to_csv(filename, fitness_dict):
        df = pd.DataFrame([fitness_dict['reward']])
        csvfile = os.path.join(filename, "fitness.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

        df = pd.DataFrame([fitness_dict['action']])
        csvfile = os.path.join(filename, "actions.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

def main(_):
    env = dramsys_wrapper.make_dramsys_env()
    dram_helper = helpers()
    
    fitness_hist = {}

    # experiment name 
    exp_name = str(FLAGS.workload)+"_num_steps_" + str(FLAGS.num_steps) + "_num_episodes_" + str(FLAGS.num_episodes)

    # append logs to base path
    log_path = os.path.join(os.getcwd(), 'random_walker_logs', exp_name)

    # get the current working directory and append the exp name
    traject_dir = os.path.join(os.getcwd(), FLAGS.traject_dir, exp_name)

    # check if log_path exists else create it
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if not os.path.exists(traject_dir):
        os.makedirs(traject_dir)

    logging.info('Wrapping environment with EnvironmentLogger...')
    with envlogger.EnvLogger(env,
                 data_directory=traject_dir,
                 max_episodes_per_file=1000,
                 metadata={
                'agent_type': 'random',
                'env_type': type(env).__name__,
                'num_episodes': FLAGS.num_episodes,
                }) as env:
        
        logging.info('Done wrapping environment with EnvironmentLogger.')

        pagepolicy = [0, 1, 2, 3]
        scheduler = [0, 1, 2] #random.randint(0,2)
        schedulerbuffer = [0, 1, 2] #random.randint(0,2)
        reqest_buffer_size = [1, 2, 3, 5, 6, 7, 8] #random.randint(1,8)
        respqueue = [0, 1] #random.randint(0,1)
        refreshpolicy = [0, 1] #random.randint(0,1)
        refreshmaxpostponed = [1, 2, 3, 4, 5, 6, 7, 8] #random.randint(1,8)
        refreshmaxpulledin = [1, 2, 3, 4, 5, 6, 7, 8] #random.randint(1,8)
        powerdownpolicy = [0, 1, 2] #random.randint(0,2)
        arbiter = [0, 1, 2] #random.randint(0,2)
        maxactivetransactions = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]  #random.randint(1,128)

        total_combinations = len(pagepolicy) * len(scheduler) * len(schedulerbuffer) * len(reqest_buffer_size) * len(respqueue) * len(refreshpolicy) * len(refreshmaxpostponed) * len(refreshmaxpulledin) * len(powerdownpolicy) * len(arbiter) * len(maxactivetransactions)

        print("Total combinations: ", total_combinations)

        for each_pagepolicy in pagepolicy:
            for each_scheduler in scheduler:
                for each_schedulerbuffer in schedulerbuffer:
                    for each_reqest_buffer_size in reqest_buffer_size:
                        for each_respqueue in respqueue:
                            for each_refreshpolicy in refreshpolicy:
                                for each_refreshmaxpostponed in refreshmaxpostponed:
                                    for each_refreshmaxpulledin in refreshmaxpulledin:
                                        for each_powerdownpolicy in powerdownpolicy:
                                            for each_arbiter in arbiter:
                                                for each_maxactivetransactions in maxactivetransactions:
                                                    action = np.asarray([each_pagepolicy, 
                                                            each_scheduler,
                                                            each_schedulerbuffer,
                                                            each_reqest_buffer_size,
                                                            each_respqueue,
                                                            each_refreshpolicy,
                                                            each_refreshmaxpostponed,
                                                            each_refreshmaxpulledin,
                                                            each_powerdownpolicy,
                                                            each_arbiter, 
                                                            each_maxactivetransactions])
        
                                                    # decode the actions
                                                    action_dict = dram_helper.action_decoder_ga(action)

                                                    _, reward, _, _ = env.step(action_dict)
                                                    fitness_hist['reward'] = reward
                                                    fitness_hist['action'] = action_dict
        
                                                    # log fitness to csv
                                                    log_fitness_to_csv(log_path, fitness_hist)
 
if __name__ == '__main__':
   app.run(main)

