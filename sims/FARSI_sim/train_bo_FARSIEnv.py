#!/usr/bin/env python3
import os
import sys

# from arch_gym.envs.envHelpers import helpers

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import make_scorer
import os
os.sys.path.insert(0, os.path.abspath('../../'))
print(os.sys.path)
from  bo.FARSIEnvEstimator import FARSIEnvEstimator
import configparser
import numpy as np
import time
import pandas as pd
from absl import flags
from absl import app

# flags.DEFINE_string('workload', 'stream.stl', 'Workload trace file')
flags.DEFINE_string('workload', 'edge_detection', 'Which Workload to run')
flags.DEFINE_integer('num_iter', 10, 'Number of training steps.')
flags.DEFINE_integer('random_state', 2, 'Random state.')
flags.DEFINE_string('traject_dir', 'bo_trajectories', 'Directory to store data.')
flags.DEFINE_string('exp_log_dir', 'bo_logs', 'Directory to store logs.')
flags.DEFINE_string('exp_config_file', 'exp_config.ini', 'Experiment config file.')
flags.DEFINE_string('summary_dir', ".", 'Directory to store data.')
flags.DEFINE_string('reward_formulation', 'power', 'Reward formulation')
flags.DEFINE_bool('use_envlogger', False, 'Use EnvLogger to log environment data.')
FLAGS = flags.FLAGS

def scorer(estimator, X, y=None):
   """
   Custom scorer for DRAMSysEstimator. Default is to maximize the fitness. 
   If the reward formulation is to maximize, then the fitness is returned as is.
   If the reward formulation is to minimize, then the fitness is returned as -1*fitness.
   """
   # definition of "good" score is minimum 
   # but default is higher score is better so * -1 for our purposes 
   return 1 * estimator.fit(X, y)


def find_best_params_test(X,parameters, n_iter, seed,exp_name, traject_dir, exp_log_dir):
   workload = FLAGS.workload
   reward_formulation = FLAGS.reward_formulation
   use_envlogger = FLAGS.use_envlogger
   # exp_log_dir = FLAGS.exp_log_dir

   print("workload", workload)
   print("reward_formulation", reward_formulation)
   print("use_envlogger", use_envlogger)
   print("exp_log_dir", exp_log_dir)
  
   # use config parser to update its parameters
   config = configparser.ConfigParser()
   config.read(FLAGS.exp_config_file)
   config.set("experiment_configuration", "exp_name", str(exp_name))
   config.set("experiment_configuration", "trajectory_dir", str(traject_dir))
   config.set("experiment_configuration", "log_dir", str(exp_log_dir))
   config.set("experiment_configuration", "reward_formulation", str(FLAGS.reward_formulation))
   config.set("experiment_configuration", "workload", str(FLAGS.workload))
   config.set("experiment_configuration", "use_envlogger", str(FLAGS.use_envlogger))

   # write the updated config file
   with open(FLAGS.exp_config_file, 'w') as configfile:
      config.write(configfile)

   # wait for the config file to be written
   time.sleep(0.1)

   model = FARSIEnvEstimator(traj_dir=traject_dir, exp_name=exp_name, log=exp_log_dir, reward_formulation=reward_formulation, use_envlogger=use_envlogger, workload=workload,
   pe_allocation_0 = 0, pe_allocation_1 = 0, pe_allocation_2 = 0, 
   mem_allocation_0 = 0, mem_allocation_1 = 0, mem_allocation_2 = 0,
   bus_allocation_0 = 0, bus_allocation_1 = 0, bus_allocation_2 = 0,
   pe_to_bus_connection_0 = 0, pe_to_bus_connection_1 = 0,pe_to_bus_connection_2 = 0,
   bus_to_bus_connection_0 = -1, bus_to_bus_connection_1 = -1, bus_to_bus_connection_2 = -1,
   bus_to_mem_connection_0 = -1, bus_to_mem_connection_1 = -1, bus_to_mem_connection_2 = -1,

   task_to_pe_mapping_0  = 0, task_to_pe_mapping_1  = 0, task_to_pe_mapping_2  = 0, task_to_pe_mapping_3  = 0, task_to_pe_mapping_4  = 0, task_to_pe_mapping_5  = 0, task_to_pe_mapping_6  = 0, task_to_pe_mapping_7  = 0,

   task_to_mem_mapping_0  = 0, task_to_mem_mapping_1  = 0, task_to_mem_mapping_2  = 0, task_to_mem_mapping_3  = 0,  task_to_mem_mapping_4  = 0, task_to_mem_mapping_5  = 0, task_to_mem_mapping_6  = 0, task_to_mem_mapping_7  = 0)


    # Note need to use scipy=1.5.2 & scikit-learn=0.23.2 for this, see:
    # https://github.com/scikit-optimize/scikit-optimize/issues/978
   opt = BayesSearchCV(
        estimator=model,
        search_spaces = parameters, # devashree: add FARSI search space here
        n_iter=n_iter,
        random_state=seed,
        scoring=scorer,
        n_jobs=1,
        cv = 2,
   )
   print("opt type"); print(type(opt))

   # executes bayesian optimization
   opt.fit(X)
   print(opt.best_params_)
    
   return opt.best_params_


def main(_):

   # To do : Configure the workload trace here
   dummy_X = np.array([1,2,3,4,5,6])

   # define architectural parameters to search over
   # define architectural parameters to search over
   parameters = {"pe_allocation_0": Integer(0,1),
                  "pe_allocation_1": Integer(0,1),
                  "pe_allocation_2": Integer(0,1),
                  "mem_allocation_0": Integer(0,3),
                  "mem_allocation_1": Integer(0,3),
                  "mem_allocation_2": Integer(0,3),
                  "bus_allocation_0": Integer(0,3),
                  "bus_allocation_1": Integer(0,3),
                  "bus_allocation_2": Integer(0,3),
                  "pe_to_bus_connection_0": Integer(0,2),
                  "pe_to_bus_connection_1": Integer(0,2),
                  "pe_to_bus_connection_2": Integer(0,2),
                  "bus_to_bus_connection_0": Integer(-1,2),
                  "bus_to_bus_connection_1": Integer(-1,2),
                  "bus_to_bus_connection_2": Integer(-1,2),
                  "bus_to_mem_connection_0": Integer(-1,2),
                  "bus_to_mem_connection_1": Integer(-1,2),
                  "bus_to_mem_connection_2": Integer(-1,2),
                  "task_to_pe_mapping_0": Integer(0,2),
                  "task_to_pe_mapping_1": Integer(0,2),
                  "task_to_pe_mapping_2": Integer(0,2),
                  "task_to_pe_mapping_3": Integer(0,2),
                  "task_to_pe_mapping_4": Integer(0,2),
                  "task_to_pe_mapping_5": Integer(0,2),
                  "task_to_pe_mapping_6": Integer(0,2),
                  "task_to_pe_mapping_7": Integer(0,2),
                  "task_to_mem_mapping_0": Integer(0,2),
                  "task_to_mem_mapping_1": Integer(0,2),
                  "task_to_mem_mapping_2": Integer(0,2),
                  "task_to_mem_mapping_3": Integer(0,2),
                  "task_to_mem_mapping_4": Integer(0,2),
                  "task_to_mem_mapping_5": Integer(0,2),
                  "task_to_mem_mapping_6": Integer(0,2),
                  "task_to_mem_mapping_7": Integer(0,2),
               }

   reward_formulation_list = FLAGS.reward_formulation.split(" ")

   # join the list of reward formulations with a _
   reward_formulation = "_".join(reward_formulation_list)

   # Construct the exp name from seed and num_iter
   exp_name = str(FLAGS.workload) + "_random_state_" + str(FLAGS.random_state) + "_num_iter_" + str(FLAGS.num_iter)
   
   # get the current working directory and append the exp name
   traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, reward_formulation, exp_name)

   # log directories for storing exp csvs
   exp_log_dir = os.path.join(FLAGS.summary_dir, FLAGS.exp_log_dir, reward_formulation, exp_name)

   if not os.path.exists(exp_log_dir):
       os.makedirs(exp_log_dir)
   
   print("Trajectory directory: " + traject_dir)
   print("Log directory: " + exp_log_dir)

   find_best_params_test(dummy_X, parameters,
                        FLAGS.num_iter,
                        FLAGS.random_state,
                        exp_name,
                        traject_dir,
                        exp_log_dir
                        )
  

if __name__ == '__main__':
   app.run(main)
