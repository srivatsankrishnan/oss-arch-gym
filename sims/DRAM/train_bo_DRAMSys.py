#!/usr/bin/env python3
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import make_scorer
import os
os.sys.path.insert(0, os.path.abspath('../../'))
print(os.sys.path)
from  bo.DRAMSysEstimator import DRAMSysEstimator
import configparser
import numpy as np
import time
import pandas as pd
from absl import flags
from absl import app

flags.DEFINE_string('workload', 'stream.stl', 'Workload trace file')
flags.DEFINE_integer('num_iter', 16, 'Number of training steps.')
flags.DEFINE_integer('random_state', 2, 'Random state.')
flags.DEFINE_string('traject_dir', 'bo_trajectories', 'Directory to store data.')
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


def find_best_params_test(X,parameters, n_iter, seed, exp_name, traject_dir, exp_log_dir):
   
   model = DRAMSysEstimator(PagePolicy= parameters['PagePolicy'],
                           Scheduler= parameters['Scheduler'],
                           SchedulerBuffer= parameters['SchedulerBuffer'],
                           RequestBufferSize= parameters['RequestBufferSize'],
                           RespQueue= parameters['RespQueue'],
                           RefreshPolicy= parameters['RefreshPolicy'],
                           RefreshMaxPostponed= parameters['RefreshMaxPostponed'],
                           RefreshMaxPulledin= parameters['RefreshMaxPulledin'],
                           Arbiter= parameters['Arbiter'],
                           MaxActiveTransactions= parameters['MaxActiveTransactions'],
                           exp_name= exp_name,
                           traject_dir= traject_dir)

    
   # use config parser to update its parameters
   config = configparser.ConfigParser()
   config.read(FLAGS.exp_config_file)
   config.set("experiment_configuration", "exp_name", str(exp_name))
   config.set("experiment_configuration", "trajectory_dir", str(traject_dir))
   config.set("experiment_configuration", "log_dir", str(exp_log_dir))
   config.set("experiment_configuration", "reward_formulation", str(FLAGS.reward_formulation))
   config.set("experiment_configuration", "use_envlogger", str(FLAGS.use_envlogger))

   # write the updated config file
   with open(FLAGS.exp_config_file, 'w') as configfile:
      config.write(configfile)

    # Note need to use scipy=1.5.2 & scikit-learn=0.23.2 for this, see:
    # https://github.com/scikit-optimize/scikit-optimize/issues/978
   opt = BayesSearchCV(
        estimator=model,
        search_spaces = parameters,
        n_iter=FLAGS.num_iter,
        random_state=FLAGS.random_state,
        scoring=scorer,
        n_jobs=1,
        cv = 2,
   )
   
   # executes bayesian optimization
   opt.fit(X)
   print(opt.best_params_)
    
   return opt.best_params_


def main(_):

   # To do : Configure the workload trace here
   dummy_X = np.array([1,2,3,4,5,6])

   # define architectural parameters to search over
   parameters = {"PagePolicy": Categorical(['Open', 'OpenAdaptive', 'Closed', 'ClosedAdaptive']),
                "Scheduler":  Categorical(["Fifo", "FrFcfsGrp", "FrFcfs"]),
                "SchedulerBuffer": Categorical(["Bankwise", "ReadWrite","Shared"]),
                "RequestBufferSize": Integer(1,8),
                "RespQueue": Categorical(["Fifo","Reorder"]),
                "RefreshPolicy": Categorical(["NoRefresh","AllBank"]),
                "RefreshMaxPostponed": Integer(1,8),
                "RefreshMaxPulledin" : Integer(1,8),
                "Arbiter": Categorical(["Simple", "Fifo","Reorder"]),
                "MaxActiveTransactions": Integer(1,128)
                }

   # Construct the exp name from seed and num_iter
   exp_name = str(FLAGS.workload) + "_random_state_" + str(FLAGS.random_state) + "_num_iter_" + str(FLAGS.num_iter)
   
   # get the current working directory and append the exp name
   traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)

   # log directories for storing exp csvs
   exp_log_dir = os.path.join(FLAGS.summary_dir, "bo_logs", FLAGS.reward_formulation, exp_name)

   print("Trajectory directory: " + traject_dir)

   find_best_params_test(dummy_X, parameters,
                        FLAGS.num_iter,
                        FLAGS.random_state,
                        exp_name,
                        traject_dir,
                        exp_log_dir
                        )
  

if __name__ == '__main__':
   app.run(main)
