#!/usr/bin/env python3
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import make_scorer
import os
import sys
os.sys.path.insert(0, os.path.abspath('../../'))
os.sys.path.insert(0, os.path.abspath('../../arch_gym'))
from arch_gym.envs.envHelpers import helpers
from  bo.AstraSimEstimator import AstraSimEstimator

import configparser
import numpy as np
import time
import pandas as pd
from absl import flags
from absl import app

flags.DEFINE_string('workload', 'resnet18', 'Workload trace file')
flags.DEFINE_integer('layer_id', 2, 'Layer id')
flags.DEFINE_integer('num_iter', 16, 'Number of training steps.')
flags.DEFINE_integer('random_state', 2, 'Random state.')
flags.DEFINE_string('traject_dir', 'bo_trajectories', 'Directory to store data.')
flags.DEFINE_string('exp_config_file', 'exp_config.ini', 'Experiment config file.')
flags.DEFINE_string('summary_dir', ".", 'Directory to store data.')
flags.DEFINE_string('reward_formulation', 'power', 'Reward formulation')
flags.DEFINE_bool('use_envlogger', True, 'Use EnvLogger to log environment data.')
FLAGS = flags.FLAGS


def scorer(estimator, X, y=None):
   """
   Custom scorer for AstraSimEstimator. Default is to maximize the fitness. 
   If the reward formulation is to maximize, then the fitness is returned as is.
   If the reward formulation is to minimize, then the fitness is returned as -1*fitness.
   """
   # definition of "good" score is minimum 
   # but default is higher score is better so * -1 for our purposes 
   return 1 * estimator.fit(X, y)


def find_best_params_test(X, parameters, n_iter, seed, exp_name, traject_dir, exp_log_dir):
   
   model = AstraSimEstimator(scheduling_policy = parameters['scheduling_policy'],
                            collective_optimization = parameters['collective_optimization'],
                            intra_dimension_scheduling = parameters['intra_dimension_scheduling'],
                            inter_dimension_scheduling = parameters['inter_dimension_scheduling'],
                            exp_name= exp_name, traject_dir= traject_dir)

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

    # helper
    h = helpers()

    # get the dimensions of the layers
    astrasim_helpers = helpers()
    dimension,_ = astrasim_helpers.get_dimensions(FLAGS.workload, FLAGS.layer_id)
    
    # define architectural parameters to search over
    parameters = {"scheduling_policy": Categorical(["FIFO", "LIFO"]),
                "collective_optimization":  Categorical(["localBWAware", "baseline"]),
                "intra_dimension_scheduling": Categorical(["FIFO", "SCF"]),
                "inter_dimension_scheduling": Categorical(["baseline", "themis"])
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