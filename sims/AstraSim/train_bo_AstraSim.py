#!/usr/bin/env python3
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import make_scorer
import os
import sys
os.sys.path.insert(0, os.path.abspath('../../'))
os.sys.path.insert(0, os.path.abspath('../../arch_gym'))
from arch_gym.envs.envHelpers import helpers
import bo.AstraSimEstimator as bo

import configparser
import numpy as np
import time
import pandas as pd
from absl import flags
from absl import app

flags.DEFINE_string('workload', 'resnet18', 'Workload trace file')
flags.DEFINE_integer('layer_id', 2, 'Layer id')
flags.DEFINE_integer('num_iter', 5, 'Number of training steps.')
flags.DEFINE_integer('random_state', 2, 'Random state.')
flags.DEFINE_string('traject_dir', 'bo_trajectories', 'Directory to store data.')
flags.DEFINE_string('exp_config_file', 'exp_config.ini', 'Experiment config file.')
flags.DEFINE_string('summary_dir', "./all_logs", 'Directory to store data.')
flags.DEFINE_string('reward_formulation', 'cycles', 'Reward formulation')
flags.DEFINE_bool('use_envlogger', True, 'Use EnvLogger to log environment data.')
flags.DEFINE_string('knobs', 'astrasim_220_example/knobs.py', "path to knobs spec file")
flags.DEFINE_string('network', 'astrasim_220_example/network_input.yml', "path to network input file")
flags.DEFINE_string('system', 'astrasim_220_example/system_input.json', "path to system input file")
flags.DEFINE_string('workload_file', 'astrasim_220_example/workload_cfg.json', "path to workload input file")
flags.DEFINE_bool('congestion_aware', True, "astra-sim congestion aware or not")
# FLAGS.workload_file = astrasim_220_example/workload_cfg.json if GENERATE_WORKLOAD = True
# FLAGS.workload_file = astrasim_220_example/workload-et/generated if GENERATE_WORKLOAD = False

FLAGS = flags.FLAGS

# define AstraSim version
VERSION = 2

def scorer(estimator, X, y=None):
   """
   Custom scorer for AstraSimEstimator. Default is to maximize the fitness. 
   If the reward formulation is to maximize, then the fitness is returned as is.
   If the reward formulation is to minimize, then the fitness is returned as -1*fitness.
   """
   # definition of "good" score is minimum 
   # but default is higher score is better so * -1 for our purposes 
   return 1 * estimator.fit(X, y)


def find_best_params_test(X, parameters, n_iter, seed, exp_name, traject_dir, exp_log_dir, dimension_count):
   
   # model = bo.AstraSimEstimator(exp_name=exp_name, traject_dir=traject_dir, **parameters)

   model = bo.AstraSimEstimator(exp_name, traject_dir, **parameters)

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

   settings_file_path = os.path.realpath(__file__)
   settings_dir_path = os.path.dirname(settings_file_path)
   proj_root_path = os.path.abspath(settings_dir_path)
   astrasim_archgym = os.path.join(proj_root_path, "astrasim-archgym")

   archgen_v1_knobs = os.path.join(astrasim_archgym, "dse/archgen_v1_knobs")
   networks_folder = os.path.join(archgen_v1_knobs, "templates/network")
   if VERSION == 1:
      network_file = os.path.join(networks_folder, "4d_ring_fc_ring_switch.json")
   else:
      network_file = os.path.join(proj_root_path, FLAGS.network)

   knobs_spec = os.path.join(proj_root_path, FLAGS.knobs)

   network_parsed = {}
   h.parse_network_astrasim(network_file, network_parsed, VERSION)
   system_knob, network_knob, workload_knob = h.parse_knobs_astrasim(knobs_spec)
   dicts = [system_knob, network_knob, workload_knob]

   if VERSION == 1:
      dimension_count = network_parsed['network']["dimensions-count"]
   else:
      dimension_count = len(network_parsed['network']["topology"])

   if "dimensions-count" in network_knob:
      dimensions = sorted(list(network_knob["dimensions-count"][0]))
   else:
      dimensions = [dimension_count]

   # parameters = {"scheduling_policy": Categorical(["FIFO", "LIFO"])}
   # print("params_TEST: ", parameters)

   for d in dimensions:
      parameters = {}
      for knobs_dict in dicts:
         for knob in knobs_dict:
            if knob == "dimensions-count":
               continue
            # converts hyphens to underscores
            knob_converted = h.convert_knob_bo_astrasim(knob)
            if isinstance(knobs_dict[knob][0], set):
               if knobs_dict[knob][1] == "FALSE":
                  for i in range(1, d + 1):
                     # 'topology': ({"Ring", "Switch", "FullyConnected"}, 'FALSE')
                     knob_dimension = knob_converted + str(i)
                     list_sorted = sorted(list(knobs_dict[knob][0]))
                     parameters[knob_dimension] = Categorical(list_sorted)
               else:
                  list_sorted = sorted(list(knobs_dict[knob][0]))
                  parameters[knob_converted] = Categorical(list_sorted)
            else:
               # 'num_npus': ((64, 64, 1), 'N/A'),
               if knobs_dict[knob][1] == "FALSE":
                  for i in range(1, d + 1):
                     knob_dimension = knob_converted + str(i)
                     parameters[knob_dimension] = Integer(knobs_dict[knob][0][0], knobs_dict[knob][0][1])
               else:
                  parameters[knob_converted] = Integer(knobs_dict[knob][0][0], knobs_dict[knob][0][1])

      # Construct the exp name from seed and num_iter
      exp_name = str(FLAGS.workload) + "_random_state_" + str(FLAGS.random_state) + "_num_iter_" + str(FLAGS.num_iter)

      # get the current working directory and append the exp name
      traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)

      # log directories for storing exp csvs
      exp_log_dir = os.path.join(FLAGS.summary_dir, "bo_logs", FLAGS.reward_formulation, exp_name)

      print("Trajectory directory: " + traject_dir)

      flag_dict = {"knobs": str(FLAGS.knobs), "network": str(FLAGS.network), "system": str(FLAGS.system), "workload": str(FLAGS.workload_file), 
                  "reward_formulation": str(FLAGS.reward_formulation), "congestion_aware": FLAGS.congestion_aware, 'summary_dir': FLAGS.summary_dir, 'dimension': d}

      # write flags to json file for AstraSimEstimator to read
      with open(os.path.join(proj_root_path, "bo_vars.json"), 'w') as file:
         file.write('{\n')
         for key, value in flag_dict.items():
            file.write(f'"{key}": "{value}",\n')
         file.seek(file.tell() - 2, os.SEEK_SET)
         file.write('\n')
         file.write('}')

      find_best_params_test(dummy_X, parameters,
                              FLAGS.num_iter,
                              FLAGS.random_state,
                              exp_name,
                              traject_dir,
                              exp_log_dir,
                              d
                              )

if __name__ == '__main__':
    app.run(main)