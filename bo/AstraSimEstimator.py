from sklearn.base import BaseEstimator, ClassifierMixin
import os
os.sys.path.insert(0, os.path.abspath('/../arch_gym/envs/'))
os.sys.path.insert(0, os.path.abspath('/../'))
from configs import arch_gym_configs
import json
from arch_gym.envs.envHelpers import helpers
from arch_gym.envs.AstraSimEnv import AstraSimEnv
from arch_gym.envs.AstraSimWrapper import make_astraSim_env
import configparser
import envlogger
import sys
import numpy as np
import pandas as pd
import time

from absl import logging
from absl import flags

# define AstraSim version
VERSION = 1

class AstraSimEstimator(BaseEstimator):

    def __init__(self, exp_name="test", traject_dir="traj", **parameters):
        
        ''' All the default values of AstraSim should be initialized here. 
            Take all the parameters here and write it to the config files
        '''
        # To do: Implement some default parameters 
        self.env = AstraSimEnv()
        self.helper = helpers()
        self.action_dict = {}
    
        settings_file_path = os.path.realpath(__file__)
        settings_dir_path = os.path.dirname(settings_file_path)
        proj_root_path = os.path.join(settings_dir_path, '..')
        astrasim_archgym = os.path.join(proj_root_path, "sims/AstraSim/astrasim-archgym")

        archgen_v1_knobs = os.path.join(astrasim_archgym, "dse/archgen_v1_knobs")
        knobs_spec = os.path.join(archgen_v1_knobs, "archgen_v1_knobs_spec.py")
        networks_folder = os.path.join(archgen_v1_knobs, "templates/network")
        systems_folder = os.path.join(astrasim_archgym, "themis/inputs/system")
        workloads_folder = os.path.join(astrasim_archgym, "themis/inputs/workload")

        self.network_file = os.path.join(networks_folder, "4d_ring_fc_ring_switch.json")
        self.system_file = os.path.join(systems_folder, "4d_ring_fc_ring_switch_baseline.txt")
        self.workload_file = os.path.join(workloads_folder, "all_reduce/allreduce_0.20.txt")

        self.system_knob, self.network_knob, self.workload_knob = self.helper.parse_knobs_astrasim(knobs_spec)
        self.dicts = [(self.system_knob, 'system'), (self.network_knob, 'network'), (self.workload_knob, 'workload')]

        for param_key, _ in parameters.items():
            knob_reverted = self.helper.revert_knob_bo_astrasim(param_key)
            for dict_type, _ in self.dicts:
                if knob_reverted in dict_type.keys():
                    self.action_dict[param_key] = 0
    
        self.exp_name = exp_name
        self.traject_dir = traject_dir
        self.fitness_hist = []
        self.exp_log_dir = os.path.join(os.getcwd(), "bo_logs")
        self.reward_formulation = 'power'
        
        print("[Experiment]: ", self.exp_name)
        print("[Trajectory Log path]: ", self.traject_dir)

        
        self.bo_steps=0

        
    def wrap_in_envlogger(self, env, envlogger_dir, use_envlogger):
        metadata = {
            'agent_type': 'Bayesian Optimization',
            'env_type': type(env).__name__,
        }
        if use_envlogger == 'True':
            logging.info('Wrapping environment with EnvironmentLogger...')
            env = envlogger.EnvLogger(env,
                                    data_directory=envlogger_dir,
                                    max_episodes_per_file=1000,
                                    metadata=metadata)
            logging.info('Done wrapping environment with EnvironmentLogger.')
            return env
        else:
            print("Not using envlogger")
            return env
        

    def fit (self, X, y=None):
        '''
        1) Call the AstraSim simulator and return performance, power, and energy
        2) The parameter must be updated before the calling the AstraSim simulator
        3)  X is the trace files (.e., Workload)
        '''
        self.bo_steps += 1

        def step_fn(unused_timestep, unused_action, unused_env):
            return {'timestamp': time.time()}
            
        reward = 0
        self.fitness_hist = {}

        # read from the config file
        config = configparser.ConfigParser()
        config.read("exp_config.ini")

        # read all the parameters from exp_config.ini
        traj_dir = config.get("experiment_configuration", "trajectory_dir")
        exp_name = config.get("experiment_configuration", "exp_name")
        log_dir = config.get("experiment_configuration", "log_dir")
        reward_formulation = config.get("experiment_configuration", "reward_formulation")
        use_envlogger = config.get("experiment_configuration", "use_envlogger")

        env_wrapper = make_astraSim_env(reward_formulation = reward_formulation,
            rl_form = 'bo')
        
        # check if trajectory directory exists
        if use_envlogger == 'True':
            if not os.path.exists(traj_dir):
                os.makedirs(traj_dir)
        # check if log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        env = self.wrap_in_envlogger(env_wrapper, self.exp_log_dir, use_envlogger)
        env.reset()
        print("Action dict: ", self.action_dict)

        actual_action = {}
        actual_action['workload'] = {"path": self.workload_file}
        actual_action['network'] = self.helper.parse_network_astrasim(self.network_file, actual_action, VERSION)
        actual_action['system'] = self.helper.parse_system_astrasim(self.system_file, actual_action, VERSION)

        # ASSUMES dimension is specified by input files (otherwise randomized)
        if "dimensions-count" in self.network_knob.keys():
            actual_action['network']["dimensions-count"] = random.choice(network_knob["dimensions-count"])
            dimension = actual_action['network']["dimensions-count"]
            self.network_knob.remove("dimensions-count")
        else:
            dimension = actual_action['network']["dimensions-count"]

        for dict_type, dict_name in self.dicts:
            for knob in dict_type.keys():
                knob_converted = self.helper.convert_knob_bo_astrasim(knob)
                if dict_type[knob][1] == "FALSE":
                    actual_action[dict_name][knob] = [self.action_dict[knob_converted + str(i)] for i in range(1, dimension+1)]
                elif dict_type[knob][1] == "TRUE":
                    actual_action[dict_name][knob] = [self.action_dict[knob_converted] for _ in range(dimension)]
                else:
                    actual_action[dict_name][knob] = self.action_dict[knob_converted]

        _, reward, _, info = env.step(actual_action)

        self.fitness_hist['reward'] = reward
        self.fitness_hist['action'] = self.action_dict
        self.fitness_hist['obs'] = info

        fitness_filename = os.path.join(self.exp_name)

        # logging twice due to the cv. So we will track the bo_steps and log only once
        if self.bo_steps == 1:
            self.log_fitness_to_csv(log_dir)

        # clear the self.fitness_hist
        self.fitness_hist = []
        
        return reward

    def predict(self, X, y):
        return NotImplementedError

    def score(self,X, y=None):
        return NotImplementedError

    def get_params(self, deep=False):
        param_dict = {}
        for knob in self.action_dict.keys():
            param_dict[knob] = self.action_dict[knob]

        return param_dict
      
    def set_params(self, **params):
        _params = params
        for knob in _params.keys():
            self.action_dict[knob] = _params[knob]

        return self


    def calculate_reward(self, energy, latency):
        sum = ((float(latency) - 1) ** 2)
        return 1 / (sum ** 0.5)

    
    def log_fitness_to_csv(self, filename):
        df = pd.DataFrame([self.fitness_hist['reward']])
        csvfile = os.path.join(filename, "fitness.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

        df = pd.DataFrame([self.fitness_hist])
        csvfile = os.path.join(filename, "actions.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')
