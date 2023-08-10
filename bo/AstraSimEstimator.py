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


class AstraSimEstimator(BaseEstimator):

    def __init__(self, scheduling_policy="FIFO", collective_optimization="baseline", 
                 intra_dimension_scheduling="FIFO", inter_dimension_scheduling="baseline",
                 exp_name="test", traject_dir="traj"):
        
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

        # TODO: V1 SPEC:
        archgen_v1_knobs = os.path.join(astrasim_archgym, "dse/archgen_v1_knobs")
        knobs_spec = os.path.join(archgen_v1_knobs, "archgen_v1_knobs_spec.py")
        networks_folder = os.path.join(archgen_v1_knobs, "templates/network")
        systems_folder = os.path.join(astrasim_archgym, "themis/inputs/system")
        workloads_folder = os.path.join(astrasim_archgym, "themis/inputs/workload")


        self.network_file = "4d_ring_fc_ring_switch.json"
        self.system_file = os.path.join(systems_folder, "4d_ring_fc_ring_switch_baseline.txt")
        self.workload_file = "all_reduce/allreduce_0.20.txt"
        
        # self.action_dict['network'] = {"path": self.network_file}
        # self.action_dict['workload'] = {"path": self.workload_file}
        
        # self.parse_system(self.system_file, self.action_dict)
    
        # self.action_dict["system"]["scheduling-policy"] = scheduling_policy
        self.action_dict["scheduling_policy"] = scheduling_policy
        self.action_dict["collective_optimization"] = collective_optimization
        self.action_dict["intra_dimension_scheduling"] = intra_dimension_scheduling
        self.action_dict["inter_dimension_scheduling"] = inter_dimension_scheduling
    
        self.exp_name = exp_name
        self.traject_dir = traject_dir
        self.fitness_hist = []
        self.exp_log_dir = os.path.join(os.getcwd(), "bo_logs")
        self.reward_formulation = 'power'
        
        print("[Experiment]: ", self.exp_name)
        print("[Trajectory Log path]: ", self.traject_dir)

        
        self.bo_steps=0
    

    def parse_system(self, system_file, action_dict):
        # parse system_file (above is the content) into dict
        action_dict['system'] = {}
        with open(system_file, 'r') as file:
            lines = file.readlines()

            for line in lines:
                key, value = line.strip().split(': ')
                action_dict['system'][key] = value

        
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

        # read the all the parameters from exp_config.ini
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
        
        # convert the action dict to a list with the same order 
        # action_list = []

        actual_action = {}
        actual_action['network'] = {"path": self.network_file}
        actual_action['workload'] = {"path": self.workload_file}
        self.parse_system(self.system_file, actual_action)

        actual_action["system"]["scheduling-policy"] = self.action_dict["scheduling_policy"]
        actual_action["system"]["collective-optimization"] = self.action_dict["collective_optimization"]
        actual_action["system"]["intra-dimension-scheduling"] = self.action_dict["intra_dimension_scheduling"]
        actual_action["system"]["inter-dimension-scheduling"] = self.action_dict["inter_dimension_scheduling"]

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
        return {
            "scheduling_policy": self.action_dict["scheduling_policy"],
            "collective_optimization": self.action_dict["collective_optimization"],
            "intra_dimension_scheduling": self.action_dict["intra_dimension_scheduling"],
            "inter_dimension_scheduling": self.action_dict["inter_dimension_scheduling"]
        }
      
    def set_params(self, **params):
        """
        scheduling-policy: LIFO
        endpoint-delay: 1
        active-chunks-per-dimension: 1
        preferred-dataset-splits: 64
        boost-mode: 1
        all-reduce-implementation: direct_ring_halvingDoubling
        all-gather-implementation: direct_ring_halvingDoubling
        reduce-scatter-implementation: direct_ring_halvingDoubling
        all-to-all-implementation: direct_direct_direct
        collective-optimization: localBWAware
        intra-dimension-scheduling: FIFO
        inter-dimension-scheduling: baseline
        """
        _params = params
        self.action_dict["scheduling_policy"] = _params["scheduling_policy"]
        self.action_dict["collective_optimization"] = _params["collective_optimization"]
        self.action_dict["intra_dimension_scheduling"] = _params["intra_dimension_scheduling"]
        self.action_dict["inter_dimension_scheduling"] = _params["inter_dimension_scheduling"]
        
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
