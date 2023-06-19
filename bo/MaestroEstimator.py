from sklearn.base import BaseEstimator, ClassifierMixin
import os
os.sys.path.insert(0, os.path.abspath('/../arch_gym/envs/'))
os.sys.path.insert(0, os.path.abspath('/../'))
from configs import arch_gym_configs
import json
from arch_gym.envs.envHelpers import helpers
from arch_gym.envs.MasteroEnv import MasteroEnv
from arch_gym.envs.maestero_wrapper import make_maestro_env
import configparser
import envlogger
import sys
import numpy as np
import pandas as pd
import time

from absl import logging
from absl import flags


class MaestroEstimator(BaseEstimator):

    def __init__(self, seed_l2= 123,ckxy_l2 = 2, s_l2 = 2, r_l2 = 2,
                            k_l2 = 1, c_l2 = 1,
                            x_l2 = 2, y_l2 = 2,
                            ckxy_l1 = 1, s_l1 = 2,
                            r_l1 = 2, k_l1 = 1,
                            c_l1 = 1, x_l1 = 2,
                            y_l1 = 1, seed_l1 = 1,
                            num_pe = 4, exp_name="test", traject_dir="traj"):
        
        ''' All the default values of the Maestro should be initialized here. 
            Take all the parameters here and write it to the config files
        '''
        # To do: Implement some default parameters 
        self.env = MasteroEnv()
        self.helper = helpers()
        self.action_dict = {}
        self.action_dict['seed_l2'] = seed_l2
        self.action_dict['ckxy_l2'] = ckxy_l2
        self.action_dict['s_l2'] = s_l2
        self.action_dict['r_l2'] = r_l2
        self.action_dict['k_l2'] = k_l2
        self.action_dict['c_l2'] = c_l2
        self.action_dict['x_l2'] = x_l2
        self.action_dict['y_l2'] = y_l2
        self.action_dict['ckxy_l1'] = ckxy_l1
        self.action_dict['s_l1'] = s_l1
        self.action_dict['r_l1'] = r_l1
        self.action_dict['k_l1'] = k_l1
        self.action_dict['c_l1'] = c_l1
        self.action_dict['x_l1'] = x_l1
        self.action_dict['y_l1'] = y_l1
        self.action_dict['seed_l1'] = seed_l1
        self.action_dict['num_pe'] = num_pe
        
        self.exp_name = exp_name
        self.traject_dir = traject_dir
        self.fitness_hist = []
        self.exp_log_dir = os.path.join(os.getcwd(),"logs")
        self.reward_formulation = 'power'
        
        print("[Experiment]: ", self.exp_name)
        print("[Trajectory Log path]: ", self.traject_dir)

        
        self.bo_steps=0
        
    def wrap_in_envlogger(self, env, envlogger_dir, use_envlogger):
        metadata = {
            'agent_type': 'RandomWalker',
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
        1) Call the DRAMSys simulator and return performance, power, and energy
        2) The parameter must be updated before the calling the DRAMSys simulator
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

        env_wrapper = make_maestro_env(reward_formulation = reward_formulation,
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
        action_list = [
            self.action_dict['seed_l2'],
            self.action_dict['ckxy_l2'],
            self.action_dict['s_l2'],
            self.action_dict['r_l2'],
            self.action_dict['k_l2'],
            self.action_dict['c_l2'],
            self.action_dict['x_l2'],
            self.action_dict['y_l2'],
            self.action_dict['ckxy_l1'],
            self.action_dict['s_l1'],
            self.action_dict['r_l1'],
            self.action_dict['k_l1'],
            self.action_dict['c_l1'],
            self.action_dict['x_l1'],
            self.action_dict['y_l1'],
            self.action_dict['seed_l1'],
            self.action_dict['num_pe']
        ]
        _, reward, _, info = env.step(action_list)

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
        return {'seed_l2': self.action_dict['seed_l2'],
                'ckxy_l2': self.action_dict['ckxy_l2'],
                's_l2': self.action_dict['s_l2'],
                'r_l2': self.action_dict['r_l2'],
                'k_l2': self.action_dict['k_l2'],
                'c_l2': self.action_dict['c_l2'],
                'x_l2': self.action_dict['x_l2'],
                'y_l2': self.action_dict['y_l2'],
                'ckxy_l1': self.action_dict['ckxy_l1'],
                's_l1': self.action_dict['s_l1'],
                'r_l1': self.action_dict['r_l1'],
                'k_l1': self.action_dict['k_l1'],
                'c_l1': self.action_dict['c_l1'],
                'x_l1': self.action_dict['x_l1'],
                'y_l1': self.action_dict['y_l1'],
                'seed_l1': self.action_dict['seed_l1'],
                'num_pe': self.action_dict['num_pe']
               }
      
    def set_params(self, **params):
        _params = params
        self.action_dict['seed_l2'] = _params['seed_l2']
        self.action_dict['ckxy_l2'] = _params['ckxy_l2']
        self.action_dict['s_l2'] = _params['s_l2']
        self.action_dict['r_l2'] = _params['r_l2']
        self.action_dict['k_l2'] = _params['k_l2']
        self.action_dict['c_l2'] = _params['c_l2']
        self.action_dict['x_l2'] = _params['x_l2']
        self.action_dict['y_l2'] = _params['y_l2']
        self.action_dict['ckxy_l1'] = _params['ckxy_l1']
        self.action_dict['s_l1'] = _params['s_l1']
        self.action_dict['r_l1'] = _params['r_l1']
        self.action_dict['k_l1'] = _params['k_l1']
        self.action_dict['c_l1'] = _params['c_l1']
        self.action_dict['x_l1'] = _params['x_l1']
        self.action_dict['y_l1'] = _params['y_l1']
        self.action_dict['seed_l1'] = _params['seed_l1']
        self.action_dict['num_pe'] = _params['num_pe']
        
        return self
    def calculate_reward(self, energy, latency):
        reward = np.sum([np.square(configs.target_energy - energy),np.square(configs.target_latency - latency)])
        reward = np.sqrt(reward)
        return reward

    
    def log_fitness_to_csv(self, filename):
        df = pd.DataFrame([self.fitness_hist['reward']])
        csvfile = os.path.join(filename, "fitness.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

        df = pd.DataFrame([self.fitness_hist])
        csvfile = os.path.join(filename, "actions.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')
               



      