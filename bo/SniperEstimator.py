from sklearn.base import BaseEstimator, ClassifierMixin
import os
os.sys.path.insert(0, os.path.abspath('/../configs'))
from configs import arch_gym_configs
import json
from arch_gym.envs.envHelpers import helpers
from arch_gym.envs import sniperenv_wrapper
import sys
import numpy as np
import pandas as pd
import math
import time

import envlogger
class SniperEstimator(BaseEstimator):

    def __init__(self, core_dispatch_width=2, core_window_size=16, 
                core_outstanding_loads=2, core_outstanding_stores=2, 
                core_commit_width=2, core_rs_entries=2,
                l1_icache_size=4, l1_dcache_size=4, l2_cache_size=128, 
                l3_cache_size=4096, random_state= 1, num_iter=32):
        '''
        Initialize default values for all the parameters in Sniper
        '''

        #self.env = sniperenv_wrapper.make_sniper_env()
        self.helper = helpers()
        self.action_dict = {}
        self.action_dict["core_dispatch_width"] = core_dispatch_width
        self.action_dict["core_window_size"] = core_window_size
        self.action_dict["core_outstanding_loads"] = core_outstanding_loads
        self.action_dict["core_outstanding_stores"] = core_outstanding_stores
        self.action_dict["core_commit_width"] = core_commit_width
        self.action_dict["core_rs_entries"] = core_rs_entries
        self.action_dict["l1_icache_size"] = l1_icache_size
        self.action_dict["l1_dcache_size"] = l1_dcache_size
        self.action_dict["l2_cache_size"] = l2_cache_size
        self.action_dict["l3_cache_size"] = l3_cache_size

        self.hyperparam_rand_state = random_state
        self.hyperparam_num_iter = num_iter
        
        self.bo_steps =  0
        # check if log_path exists else create it
        if not os.path.exists(arch_gym_configs.sniper_envlogger_path):
            os.makedirs(arch_gym_configs.sniper_envlogger_path)
        self.fitness_hist = {}
        

    def fit (self, X, y=None):
        '''
        1) Call the Sniper simulator and return performance, power, and energy
        2) The parameter must be updated before the calling the Sniper simulator
        3)  X is the trace files (.e., Workload)
        '''
        #construct filename with rand_state and num_iter
        dir_name = arch_gym_configs.sniper_binary_path
        filename = "bo_" + str(self.hyperparam_rand_state) + "_" + str(self.hyperparam_num_iter)
        filename_full = os.path.join(dir_name, filename)
        env.reset()
        env = sniperenv_wrapper.make_sniper_env()
        
        reward = 0
        
        def step_fn(unused_timestep, unused_action, unused_env):
            return {'timestamp': time.time()}
        
        self.fitness_hist = {}
        with envlogger.EnvLogger(env,
                 data_directory=arch_gym_configs.sniper_envlogger_path,
                 max_episodes_per_file=1000,
                 metadata={
                    'agent_type': 'random',
                    'env_type': type(env).__name__
                    },
                    step_fn=step_fn) as env:
            # The Bayesian Optimization loop calls fit() number_iter times
            # For Envlogger, we need a loop and step function inside.
            # So ensure, it atleast run once per BO iteration
            while (self.bo_steps < 2):
                self.bo_steps += 1
                obs, reward, _, _ = env.step(self.action_dict)
                
                self.fitness_hist['fitness'] = reward
                self.fitness_hist['action'] = self.action_dict
        
                print("Logging fitness:", filename_full)
                self.log_fitness_to_csv(filename_full)
                #return reward
        return reward

    def predict(self, X, y):
        return NotImplementedError
    
    def score(self,X, y=None):
        return NotImplementedError

    def get_params(self, deep=False):

        return {
            'core_dispatch_width': self.action_dict["core_dispatch_width"],
            'core_window_size': self.action_dict["core_window_size"],
            'core_outstanding_loads': self.action_dict["core_outstanding_loads"],
            'core_outstanding_stores': self.action_dict["core_outstanding_stores"],
            'core_commit_width': self.action_dict["core_commit_width"],
            'core_rs_entries': self.action_dict["core_rs_entries"],
            'l1_icache_size': self.action_dict["l1_icache_size"],
            'l1_dcache_size': self.action_dict["l1_dcache_size"],
            'l2_cache_size': self.action_dict["l2_cache_size"],
            'l3_cache_size': self.action_dict["l3_cache_size"]
        }
    
    def set_params(self, **params):
        
        _params = self.transform_actions(params)
        self.action_dict["core_dispatch_width"] = _params["core_dispatch_width"]
        self.action_dict["core_window_size"] = _params["core_window_size"]
        self.action_dict["core_outstanding_loads"] = _params["core_outstanding_loads"]
        self.action_dict["core_outstanding_stores"] = _params["core_outstanding_stores"]
        self.action_dict["core_commit_width"] = _params["core_commit_width"]
        self.action_dict["core_rs_entries"] = _params["core_rs_entries"]
        self.action_dict["l1_icache_size"] = _params["l1_icache_size"]
        self.action_dict["l1_dcache_size"] = _params["l1_dcache_size"]
        self.action_dict["l2_cache_size"] = _params["l2_cache_size"]
        self.action_dict["l3_cache_size"] = _params["l3_cache_size"]

        print(self.action_dict)
        return self
    
    def transform_actions(self, params):
        '''
        Transform the parameters so its power of 2.
        '''
        _params = {}
        _params["core_dispatch_width"] = self.transformer(params["core_dispatch_width"])
        _params["core_window_size"] = self.transformer(params["core_window_size"])
        _params["core_outstanding_loads"] = self.transformer(params["core_outstanding_loads"])
        _params["core_outstanding_stores"] = self.transformer(params["core_outstanding_stores"])
        _params["core_commit_width"] = self.transformer(params["core_commit_width"])
        _params["core_rs_entries"] = self.transformer(params["core_rs_entries"])
        _params["l1_icache_size"] = self.transformer(params["l1_icache_size"])
        _params["l1_dcache_size"] = self.transformer(params["l1_dcache_size"])
        _params["l2_cache_size"] = self.transformer(params["l2_cache_size"])
        _params["l3_cache_size"] = self.transformer(params["l3_cache_size"])
        
        return _params

    def transformer(self,x):
        return int(math.pow(2, int(math.log2(x))))

    
    def log_fitness_to_csv(self, filename_full):
        #convert dictionary to dataframe
        #action log filename 
        action_log_filename = filename_full + "_action_log.csv"
        fitness_log_filename = filename_full + "_fitness_log.csv"
        
        df_action = pd.DataFrame([self.fitness_hist['action']])
        df_action.to_csv(action_log_filename, index=False, header=False, mode='a')
        
        df_fitness = pd.DataFrame([self.fitness_hist['fitness']])
        df_fitness.to_csv(fitness_log_filename, index=False, header=False, mode='a')