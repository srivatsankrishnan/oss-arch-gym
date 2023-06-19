from sklearn.base import BaseEstimator, ClassifierMixin
import os
os.sys.path.insert(0, os.path.abspath('/../'))
from configs import arch_gym_configs
import json
from arch_gym.envs.envHelpers import helpers
from arch_gym.envs import FARSI_sim_wrapper
import configparser
import envlogger
import sys
import numpy as np
import pandas as pd
import time

from absl import logging
from absl import flags
from collections import OrderedDict

class FARSIEnvEstimator(BaseEstimator):

    def __init__(self, 
                traj_dir=None, exp_name=None, log=None, reward_formulation=None, use_envlogger=None, workload=None,
        pe_allocation_0 = 0, pe_allocation_1 = 0, pe_allocation_2 = 0, 
        mem_allocation_0 = 0, mem_allocation_1 = 0, mem_allocation_2 = 0,
        bus_allocation_0 = 0, bus_allocation_1 = 0, bus_allocation_2 = 0,
        pe_to_bus_connection_0 = 0, pe_to_bus_connection_1 = 0,pe_to_bus_connection_2 = 0,
        bus_to_bus_connection_0 = -1, bus_to_bus_connection_1 = -1, bus_to_bus_connection_2 = -1,
        bus_to_mem_connection_0 = -1, bus_to_mem_connection_1 = -1, bus_to_mem_connection_2 = -1,

        task_to_pe_mapping_0  = 0, task_to_pe_mapping_1  = 0, task_to_pe_mapping_2  = 0, task_to_pe_mapping_3  = 0, task_to_pe_mapping_4  = 0, task_to_pe_mapping_5  = 0, task_to_pe_mapping_6  = 0, task_to_pe_mapping_7  = 0,

        task_to_mem_mapping_0  = 0, task_to_mem_mapping_1  = 0, task_to_mem_mapping_2  = 0, task_to_mem_mapping_3  = 0,  task_to_mem_mapping_4  = 0, task_to_mem_mapping_5  = 0, task_to_mem_mapping_6  = 0, task_to_mem_mapping_7  = 0):

        
        
        ''' All the default values of the FARSI should be initialized here. 
            Take all the parameters here and write it to the config files
        '''
        # To do: Implement some default parameters 
        self.helper = helpers()
        self.action_dict = {}

        self.action_dict["pe_allocation_0"] = pe_allocation_0
        self.action_dict["pe_allocation_1"] = pe_allocation_1
        self.action_dict["pe_allocation_2"] = pe_allocation_2

        self.action_dict["mem_allocation_0"] = mem_allocation_0
        self.action_dict["mem_allocation_1"] = mem_allocation_1
        self.action_dict["mem_allocation_2"] = mem_allocation_2
        
        self.action_dict["bus_allocation_0"] = bus_allocation_0
        self.action_dict["bus_allocation_1"] = bus_allocation_1
        self.action_dict["bus_allocation_2"] = bus_allocation_2
        
        self.action_dict["pe_to_bus_connection_0"] = pe_to_bus_connection_0
        self.action_dict["pe_to_bus_connection_1"] = pe_to_bus_connection_1
        self.action_dict["pe_to_bus_connection_2"] = pe_to_bus_connection_2

        self.action_dict["bus_to_bus_connection_0"] = bus_to_bus_connection_0
        self.action_dict["bus_to_bus_connection_1"] = bus_to_bus_connection_1
        self.action_dict["bus_to_bus_connection_2"] = bus_to_bus_connection_2

        self.action_dict["bus_to_mem_connection_0"] = bus_to_mem_connection_0
        self.action_dict["bus_to_mem_connection_1"] = bus_to_mem_connection_1
        self.action_dict["bus_to_mem_connection_2"] = bus_to_mem_connection_2

        self.action_dict["task_to_pe_mapping_0"] = task_to_pe_mapping_0
        self.action_dict["task_to_pe_mapping_1"] = task_to_pe_mapping_1
        self.action_dict["task_to_pe_mapping_2"] = task_to_pe_mapping_2
        self.action_dict["task_to_pe_mapping_3"] = task_to_pe_mapping_3
        self.action_dict["task_to_pe_mapping_4"] = task_to_pe_mapping_4
        self.action_dict["task_to_pe_mapping_5"] = task_to_pe_mapping_5
        self.action_dict["task_to_pe_mapping_6"] = task_to_pe_mapping_6
        self.action_dict["task_to_pe_mapping_7"] = task_to_pe_mapping_7
        
        self.action_dict["task_to_mem_mapping_0"] = task_to_mem_mapping_0
        self.action_dict["task_to_mem_mapping_1"] = task_to_mem_mapping_1
        self.action_dict["task_to_mem_mapping_2"] = task_to_mem_mapping_2
        self.action_dict["task_to_mem_mapping_3"] = task_to_mem_mapping_3
        self.action_dict["task_to_mem_mapping_4"] = task_to_mem_mapping_4
        self.action_dict["task_to_mem_mapping_5"] = task_to_mem_mapping_5
        self.action_dict["task_to_mem_mapping_6"] = task_to_mem_mapping_6
        self.action_dict["task_to_mem_mapping_7"] = task_to_mem_mapping_7
               
        print("Action")
        print(self.action_dict)
        
        self.fitness_hist = []
        self.exp_log_dir = os.path.join(os.getcwd(),"logs")
        
        
        # read from the config file

        config = configparser.ConfigParser()
        config.read("exp_config.ini")
        
        traj_dir = config.get("experiment_configuration", "trajectory_dir")
        exp_name = config.get("experiment_configuration", "exp_name")
        log_dir = config.get("experiment_configuration", "log_dir")
        reward_formulation = config.get("experiment_configuration", "reward_formulation")
        workload = config.get("experiment_configuration", "workload")
        use_envlogger = config.get("experiment_configuration", "use_envlogger")

        # read the all the parameters from exp_config.ini
        self.traj_dir = traj_dir
        self.exp_name = exp_name
        self.log = log_dir
        self.reward_formulation = reward_formulation
        self.use_envlogger = use_envlogger
        self.workload = workload


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
        1) Call the FARSI simulator and return performance, power, and energy
        2) The parameter must be updated before the calling the FARSI simulator
        3)  X is the trace files (.e., Workload)
        '''
        self.bo_steps += 1

        def step_fn(unused_timestep, unused_action, unused_env):
            return {'timestamp': time.time()}
        reward = 0
        self.fitness_hist = {}

        # read from the config file
        os.chdir("/workdir/arch-gym/sims/FARSI_sim")
        config = configparser.ConfigParser()
        config.read("exp_config.ini")
        # read the all the parameters from exp_config.ini
        traj_dir = config.get("experiment_configuration", "trajectory_dir")
        exp_name = config.get("experiment_configuration", "exp_name")
        log_dir = config.get("experiment_configuration", "log_dir")
        reward_formulation = config.get("experiment_configuration", "reward_formulation")
        workload = config.get("experiment_configuration", "workload")
        use_envlogger = config.get("experiment_configuration", "use_envlogger")
        
        env_wrapper = FARSI_sim_wrapper.make_FARSI_sim_env(reward_formulation = reward_formulation, workload=workload)
        os.chdir("/workdir/arch-gym/sims/FARSI_sim")
        FARSI_sim_helper = helpers()
        design_space_mode = "limited"  # ["limited", "comprehensive"]
        SOC_design_space = FARSI_sim_helper.gen_SOC_design_space(env_wrapper, design_space_mode)
        encoding_dictionary = FARSI_sim_helper.gen_SOC_encoding(env_wrapper, SOC_design_space)
        env = self.wrap_in_envlogger(env_wrapper, self.exp_log_dir, use_envlogger)
        
        # check if trajectory directory exists
        if use_envlogger == 'True':
            if not os.path.exists(traj_dir):
                os.makedirs(traj_dir)
        
        
        env.reset()

        action_dict = OrderedDict()
        action_dict["pe_allocation"] =  [self.action_dict["pe_allocation_0"],  
                                         self.action_dict["pe_allocation_1"], 
                                         self.action_dict["pe_allocation_2"]
                                         ]
        action_dict["mem_allocation"] = [self.action_dict["mem_allocation_0"], 
                                         self.action_dict["mem_allocation_1"], 
                                         self.action_dict["mem_allocation_2"]
                                         ] 
        action_dict["bus_allocation"] = [self.action_dict["bus_allocation_0"], 
                                         self.action_dict["bus_allocation_1"], 
                                         self.action_dict["bus_allocation_2"]
                                         ]
        action_dict["pe_to_bus_connection"] =  [self.action_dict["pe_to_bus_connection_0"], 
                                                self.action_dict["pe_to_bus_connection_1"],
                                                self.action_dict["pe_to_bus_connection_2"]
                                                ]
        action_dict["bus_to_bus_connection"] = [self.action_dict["bus_to_bus_connection_0"], 
                                                self.action_dict["bus_to_bus_connection_1"],
                                                self.action_dict["bus_to_bus_connection_2"]
                                                ]
        action_dict["bus_to_mem_connection"] = [self.action_dict["bus_to_mem_connection_0"], 
                                                self.action_dict["bus_to_mem_connection_1"], 
                                                self.action_dict["bus_to_mem_connection_2"]
                                               ]
        action_dict["task_to_pe_mapping"] = [self.action_dict["task_to_pe_mapping_0"], 
                                                    self.action_dict["task_to_pe_mapping_1"], 
                                                    self.action_dict["task_to_pe_mapping_2"], 
                                                    self.action_dict["task_to_pe_mapping_3"], 
                                                    self.action_dict["task_to_pe_mapping_4"], 
                                                    self.action_dict["task_to_pe_mapping_5"], 
                                                    self.action_dict["task_to_pe_mapping_6"], 
                                                    self.action_dict["task_to_pe_mapping_7"], 
                                                    ]
        action_dict["task_to_mem_mapping"] = [self.action_dict["task_to_mem_mapping_0"], 
                                                    self.action_dict["task_to_mem_mapping_1"], 
                                                    self.action_dict["task_to_mem_mapping_2"], 
                                                    self.action_dict["task_to_mem_mapping_3"], 
                                                    self.action_dict["task_to_mem_mapping_4"], 
                                                    self.action_dict["task_to_mem_mapping_5"], 
                                                    self.action_dict["task_to_mem_mapping_6"], 
                                                    self.action_dict["task_to_mem_mapping_7"], 
                                                    ]
        # decode the actions
        flattened_action = []
        for key, value in action_dict.items():
            flattened_action.extend(value) 
        
        action = FARSI_sim_helper.action_decoder_FARSI(flattened_action, encoding_dictionary)
    
        
        _, reward, _, info = env.step(action)


        action_dict_for_logging={}
        for key in action.keys():
            if "encoding" not in key:
                action_dict_for_logging[key] = action[key]
    
        self.fitness_hist["action"] = action_dict_for_logging
        self.fitness_hist["reward"] = reward.item()
        self.fitness_hist["obs"] = [metric.item() for metric in info]

        fitness_filename = os.path.join(self.exp_name)

        # check if log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
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
        return {'pe_allocation_0': self.action_dict['pe_allocation_0'],
                'pe_allocation_1': self.action_dict['pe_allocation_1'],
                'pe_allocation_2': self.action_dict['pe_allocation_2'],

                'mem_allocation_0': self.action_dict['mem_allocation_0'],
                'mem_allocation_1': self.action_dict['mem_allocation_1'],
                'mem_allocation_2': self.action_dict['mem_allocation_2'],

                'bus_allocation_0': self.action_dict['bus_allocation_0'],
                'bus_allocation_1': self.action_dict['bus_allocation_1'],
                'bus_allocation_2': self.action_dict['bus_allocation_2'],

                'pe_to_bus_connection_0': self.action_dict['pe_to_bus_connection_0'],
                'pe_to_bus_connection_1': self.action_dict['pe_to_bus_connection_1'],
                'pe_to_bus_connection_2': self.action_dict['pe_to_bus_connection_2'],

                'bus_to_bus_connection_0': self.action_dict['bus_to_bus_connection_0'],
                'bus_to_bus_connection_1': self.action_dict['bus_to_bus_connection_1'],
                'bus_to_bus_connection_2': self.action_dict['bus_to_bus_connection_2'],

                'bus_to_mem_connection_0': self.action_dict['bus_to_mem_connection_0'],
                'bus_to_mem_connection_1': self.action_dict['bus_to_mem_connection_1'],
                'bus_to_mem_connection_2': self.action_dict['bus_to_mem_connection_2'],

                'task_to_pe_mapping_0': self.action_dict['task_to_pe_mapping_0'],
                'task_to_pe_mapping_1': self.action_dict['task_to_pe_mapping_1'],
                'task_to_pe_mapping_2': self.action_dict['task_to_pe_mapping_2'],
                'task_to_pe_mapping_3': self.action_dict['task_to_pe_mapping_3'],
                'task_to_pe_mapping_4': self.action_dict['task_to_pe_mapping_4'],
                'task_to_pe_mapping_5': self.action_dict['task_to_pe_mapping_5'],
                'task_to_pe_mapping_6': self.action_dict['task_to_pe_mapping_6'],
                'task_to_pe_mapping_7': self.action_dict['task_to_pe_mapping_7'],

                'task_to_mem_mapping_0': self.action_dict['task_to_mem_mapping_0'],
                'task_to_mem_mapping_1': self.action_dict['task_to_mem_mapping_1'],
                'task_to_mem_mapping_2': self.action_dict['task_to_mem_mapping_2'],
                'task_to_mem_mapping_3': self.action_dict['task_to_mem_mapping_3'],
                'task_to_mem_mapping_4': self.action_dict['task_to_mem_mapping_4'],
                'task_to_mem_mapping_5': self.action_dict['task_to_mem_mapping_5'],
                'task_to_mem_mapping_6': self.action_dict['task_to_mem_mapping_6'],
                'task_to_mem_mapping_7': self.action_dict['task_to_mem_mapping_7']
               }
      
    def set_params(self, **params):
        _params = params
        self.action_dict['pe_allocation_0'] = _params['pe_allocation_0']
        self.action_dict['pe_allocation_1'] = _params['pe_allocation_1']
        self.action_dict['pe_allocation_2'] = _params['pe_allocation_2']

        self.action_dict['mem_allocation_0'] = _params['mem_allocation_0']
        self.action_dict['mem_allocation_1'] = _params['mem_allocation_1']
        self.action_dict['mem_allocation_2'] = _params['mem_allocation_2']

        self.action_dict['bus_allocation_0'] = _params['bus_allocation_0']
        self.action_dict['bus_allocation_1'] = _params['bus_allocation_1']
        self.action_dict['bus_allocation_2'] = _params['bus_allocation_2']

        self.action_dict['pe_to_bus_connection_0'] = _params['pe_to_bus_connection_0']
        self.action_dict['pe_to_bus_connection_1'] = _params['pe_to_bus_connection_1']
        self.action_dict['pe_to_bus_connection_2'] = _params['pe_to_bus_connection_2']
        
        self.action_dict['bus_to_bus_connection_0'] = _params['bus_to_bus_connection_0']
        self.action_dict['bus_to_bus_connection_1'] = _params['bus_to_bus_connection_1']
        self.action_dict['bus_to_bus_connection_2'] = _params['bus_to_bus_connection_2']

        self.action_dict['bus_to_mem_connection_0'] = _params['bus_to_mem_connection_0']
        self.action_dict['bus_to_mem_connection_1'] = _params['bus_to_mem_connection_1']
        self.action_dict['bus_to_mem_connection_2'] = _params['bus_to_mem_connection_2']

        self.action_dict['task_to_pe_mapping_0'] = _params['task_to_pe_mapping_0']
        self.action_dict['task_to_pe_mapping_1'] = _params['task_to_pe_mapping_1']
        self.action_dict['task_to_pe_mapping_2'] = _params['task_to_pe_mapping_2']
        self.action_dict['task_to_pe_mapping_3'] = _params['task_to_pe_mapping_3']
        self.action_dict['task_to_pe_mapping_4'] = _params['task_to_pe_mapping_4']
        self.action_dict['task_to_pe_mapping_5'] = _params['task_to_pe_mapping_5']
        self.action_dict['task_to_pe_mapping_6'] = _params['task_to_pe_mapping_6']
        self.action_dict['task_to_pe_mapping_7'] = _params['task_to_pe_mapping_7']

        self.action_dict['task_to_mem_mapping_0'] = _params['task_to_mem_mapping_0']
        self.action_dict['task_to_mem_mapping_1'] = _params['task_to_mem_mapping_1']
        self.action_dict['task_to_mem_mapping_2'] = _params['task_to_mem_mapping_2']
        self.action_dict['task_to_mem_mapping_3'] = _params['task_to_mem_mapping_3']
        self.action_dict['task_to_mem_mapping_4'] = _params['task_to_mem_mapping_4']
        self.action_dict['task_to_mem_mapping_5'] = _params['task_to_mem_mapping_5']
        self.action_dict['task_to_mem_mapping_6'] = _params['task_to_mem_mapping_6']
        self.action_dict['task_to_mem_mapping_7'] = _params['task_to_mem_mapping_7']

        return self


    def log_fitness_to_csv(self, filename):
        df = pd.DataFrame([self.fitness_hist['reward']])
        csvfile = os.path.join(filename, "fitness.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

        df = pd.DataFrame([self.fitness_hist])
        csvfile = os.path.join(filename, "actions.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')
               
