from sklearn.base import BaseEstimator, ClassifierMixin
import os
os.sys.path.insert(0, os.path.abspath('/../'))
from configs import arch_gym_configs
import json
from arch_gym.envs.envHelpers import helpers
from arch_gym.envs.DRAMEnv import DRAMEnv
from arch_gym.envs.dramsys_wrapper import make_dramsys_env
import configparser
import envlogger
import sys
import numpy as np
import pandas as pd
import time

from absl import logging
from absl import flags


class DRAMSysEstimator(BaseEstimator):

    def __init__(self, PagePolicy='Open', Scheduler='Fifo', SchedulerBuffer="Bankwise", 
                RequestBufferSize=1, RespQueue="Fifo", RefreshPolicy='NoRefresh', 
                RefreshMaxPulledin=1, RefreshMaxPostponed=1, Arbiter='Simple', MaxActiveTransactions=128,
                exp_name="test", traject_dir="traj"):
        
        ''' All the default values of the DRAMSys should be initialized here. 
            Take all the parameters here and write it to the config files
        '''
        # To do: Implement some default parameters 
        self.env = DRAMEnv()
        self.helper = helpers()
        self.action_dict = {}
        self.action_dict['PagePolicy'] = PagePolicy
        self.action_dict['Scheduler'] = Scheduler
        self.action_dict['SchedulerBuffer'] = SchedulerBuffer
        self.action_dict['RequestBufferSize'] = RequestBufferSize
        self.action_dict['RespQueue'] = RespQueue
        self.action_dict['RefreshPolicy'] = RefreshPolicy
        self.action_dict['RefreshMaxPostponed'] = RefreshMaxPostponed
        self.action_dict['RefreshMaxPulledin'] = RefreshMaxPulledin
        self.action_dict['Arbiter'] = Arbiter
        self.action_dict['MaxActiveTransactions'] = MaxActiveTransactions
        
        
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

        env_wrapper = make_dramsys_env(reward_formulation = reward_formulation)
        
        # check if trajectory directory exists
        if use_envlogger == 'True':
            if not os.path.exists(traj_dir):
                os.makedirs(traj_dir)
        # check if log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        env = self.wrap_in_envlogger(env_wrapper, self.exp_log_dir, use_envlogger)
        env.reset()
        _, reward, _, info = env.step(self.action_dict)

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
        return {'PagePolicy': self.action_dict['PagePolicy'],
                'Scheduler': self.action_dict['Scheduler'],
                'SchedulerBuffer': self.action_dict['SchedulerBuffer'],
                'RequestBufferSize': self.action_dict['RequestBufferSize'],
                'RespQueue': self.action_dict['RespQueue'],
                'RefreshPolicy': self.action_dict['RefreshPolicy'],
                'RefreshMaxPostponed': self.action_dict['RefreshMaxPostponed'],
                'RefreshMaxPulledin': self.action_dict['RefreshMaxPulledin'],
                'Arbiter': self.action_dict['Arbiter'],
                'MaxActiveTransactions' : self.action_dict['MaxActiveTransactions']
               }
      
    def set_params(self, **params):
        _params = params
        self.action_dict['PagePolicy'] = _params['PagePolicy']
        self.action_dict['Scheduler'] = _params['Scheduler']
        self.action_dict['SchedulerBuffer'] = _params['SchedulerBuffer']
        self.action_dict['RequestBufferSize'] = _params['RequestBufferSize']
        self.action_dict['RespQueue'] = _params['RespQueue']
        self.action_dict['RefreshPolicy'] = _params['RefreshPolicy']
        self.action_dict['RefreshMaxPostponed'] = _params['RefreshMaxPostponed']
        self.action_dict['RefreshMaxPulledin'] = _params['RefreshMaxPulledin']
        self.action_dict['Arbiter'] = _params['Arbiter']
        self.action_dict['MaxActiveTransactions'] = _params['MaxActiveTransactions']

        return self
    def calculate_reward(self, energy, latency):
        reward = np.sum([np.square(configs.target_energy - energy),np.square(configs.target_latency - latency)])
        reward = np.sqrt(reward)
        return reward

    def read_modify_write_simconfigs(self):
        mem_ctrl_filename = "policy.json"
        op_success = False
        full_path = os.path.join(configs.dram_mem_controller_config,mem_ctrl_filename)
        
        try:
            with open (full_path, "r") as JsonFile:
                data = json.load(JsonFile)
                data['mcconfig']['PagePolicy'] = self.PagePolicy
                data['mcconfig']['Scheduler'] = self.Scheduler
                data['mcconfig']['SchedulerBuffer'] = self.SchedulerBuffer
                data['mcconfig']['RequestBufferSize'] = self.RequestBufferSize
                data['mcconfig']['RespQueue'] = self.RespQueue
                data['mcconfig']['RefreshPolicy'] = self.RefreshPolicy
                data['mcconfig']['RefreshMaxPostponed'] = self.RefreshMaxPostponed
                data['mcconfig']['RefreshMaxPulledin'] = self.RefreshMaxPulledin
                data['mcconfig']['Arbiter'] = self.Arbiter
                data['mcconfig']['MaxActiveTransactions'] = self.MaxActiveTransactions

                with open (full_path, "w") as JsonFile:
                    json.dump(data,JsonFile)
                op_success = True
        except Exception as e:
            print(str(e))
            op_success = False
        return op_success

    def log_fitness_to_csv(self, filename):
        df = pd.DataFrame([self.fitness_hist['reward']])
        csvfile = os.path.join(filename, "fitness.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

        df = pd.DataFrame([self.fitness_hist])
        csvfile = os.path.join(filename, "actions.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')
               



      