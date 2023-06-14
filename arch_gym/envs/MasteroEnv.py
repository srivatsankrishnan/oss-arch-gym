import os
import sys
import csv

settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
os.sys.path.insert(0, settings_dir_path)

os.sys.path.insert(0, settings_dir_path + '/../../')

from configs import arch_gym_configs
import gym
import glob
from gym.utils import seeding
from envHelpers import helpers

from loggers import write_csv
import numpy as np

# ToDo: Have a configuration for Arch-Gym to manipulate this methods

import sys

from subprocess import Popen, PIPE
import time
import re
import numpy

import random
import pandas as pd
from math import ceil

class MasteroEnv(gym.Env):

    def __init__(self,
                 rl_form: str = 'tdm',
                 rl_algo: str = 'ppo',
                 max_steps: int = 100,
                 num_agents: int = 1,
                 reward_formulation: str = 'latency',
                 reward_scaling: str = 'false',
                 mapping_file: str = 'mapping.csv',
                 workload: str = 'resnet18',
                 layer_id: int = 2,
                 noc_bw: int = 1073741824,
                 offchip_bw: int = 1073741824,
                 l1_size: int = 1073741824,
                 l2_size: int = 1073741824,
                 num_pe: int = 1024,
                 ):
        self._executable = arch_gym_configs.exe_file
        self.mapping_file = mapping_file
        self.num_agents = num_agents
        self.NocBW = noc_bw
        self.offchipBW = offchip_bw
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.num_pe = num_pe
        self.max_steps = max_steps
        self.rl_form = rl_form
        self.steps = 0
        self.episode = 0
        self.workload = workload
        self.layer_id = layer_id
        self.helpers = helpers() 

        self.dimension, _ = self.helpers.get_dimensions(workload=self.workload, layer_id=self.layer_id)
        print("dimension: ", self.dimension) 
        
        if self.rl_form == 'macme':
            self.observation_space = [
                gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)]* self.num_agents

            self.action_space = [    
                # seed for permuting
                gym.spaces.Discrete(720),
                
                # C,K,X, Y
                gym.spaces.Discrete(4),
                
                # S
                gym.spaces.Discrete(2),
                
                # R
                gym.spaces.Discrete(2),
                
                # K
                gym.spaces.Discrete(self.dimension['K']),

                # C
                gym.spaces.Discrete(self.dimension['C']),

                # X
                gym.spaces.Discrete(self.dimension['X']),

                # Y
                gym.spaces.Discrete(self.dimension['Y']),
                
                # C,K,X, Y
                gym.spaces.Discrete(4),

                 # S
                gym.spaces.Discrete(2),
                
                # R
                gym.spaces.Discrete(2),
                
                # K
                gym.spaces.Discrete(self.dimension['K']),

                # C
                gym.spaces.Discrete(self.dimension['C']),

                # X
                gym.spaces.Discrete(self.dimension['X']),

                # Y
                gym.spaces.Discrete(self.dimension['Y']),

                # seed for permuting
                gym.spaces.Discrete(720),

                # Num PE
                gym.spaces.Discrete(10)
            ]
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=1, high=2, shape=(17,), dtype=np.float32)

    def clean_sim_files(self, file_path):

        # split the file_path into file name and directory
        dir_path = os.path.dirname(file_path)

        # use glob to get the list of files
        csv_files = glob.glob(os.path.join(dir_path, '*.csv'))
        m_files = glob.glob(os.path.join(dir_path, '*.m'))

        # remove the files
        for csv_files in csv_files:
            csv_files = os.path.join(file_path, csv_files)
            if os.path.exists(csv_files):
                os.remove(csv_files)
        
        for m_files in m_files:
            m_files = os.path.join(file_path, m_files)
            if os.path.exists(m_files):
                os.remove(m_files)

    def step(self, action):
        
        self.steps += 1
        done = False

        print("action: ", action)

        if self.rl_form == 'macme':
            # TODO(Sri) implement this
            action_decoded = self.helpers.decode_action_list_multiagent(action)
        else:
            action_discretized = self.helpers.decode_action_list_rl(action, self.dimension)
            action_decoded = self.helpers.decode_action_list(action_discretized)

        m_file = "{}".format(random.randint(0, 2**32))
        
        arch_configs = {
            "NocBW": self.NocBW,
            "offchipBW": self.offchipBW,
            "l1_size": self.l1_size,
            "l2_size": self.l2_size,
            "num_pe": self.num_pe
        }
        # write the action to the file
        m_file_path = self.helpers.write_maestro(indv = action_decoded, workload=self.workload, layer_id = self.layer_id, m_file = m_file)

        # run the maestro
        obs = self.helpers.run_maestro(self._executable, m_file, arch_configs)

        obs = obs.reshape(4,)
        print("obs: ", obs)
        reward = self.calculate_reward(obs)
        print("reward: ", reward)
        if(self.steps == self.max_steps):
            done = True
            print("Maximum steps per episodes reached!")
            self.reset()
            self.episode +=1
        
        if self.rl_form == "macme":
            obs = [obs.copy()] * self.num_agents

        # clean the files
        self.clean_sim_files(m_file_path)
        
        return obs, reward, done, {}

    def calculate_reward(self, stats):

        # flatten the list 
        runtime = stats[0]
        return 1/runtime

    def reset(self):
        self.steps = 0
        # get the current directory
        file_path = os.path.dirname(os.path.realpath(__file__))

        # find wildcard csv and m files
        csv_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
        m_files = [f for f in os.listdir(file_path) if f.endswith('.m')]

        # get the file path
        file_path = os.path.dirname(os.path.realpath(__file__))
        
        # remove the files
        for csv_files in csv_files:
            csv_files = os.path.join(file_path, csv_files)
            if os.path.exists(csv_files):
                os.remove(csv_files)
        
        for m_files in m_files:
            m_files = os.path.join(file_path, m_files)
            if os.path.exists(m_files):
                os.remove(m_files)
        if self.rl_form == 'macme':
            obs = [np.zeros(self.observation_space[0].shape)] * self.num_agents
        else:
            obs = np.zeros(self.observation_space.shape)
        return obs 


# For testing
if __name__ == "__main__":
        
        exe_file = "../../cost_model/maestro"
        mapping_file = "1322331445"
        workload = "resnet18"
        noc_bw = 1073741824
        offchip_bw = 1073741824
        l1_size = 1073741824
        l2_size = 1073741824
        num_pe = 1024
        
        
        env = MasteroEnv(
            mapping_file = mapping_file,
            noc_bw = noc_bw,
            offchip_bw = offchip_bw,
            l1_size = l1_size,
            l2_size = l2_size,
            num_pe = num_pe,
            workload = workload
        )
        
        dimension, _ = env.helpers.get_dimensions(workload=workload, layer_id=2)
        print("dimension: ", dimension) 

        lb=[0, 0,  dimension['S']-1, dimension['R']-1, 1, 1, 1, 1, 0, dimension['S']-1, dimension['R']-1, 1, 1, 1, 1, 0,1],
        ub=[(2**32)-1, 3, dimension['S'], dimension['R'],
            dimension['K'], dimension['C'], dimension['X'],
            dimension['Y'], 3, dimension['S'], dimension['R'],
            dimension['K'], dimension['C'], dimension['X'], dimension['Y'], (2**32)-1, 1024]

        # generate random action with lower bound of lb and upper bound of ub.
        action = np.random.randint(lb, ub)

        # reset the environment
        env.reset()

        # take a step
        obs, reward, _,_ = env.step(action)
        
        print("obs: ", obs)
        print("reward: ", reward)
        # reset the environment
        env.reset()

 