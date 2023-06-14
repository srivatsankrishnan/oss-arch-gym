import os
os.sys.path.insert(0, os.path.abspath('../../configs'))

import arch_gym_configs
from arch_gym.envs.SniperEnv import SniperEnv
from arch_gym.envs.envHelpers import helpers

import json
import numpy as np
import random
import time
import pandas as pd
import collections



class SniperRandomWalker():
    def __init__(self):
        # Define your parameter set here
        self.param_core_dispatch_width = [2,4,8]
        self.param_core_window_size = [16,32,64,128,256,512]
        self.param_core_outstanding_loads = [32,48,72,96]
        self.param_core_outsanding_stores = [24,32,48,64]
        self.param_core_commit_width = [32,64,96,128,192]
        self.param_core_rs_entries = [18,24,36,48,72]
        self.param_l1_icache_size = [4,8,16,32,64,128]
        self.param_l1_dcache_size = [4,8,16,32,64,128]
        self.param_l2_cache_size = [128,256,512,1024,2048]
        self.param_l3_cache_size = [4096, 8192, 16384]


        # total parameter space
        # python new line long line

        self.param_space =  (len(self.param_core_dispatch_width) * 
                            len(self.param_core_window_size) * 
                            len(self.param_core_outstanding_loads) *
                            len(self.param_core_outsanding_stores) *
                            len(self.param_core_commit_width) * 
                            len(self.param_core_rs_entries) * 
                            len(self.param_l1_icache_size) * 
                            len(self.param_l1_dcache_size) * 
                            len(self.param_l2_cache_size) * 
                            len(self.param_l3_cache_size)) 
                       
        # pretty pring the parameter space
        print("Parameter space:",self.param_space)
   
        # Stopping conditions
        self.max_steps = 2

        # Initialize the environment
        self.env = SniperEnv()

    
    def take_random_actions(self, num_agents):
        action_dicts = collections.defaultdict(dict)

        for i in range(num_agents):
            agent_name = "agent_" + str(i)
            action_dicts[agent_name]["core_dispatch_width"] = random.sample(self.param_core_dispatch_width,1)[0]
            action_dicts[agent_name]["core_window_size"] = random.sample(self.param_core_window_size,1)[0]
            action_dicts[agent_name]["core_outstanding_loads"] = random.sample(self.param_core_outstanding_loads,1)[0]
            action_dicts[agent_name]["core_outstanding_stores"] = random.sample(self.param_core_outsanding_stores,1)[0]
            action_dicts[agent_name]["core_commit_width"] = random.sample(self.param_core_commit_width,1)[0]
            action_dicts[agent_name]["core_rs_entries"] = random.sample(self.param_core_rs_entries,1)[0]
            action_dicts[agent_name]["l1_icache_size"] = random.sample(self.param_l1_icache_size,1)[0]
            action_dicts[agent_name]["l1_dcache_size"] = random.sample(self.param_l1_dcache_size,1)[0]
            action_dicts[agent_name]["l2_cache_size"] = random.sample(self.param_l2_cache_size,1)[0]
            action_dicts[agent_name]["l3_cache_size"] = random.sample(self.param_l3_cache_size,1)[0]

        return action_dicts

if __name__ == "__main__":
    # Initialize the environment
    env = SniperEnv()
    agents = SniperRandomWalker()

    # number of agents 
    num_agents = 2

    action_dicts = agents.take_random_actions(num_agents)
    

    # run sniper in batch model with batched actions
  
    obs, reward, done, info = env.step_multiagent(action_dicts)
    


    