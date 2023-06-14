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



class SniperRandomWalker():
    def __init__(self):
        # Define your parameter set here
        self.param_core_dispatch_width = [2,4,8]
        self.param_core_window_size = [16,32,64,128,256,512]
        self.param_l1_icache_size = [4,8,16,32,64,128]
        self.param_l1_dcache_size = [4,8,16,32,64,128]
        self.param_l2_cache_size = [128,256,512,1024,2048]
        self.param_l3_cache_size = [4096, 8192, 16384]

        # total parameter space
        # python new line long line

        self.param_space =  (len(self.param_core_dispatch_width) * 
                            len(self.param_core_window_size) * 
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

    def take_random_action(self):
        action_dict = {}
        action_dict["core_dispatch_width"] = random.sample(self.param_core_dispatch_width,1)[0]
        action_dict["core_window_size"] = random.sample(self.param_core_window_size,1)[0]
        action_dict["l1_icache_size"] = random.sample(self.param_l1_icache_size,1)[0]
        action_dict["l1_dcache_size"] = random.sample(self.param_l1_dcache_size,1)[0]
        action_dict["l2_cache_size"] = random.sample(self.param_l2_cache_size,1)[0]
        action_dict["l3_cache_size"] = random.sample(self.param_l3_cache_size,1)[0]


        return action_dict


if __name__ == "__main__":
    # Initialize the random walker
    randomwalker = SniperRandomWalker()

    metric_stats = []
    for i in range(randomwalker.max_steps):
        print("Step:",i)
        # reset the environment
        obs = randomwalker.env.reset()
        time.sleep(2)
        action_dict = randomwalker.take_random_action()
        obs,_,_,_ = randomwalker.env.step(action_dict)
        time.sleep(2)
        metric_stats.append(obs)

    # convert metric_states to pandas dataframe
    metric_stats = np.array(metric_stats)
    metric_stats = pd.DataFrame(metric_stats)

    # add column names for dataframe
    
    metric_stats.columns = ["runtime",
                            "branch_predictor_mpki",
                            "branch_mispredict_rate",
                            "l1_dcache_mpki",
                            "l1_dcache_missrate",
                            "l1_icache_mpki",
                            "l1_icache_missrate",
                            "l2_mpki",
                            "l2_missrate",
                            "l3_mpki",
                            "l3_missrate"]

    # save the dataframe to csv
    metric_stats.to_csv("sniper_randomwalk.csv")

    