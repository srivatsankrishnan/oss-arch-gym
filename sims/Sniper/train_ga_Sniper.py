import os
import sys
os.sys.path.insert(0, os.path.abspath('../../'))
from configs import arch_gym_configs

from arch_gym.envs.SniperEnv import SniperEnv
from arch_gym.envs.envHelpers import helpers
from sko.GA import GA
import json
import numpy as np
import collections
import time
import json

import pandas as pd
import matplotlib.pyplot as plt


def transform_actions(p):
    # convert each element in the list to closest power of 2
    # Since sniper wants everything in powers of 2
    
    
    return [np.power(2, np.round(np.log2(x))) for x in p]

def sniper_optimization_func_parallel(p):
    # To do: Leverage the sniper batch mode
    # number of agents (parallel jobs )
    print("Number of Agents:", p.shape[0])
    
    p = transform_actions(p)
    

    env = SniperEnv()
    
    # Clear the old results before new run
    env.reset_multiagent()
    
    action_dicts = collections.defaultdict(dict)
    
    # convert list of numpy array to list of lists
 
    for i in range(len(p)):
        agent_name = "agent_" + str(i)
        action_dicts[agent_name]["core_dispatch_width"] = int(p[i][0])
        action_dicts[agent_name]["core_window_size"] = int(p[i][1])
        action_dicts[agent_name]["core_outstanding_loads"] = int(p[i][2])
        action_dicts[agent_name]["core_outstanding_stores"] = int(p[i][3])
        action_dicts[agent_name]["core_commit_width"] = int(p[i][4])
        action_dicts[agent_name]["core_rs_entries"] = int(p[i][5])
        action_dicts[agent_name]["l1_icache_size"] = int(p[i][6])
        action_dicts[agent_name]["l1_dcache_size"] = int(p[i][7])
        action_dicts[agent_name]["l2_cache_size"] = int(p[i][8])
        action_dicts[agent_name]["l3_cache_size"] = int(p[i][9])

    obs, rewards, _, _ = env.step_multiagent(action_dicts)

    # clean up after the run
    env.reset_multiagent()
    
    

    return [each for each in rewards]
    
def sniper_optimization_function(p):

    p = transform_actions(p)
    print("p: ", p)
    action_dict = {}
    action_dict["core_dispatch_width"] = p[0]
    action_dict["core_window_size"] = p[1]
    action_dict["core_outstanding_loads"] = p[2]
    action_dict["core_outstanding_stores"] = p[3]
    action_dict["core_commit_width"] = p[4]
    action_dict["core_rs_entries"] = p[5]
    action_dict["l1_icache_size"] = p[6]
    action_dict["l1_dcache_size"] = p[7]
    action_dict["l2_cache_size"] = p[8]
    action_dict["l3_cache_size"] = p[9]
    
    # Initialize the environment
    env = SniperEnv()
    env.reset()
    obs,_,_,_ = env.step(action_dict)

    # lowers the runtime
    return obs[0]

if __name__ == '__main__':
    # initialize GA

    data = sys.argv[1]
    prob_mut = float(data.split("_")[2])
    num_agents = int(data.split("_")[3])

    ga = GA(
        func= sniper_optimization_func_parallel, 
        n_dim=10, 
        size_pop=num_agents, 
        max_iter=arch_gym_configs.num_iter,
        prob_mut=prob_mut,
        lb=[2, 16, 32, 24, 32, 18, 16, 16, 128, 4096],
        ub=[8, 512, 96, 64, 192, 72, 128, 128, 2048, 16384],
        precision=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )

    # run GA
    best_x, best_y = ga.run()
    

# log the best parameters for the best fitness
dir_name = sys.argv[1]
best_parms = pd.DataFrame(transform_actions(best_x))
best_parms.to_csv(dir_name + "/best_parms.csv")
Y_history = pd.DataFrame(ga.all_history_Y)
Y_history.to_csv(dir_name + "/ga_snipertest.csv")

fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.savefig(dir_name + "/gatest_sniper_test.png")

print("Done with Experiments! Shutting down the instance")
time.sleep(10)
