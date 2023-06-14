import os
os.sys.path.insert(0, os.path.abspath('../../configs'))
import arch_gym_configs
from arch_gym.envs.SniperEnv import SniperEnv
from arch_gym.envs.envHelpers import helpers

import json
import numpy as np
import random
import time



sniper = SniperEnv()

# Test reset works
obs = sniper.reset()
print(obs)


# Test Sniper Step works
# sleep for 2 seconds
#time.sleep(5)
obs = sniper.runSniper()
print(obs)

# Test action config works

action_dict = {}

core_dispatch_width = [2,4,8]
action_dict["core_dispatch_width"] = random.sample(core_dispatch_width,1)[0]
core_window_size = [16,32,64,128,256,512]
action_dict["core_window_size"] = random.sample(core_window_size,1)[0]

l1_icache_size = [4,8,16,32,64,128]
action_dict["l1_icache_size"] = random.sample(l1_icache_size,1)[0]
l1_dcache_size = [4,8,16,32,64,128]
action_dict["l1_dcache_size"] = random.sample(l1_dcache_size,1)[0]
l2_cache_size = [128,256,512,1024,2048]
action_dict["l2_cache_size"] = random.sample(l2_cache_size,1)[0]

l3_cache_size = [4096, 8192, 16384]
action_dict["l3_cache_size"] = random.sample(l3_cache_size,1)[0]
print(action_dict)

status = sniper.actionToConfigs(action_dict)
print(status)

obs = sniper.reset()

time.sleep(2)

obs = sniper.runSniper()
print(obs)

