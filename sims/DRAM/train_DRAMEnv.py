import os

os.sys.path.insert(0, os.path.abspath('../configs'))

import configs
import ray
import gym
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
import sys
from arch_gym.envs.DRAMEnv import DRAMEnv

import shutil


select_env = "DRAMEnv-v0"

register_env(select_env, lambda config: DRAMEnv())

# for testing environment
'''
env = gym.make(select_env)
state = env.reset()
print(state, type(state),state.shape)

#sys.exit()
'''

 
chkpt_root = "tmp/DRAMEnv"
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
ray_results = "{}/ray_results/".format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

config = ppo.DEFAULT_CONFIG.copy()


config["log_level"] = "WARN"
config["num_workers"] = 2
config["sgd_minibatch_size"] = 32
config["model"]["fcnet_hiddens"] = [100,100]


agent = ppo.PPOTrainer(config, env=select_env)



status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
result = agent.train()
'''
n_iter = 1
for n in range(n_iter):
    result = agent.train()
    chkpt_file = agent.save(chkpt_root)
    print(status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            chkpt_file
            ))
'''
