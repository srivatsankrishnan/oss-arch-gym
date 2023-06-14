import ray
import ray.rllib.agents.ppo as ppo

from ray.tune.registry import register_env
import sys
sys.path.append("/home/skrishnan/workspace/arch_gym/arch_gym")
from arch_gym.envs.simpleEnv import simpleEnv


import shutil
import os

select_env = "simpleEnv-v0"

register_env(select_env, lambda config: simpleEnv())

chkpt_root = "tmp/simpleEnv"
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
ray_results = "{}/ray_results/".format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"
agent = ppo.PPOTrainer(config, env=select_env)

status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
n_iter = 50
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


