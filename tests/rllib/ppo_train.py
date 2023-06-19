import ray
import ray.rllib.agents.ppo as ppo
import shutil
import os

ray.shutdown()
ray.init(ignore_reinit_error=True)

#print("Dashboard URL: http://{}".format(ray.get_webui_url()))

CHECKPOINT_ROOT = "tmp/ppo/cart"
ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

#SELECT_ENV = "Taxi-v3"
SELECT_ENV = "CartPole-v1"
#SELECT_ENV = "FrozenLake-v0"
config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"
agent = ppo.PPOTrainer(config, env=SELECT_ENV)


N_ITER = 30
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"


for n in range(N_ITER):
    result = agent.train()
    filename = agent.save(CHECKPOINT_ROOT)

    print(s.format(
        n+1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"],
        filename
    ))
