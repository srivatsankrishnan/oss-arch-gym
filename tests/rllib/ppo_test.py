import ray
import ray.rllib.agents.ppo as ppo
import shutil
import os

ray.shutdown()
ray.init(ignore_reinit_error=True)

#print("Dashboard URL: http://{}".format(ray.get_webui_url()))

CHECKPOINT_ROOT = "tmp/ppo/taxi"
ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

SELECT_ENV = "Taxi-v3"

config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"
agent = ppo.PPOTrainer(config, env=SELECT_ENV)

policy = agent.get_policy()
model = policy.model
print(model.base_model.summary())
