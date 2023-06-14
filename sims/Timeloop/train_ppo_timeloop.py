import os
import sys

from absl import flags
from acme.agents.jax import ppo
from acme import specs

from absl import app
from acme.jax import experiments


from arch_gym.envs import timeloop_acme_wrapper

print("Import Successful")

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_steps', 1000, 'Number of training steps.')
flags.DEFINE_integer('eval_every', 50, 'Number of evaluation steps.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episode.')
flags.DEFINE_integer('seed', 1234, 'Random seed.')


def build_experiment_config():
    """Builds the experiment configuration."""
    env = timeloop_acme_wrapper.make_timeloop_env(maximize_action=True, convert_multi_discrete=True)
    env_spec = specs.make_environment_spec(env)
    config = ppo.PPOConfig(entropy_cost=0.05, learning_rate=1e-2)
    ppo_builder = ppo.PPOBuilder(config)

    layer_sizes = (256, 256, 256)
    def make_eval_policy(network): return ppo.make_inference_fn(network, True)

    return experiments.Config(
        builder=ppo_builder,
        environment_factory=lambda seed: env,
        network_factory=lambda spec: ppo.make_networks(env_spec, layer_sizes),
        policy_network_factory=ppo.make_inference_fn,
        eval_policy_network_factory=make_eval_policy,
        seed=FLAGS.seed,
        max_number_of_steps=FLAGS.num_steps)


def main(_):
    config = build_experiment_config()
    experiments.run_experiment(experiment=config,
                               eval_every=FLAGS.eval_every,
                               num_eval_episodes=FLAGS.eval_episodes)


if __name__ == '__main__':
    app.run(main)
