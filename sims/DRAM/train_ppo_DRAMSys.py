import os
import sys


os.sys.path.insert(0, os.path.abspath('../../'))
os.sys.path.insert(0, os.path.abspath('../../acme'))
print(os.sys.path)
from acme.agents.jax import ppo
from acme import wrappers
from acme import specs

import helpers
from absl import app
from absl import flags
from absl import logging
from acme.utils import lp_utils
from acme.jax import experiments
from acme.utils.loggers.tf_summary import TFSummaryLogger
from acme.utils.loggers.terminal import TerminalLogger
from acme.utils.loggers.csv import CSVLogger
from acme.utils.loggers import aggregators
from acme.utils.loggers import base

from arch_gym.envs import dramsys_wrapper

print("Import Successful")

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_steps', 1000, 'Number of training steps.')
flags.DEFINE_integer('eval_every', 50, 'Number of evaluation steps.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episode.')
flags.DEFINE_integer('seed', 1234, 'Random seed.')
flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate.')
flags.DEFINE_float('entropy_cost', 0.01, 'Entropy cost.')
flags.DEFINE_float('ppo_clipping_epsilon', 0.2, 'PPO clipping epsilon.')
flags.DEFINE_bool('clip_value', True, 'Clip value.')
flags.DEFINE_string('summarydir', './logs', 'Directory to save summaries.')
flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a '
    'distributed way (the default is a single-threaded agent)')


def _logger_factory(logger_label: str) -> base.Logger:
  """logger factory."""
  
  if logger_label == 'train':
      terminal_logger = TerminalLogger(label=logger_label, print_fn=logging.info)
      summarydir = os.path.join(FLAGS.summarydir, logger_label)
      tb_logger = TFSummaryLogger(summarydir, label=logger_label)
      csv_logger = CSVLogger(summarydir, label=logger_label)
      serialize_fn = base.to_numpy
      logger = aggregators.Dispatcher([terminal_logger, tb_logger, csv_logger], serialize_fn)
      return logger
  elif logger_label == 'learner':
      terminal_logger = TerminalLogger(label=logger_label, print_fn=logging.info)
      summarydir = os.path.join(FLAGS.summarydir, logger_label)
      tb_logger = TFSummaryLogger(summarydir, label=logger_label)
      csv_logger = CSVLogger(summarydir, label=logger_label)
      serialize_fn = base.to_numpy
      logger = aggregators.Dispatcher([terminal_logger, tb_logger, csv_logger], serialize_fn)
      return logger
  elif logger_label == 'eval':
      terminal_logger = TerminalLogger(label=logger_label, print_fn=logging.info)
      summarydir = os.path.join(FLAGS.summarydir, logger_label)
      tb_logger = TFSummaryLogger(summarydir, label=logger_label)
      csv_logger = CSVLogger(summarydir, label=logger_label)
      serialize_fn = base.to_numpy
      logger = aggregators.Dispatcher([terminal_logger, tb_logger, csv_logger], serialize_fn)
      return logger
  else:
    raise ValueError(
        f'Improper value for logger label. Logger_label is {logger_label}')

def build_experiment_config():
    """Builds the experiment configuration."""
    env = dramsys_wrapper.make_dramsys_env()
    env_spec = specs.make_environment_spec(env)
    config = ppo.PPOConfig(entropy_cost=FLAGS.entropy_cost,
                           learning_rate=FLAGS.learning_rate,
                           ppo_clipping_epsilon=FLAGS.ppo_clipping_epsilon,
                           clip_value=FLAGS.clip_value,
                           )
    ppo_builder = ppo.PPOBuilder(config)

    layer_sizes = (32, 32, 32)
    make_eval_policy = lambda network: ppo.make_inference_fn(network, True)

    return experiments.ExperimentConfig(
        builder=ppo_builder,
        environment_factory=lambda seed: env,
        network_factory=lambda spec: ppo.make_networks(env_spec, layer_sizes),
        policy_network_factory = ppo.make_inference_fn,
        eval_policy_network_factory = make_eval_policy,
        seed = FLAGS.seed,
        logger_factory=_logger_factory,
        max_num_actor_steps=FLAGS.num_steps)

def main(_):
  config = build_experiment_config()
  if FLAGS.run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=4)
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
  else:
    experiments.run_experiment(
        experiment=config,
        eval_every=FLAGS.eval_every,
        num_eval_episodes=FLAGS.eval_episodes)

if __name__ == '__main__':
   app.run(main)