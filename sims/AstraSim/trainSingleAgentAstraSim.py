import os
import sys
import json
import subprocess
from typing import Optional

os.sys.path.insert(0, os.path.abspath('../../'))
from configs import arch_gym_configs
os.sys.path.insert(0, os.path.abspath('../../acme/'))
# print(os.sys.path)
# sys.exit()
import envlogger
from acme.agents.jax import ppo
from acme.agents.jax import sac
from acme import wrappers
from acme import specs

from absl import app
from absl import flags
from absl import logging
from acme.utils import lp_utils
from acme.jax import experiments
from acme.agents.jax import normalization

from acme.utils.loggers.tf_summary import TFSummaryLogger
from acme.utils.loggers.terminal import TerminalLogger
from acme.utils.loggers.csv import CSVLogger
from acme.utils.loggers import aggregators
from acme.utils.loggers import base

from arch_gym.envs import AstraSimWrapper
from arch_gym.envs.envHelpers import helpers

# define Astrasim version
VERSION = 2

FLAGS = flags.FLAGS

# Workload to run for training

_WORKLOAD = flags.DEFINE_string('workload', 'resnet18', 'Workload to run for training')

# select which RL algorithm to use 
_RL_AGO = flags.DEFINE_string('rl_algo', 'ppo', 'RL algorithm.')

# select which RL form to use
_RL_FORM = flags.DEFINE_string('rl_form', 'sa1', 'RL form.')

# Acceptable values for reward: power, latency, and both (both means latency & power)
_REWARD_FORM = flags.DEFINE_string('reward_form', 'both', 'Reward form.')

# Scale reward
_REWARD_SCALE = flags.DEFINE_string('reward_scale', 'false', 'Scale reward.')

# Hyperparameters for each RL algorithm
_NUM_STEPS = flags.DEFINE_integer('num_steps', 1, 'Number of training steps.')
_EVAL_EVERY = flags.DEFINE_integer('eval_every', 1, 'Number of evaluation steps.')
_EVAL_EPISODES = flags.DEFINE_integer('eval_episodes', 1, 'Number of evaluation episode.')
_SEED = flags.DEFINE_integer('seed', 42, 'Random seed.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate.')


# Hyperparameters for PPO 
_ENTROPY_COST = flags.DEFINE_float('entropy_cost', 0.1, 'Entropy cost.')
_PPO_CLIPPING_EPSILON = flags.DEFINE_float('ppo_clipping_epsilon', 0.2, 'PPO clipping epsilon.')
_CLIP_VALUE = flags.DEFINE_bool('clip_value', False, 'Clip value.')

# Experiment setup related parameters
_SUMMARYDIR = flags.DEFINE_string('summarydir', './all_logs', 'Directory to save summaries.')
_ENVLOGGER_DIR = flags.DEFINE_string('envlogger_dir', './rl_directory', 'Directory to save envlogger.')
_USE_ENVLOGGER = flags.DEFINE_bool('use_envlogger', True, 'Use envlogger.')
_RUN_DISTRIBUTED = flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a '
    'distributed way (the default is a single-threaded agent)')

# Experimental feature to scale RL policy parameters. Ideally want to keep it same as number
# of agents used in multi-agent training.
flags.DEFINE_integer("params_scaling", 1, "Number of training steps")
flags.DEFINE_string('knobs', 'astrasim_220_example/knobs.py', "path to knobs spec file")
flags.DEFINE_string('network', 'astrasim_220_example/network_input.yml', "path to network input file")
flags.DEFINE_string('system', 'astrasim_220_example/system_input.json', "path to system input file")
flags.DEFINE_string('workload_file', 'astrasim_220_example/workload_cfg.json', "path to workload input file")
flags.DEFINE_bool('congestion_aware', True, "astra-sim congestion aware or not")
# FLAGS.workload_file = astrasim_220_example/workload_cfg.json if GENERATE_WORKLOAD = True
# FLAGS.workload_file = astrasim_220_example/workload-et/generated if GENERATE_WORKLOAD = False

astrasim_helpers = helpers()
settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
proj_root_path = os.path.abspath(settings_dir_path)

def get_directory_name():
    _EXP_NAME = 'Algo_{}_rlform_{}_num_steps_{}_seed_{}_lr_{}_entropy_{}'.format(_RL_AGO.value, _RL_FORM.value,_NUM_STEPS.value, _SEED.value, _LEARNING_RATE.value, _ENTROPY_COST.value)

    return _EXP_NAME


def wrap_in_envlogger(env, envlogger_dir):
    metadata = {
        'agent_type': FLAGS.rl_algo,
        'rl_form': FLAGS.rl_form,
        'num_steps': FLAGS.num_steps,
        'env_type': type(env).__name__,
    }
    env = envlogger.EnvLogger(env,
                        data_directory = envlogger_dir,
                        metadata = metadata,
                        max_episodes_per_file = 1000)
    return env


def _logger_factory(logger_label: str, steps_key: Optional[str] = None, task_instance: Optional[int]=0) -> base.Logger:
  """logger factory."""
  _EXP_NAME = get_directory_name()
  if logger_label == 'actor':
      terminal_logger = TerminalLogger(label=logger_label, print_fn=logging.info)
      summarydir = os.path.join(FLAGS.summarydir, "rl_logs", _EXP_NAME, logger_label)
      tb_logger = TFSummaryLogger(summarydir, label=logger_label, steps_key=steps_key)
      csv_logger = CSVLogger(summarydir, label=logger_label)
      serialize_fn = base.to_numpy
      logger = aggregators.Dispatcher([terminal_logger, tb_logger, csv_logger], serialize_fn)
      return logger
  elif logger_label == 'learner':
      terminal_logger = TerminalLogger(label=logger_label, print_fn=logging.info)
      summarydir = os.path.join(FLAGS.summarydir,"rl_logs", _EXP_NAME, logger_label)
      tb_logger = TFSummaryLogger(summarydir, label=logger_label, steps_key=steps_key)
      csv_logger = CSVLogger(summarydir, label=logger_label)
      serialize_fn = base.to_numpy
      logger = aggregators.Dispatcher([terminal_logger, tb_logger, csv_logger], serialize_fn)
      # add a timestamp

      return logger
  elif logger_label == 'evaluator':
      terminal_logger = TerminalLogger(label=logger_label, print_fn=logging.info)
      summarydir = os.path.join(FLAGS.summarydir,"rl_logs", _EXP_NAME, logger_label)
      tb_logger = TFSummaryLogger(summarydir, label=logger_label, steps_key=steps_key)
      csv_logger = CSVLogger(summarydir, label=logger_label)
      serialize_fn = base.to_numpy
      logger = aggregators.Dispatcher([terminal_logger, tb_logger, csv_logger], serialize_fn)
      return logger
  else:
    raise ValueError(
        f'Improper value for logger label. Logger_label is {logger_label}')

def build_experiment_config(dimension=None):
    """Builds the experiment configuration."""
    env = AstraSimWrapper.make_astraSim_env(
        knobs_spec=FLAGS.knobs, network=FLAGS.network, system=FLAGS.system, workload=FLAGS.workload_file, 
        rl_form=FLAGS.rl_form,
        max_steps = _NUM_STEPS.value,
        reward_formulation = _REWARD_FORM.value,
        reward_scaling = _REWARD_SCALE.value, 
        congestion_aware = FLAGS.congestion_aware,
        dimension=dimension,
        seed=FLAGS.seed
    )
    if FLAGS.use_envlogger:
        envlogger_dir = os.path.join(FLAGS.summarydir, get_directory_name(), FLAGS.envlogger_dir)
        if(not os.path.exists(envlogger_dir)):
            os.makedirs(envlogger_dir)
        env = wrap_in_envlogger(env, envlogger_dir)

    env_spec = specs.make_environment_spec(env)

    if FLAGS.rl_algo == 'ppo':
        config = ppo.PPOConfig(entropy_cost=FLAGS.entropy_cost,
                            learning_rate=FLAGS.learning_rate,
                            ppo_clipping_epsilon=FLAGS.ppo_clipping_epsilon,
                            clip_value=FLAGS.clip_value,
                            )
        ppo_builder = ppo.PPOBuilder(config)

        if FLAGS.params_scaling > 1:
            size = 32 * FLAGS.params_scaling
            layer_sizes = (size, size, size)
        else:
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

    else:
        raise ValueError(f'Improper value for rl_algo. rl_algo is {FLAGS.rl_algo}')

def main(_):
    knobs_spec = os.path.join(proj_root_path, FLAGS.knobs)
    system_knob, network_knob, workload_knob = astrasim_helpers.parse_knobs_astrasim(knobs_spec)

    dimension_count = 0
    dimensions = []

    network_file = os.path.join(proj_root_path, FLAGS.network)
    action_dict = {}
    action_dict['network'] = astrasim_helpers.parse_network_astrasim(network_file, action_dict, VERSION)

    if VERSION == 1:
        dimensions_count = action_dict['network']["dimensions-count"]
    else:
        dimensions_count = len(action_dict['network']["topology"])

    if "dimensions-count" in network_knob:
        dimensions = sorted(list(network_knob["dimensions-count"][0]))
    else:
        dimensions = [dimensions_count]

    # For loop for dimensions
    for d in dimensions:
        train_singleAgent_helper = os.path.join(proj_root_path, "trainSingleAgentAstraSimHelper.py")

        command = ['python', train_singleAgent_helper, 
                    f'--run_distributed_helper={FLAGS.run_distributed}', 
                    f'--eval_every_helper={FLAGS.eval_every}',
                    f'--eval_episodes_helper={FLAGS.eval_episodes}',
                    f'--dimension={d}',
                    ]

        subprocess.run(command)


if __name__ == '__main__':
   app.run(main)