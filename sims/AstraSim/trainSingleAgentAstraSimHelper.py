from absl import flags
from absl import app

from trainSingleAgentAstraSim import build_experiment_config

import os
os.sys.path.insert(0, os.path.abspath('../../'))
from configs import arch_gym_configs
from acme.utils import lp_utils
from acme.jax import experiments


flags.DEFINE_bool('run_distributed_helper', False, 'Should an agent be executed in a '
                  'distributed way (the default is a single-threaded agent)')
flags.DEFINE_integer('eval_every_helper', 1, 'Number of evaluation steps.')
flags.DEFINE_integer('eval_episodes_helper', 1, 'Number of evaluation episode.')
flags.DEFINE_integer('dimension', 1, 'dimension value.')


FLAGS = flags.FLAGS

VERSION = 2

def main(_):
    sim_config = arch_gym_configs.sim_config
    config = build_experiment_config(dimension=FLAGS.dimension) #TODO

    if FLAGS.run_distributed_helper:
        program = experiments.make_distributed_experiment(
            experiment=config, num_actors=4)
        lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
    else:
        experiments.run_experiment(
            experiment=config,
            eval_every=FLAGS.eval_every_helper,
            num_eval_episodes=FLAGS.eval_episodes_helper)

if __name__ == '__main__':
    app.run(main)