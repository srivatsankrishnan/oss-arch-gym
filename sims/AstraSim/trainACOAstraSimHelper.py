import os
os.sys.path.insert(0, os.path.abspath('../../'))

from aco.DeepSwarm.deepswarm.backends import Dataset, AstraSimBackend
from aco.DeepSwarm.deepswarm.deepswarm import DeepSwarm
from arch_gym.envs.envHelpers import helpers
from absl import flags
from absl import app
import numpy as np

flags.DEFINE_integer('dimension', 1, 'Dimension Count')
flags.DEFINE_string('reward_formulation', 'cycles', 'Reward formulation to use.')
flags.DEFINE_bool('use_envlogger', True, 'Use EnvLogger to log environment data.')
flags.DEFINE_string('knobs', 'astrasim_220_example/knobs.py', "path to knobs spec file")
flags.DEFINE_string('network', 'astrasim_220_example/network_input.yml', "path to network input file")
flags.DEFINE_string('system', 'astrasim_220_example/system_input.json', "path to system input file")
flags.DEFINE_string('workload_file', 'astrasim_220_example/workload_cfg.json', "path to workload input file")
flags.DEFINE_bool('congestion_aware', True, "astra-sim congestion aware or not")
flags.DEFINE_integer('ant_count', 1, 'Number of Ants.')
flags.DEFINE_float('greediness', 0.5, 'How greedy you want the ants to be?.')
flags.DEFINE_float('decay', 0.1, 'Decay rate for pheromone.')
flags.DEFINE_float('evaporation', 0.25, 'Evaporation value for pheromone.')
flags.DEFINE_float('start', 0.1, 'Start value for pheromone.')
flags.DEFINE_integer('depth', 10, 'Depth of the network.')
flags.DEFINE_string('workload', 'stream.stl', 'Which workload to run')

flags.DEFINE_string('traject_dir', './all_logs/aco_trajectories', 'Directory to trajectory data.')
flags.DEFINE_string('aco_log_dir', './all_logs/aco_logs', 'Directory to store logs.')
flags.DEFINE_string('summary_dir', '.', 'Directory to store summaries.')

FLAGS = flags.FLAGS

VERSION = 2

def main(_):

    # Experiment name (combine ant_count, greediness and evaporation
    exp_name = str(FLAGS.workload)+"_ant_count_" + str(FLAGS.ant_count) + "_greediness_" + str(FLAGS.greediness) + "_evaporation_" + str(FLAGS.evaporation) + "_depth_" + str(FLAGS.depth)

    # create trajectory directory (current dir + traject_dir)
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)

    # create log directory (current dir + log_dir)
    log_dir = os.path.join(FLAGS.summary_dir, FLAGS.aco_log_dir, FLAGS.reward_formulation, exp_name)

    x_train = np.array([1,2,3,4,5])
    dataset = Dataset(training_examples=x_train, training_labels=None, testing_examples=x_train, testing_labels=None)

    backend = AstraSimBackend(dataset, 
                                exp_name=exp_name,
                                log_dir=log_dir,
                                traject_dir=traject_dir,
                                reward_formulation= FLAGS.reward_formulation,
                                use_envlogger=FLAGS.use_envlogger,
                                VERSION=VERSION,
                                knobs_spec=FLAGS.knobs,
                                network=FLAGS.network,
                                system=FLAGS.system,
                                workload=FLAGS.workload_file,
                                congestion_aware=FLAGS.congestion_aware,
                                dimension=FLAGS.dimension,
                                astrasim_ant_count=FLAGS.ant_count, 
                                astrasim_greediness = FLAGS.greediness, 
                                astrasim_decay = FLAGS.decay, 
                                astrasim_evaporation = FLAGS.evaporation, 
                                astrasim_start = FLAGS.start, 
                                astrasim_max_depth = FLAGS.depth,                             
                                )
    deepswarm = DeepSwarm(backend=backend)

    topology = deepswarm.find_topology()

if __name__ == '__main__':
   app.run(main)