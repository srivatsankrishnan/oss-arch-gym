import os
import sys

import json
import numpy as np

from absl import flags
from absl import app
from absl import logging
os.sys.path.insert(0, os.path.abspath('../../'))
os.sys.path.insert(0, os.path.abspath('../../arch_gym'))
from sko.GA import GA
from configs import arch_gym_configs
from arch_gym.envs.envHelpers import helpers
from arch_gym.envs import AstraSimWrapper
import envlogger

from configs import arch_gym_configs

import pandas as pd
import matplotlib.pyplot as plt

flags.DEFINE_integer('num_steps', 20, 'Number of training steps.')
flags.DEFINE_integer('num_agents', 4, 'Number of agents.')
flags.DEFINE_float('prob_mutation', 0.1, 'Probability of mutation.')
flags.DEFINE_string('workload','resnet18', 'ML model name')
flags.DEFINE_integer('layer_id', 2, 'Layer id')
flags.DEFINE_string('summary_dir', 'test', 'Directory to save the summary.')
flags.DEFINE_string('reward_formulation', 'latency', 'Reward formulation to use')
flags.DEFINE_string('traject_dir','ga_trajectories', 'Directory to save the dataset.')
flags.DEFINE_bool('use_envlogger', True, 'Whether to use envlogger.')

FLAGS = flags.FLAGS

def generate_run_directories():
    # Construct the exp name from seed and num_iter
    exp_name = FLAGS.workload + "_num_iter_" + str(FLAGS.num_steps) + "_num_agents_" + str(FLAGS.num_agents) + "_prob_mut_" + str(FLAGS.prob_mutation)
  
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)
    
    # log directories for storing exp csvs
    exp_log_dir = os.path.join(FLAGS.summary_dir,"ga_logs",FLAGS.reward_formulation, exp_name)

    return traject_dir, exp_log_dir


def log_fitness_to_csv(filename, fitness_dict):
        df = pd.DataFrame([fitness_dict['reward']])
        csvfile = os.path.join(filename, "fitness.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

        df = pd.DataFrame([fitness_dict['action']])
        csvfile = os.path.join(filename, "actions.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')


def wrap_in_envlogger(env, envlogger_dir):
    metadata = {
        'agent_type': 'RandomWalker',
        'num_steps': FLAGS.num_steps,
        'env_type': type(env).__name__,
    }
    if FLAGS.use_envlogger:
        logging.info('Wrapping environment with EnvironmentLogger...')
        env = envlogger.EnvLogger(env,
                                  data_directory=envlogger_dir,
                                  max_episodes_per_file=1000,
                                  metadata=metadata)
        logging.info('Done wrapping environment with EnvironmentLogger.')
        return env
    else:
        return env

def AstraSim_optimization_function(p):
    settings_file_path = os.path.realpath(__file__)
    settings_dir_path = os.path.dirname(settings_file_path)
    proj_root_path = os.path.abspath(settings_dir_path)

    astrasim_archgym = os.path.join(proj_root_path, "astrasim-archgym")

    # TODO: V1 SPEC:
    archgen_v1_knobs = os.path.join(astrasim_archgym, "dse/archgen_v1_knobs")
    knobs_spec = os.path.join(archgen_v1_knobs, "themis_knobs_spec.py")
    networks_folder = os.path.join(archgen_v1_knobs, "templates/network")
    systems_folder = os.path.join(astrasim_archgym, "themis/inputs/system")
    workloads_folder = os.path.join(astrasim_archgym, "themis/inputs/workload")

    # DEFINE NETWORK AND SYSTEM AND WORKLOAD
    network_file = "4d_ring_fc_ring_switch.json"
    system_file = os.path.join(systems_folder, "4d_ring_fc_ring_switch_baseline.txt")
    workload_file = "all_reduce/allreduce_0.65.txt"

    # parse knobs
    system_knob, network_knob = parse_knobs(knobs_spec)

    
    env = AstraSimWrapper.make_astraSim_env(rl_form='random_walker')
    fitness_hist = {}
    astraSim_helper = helpers()

    traject_dir, exp_log_dir = generate_run_directories()
    
    # check if log_path exists else create it
    if not os.path.exists(exp_log_dir):
        os.makedirs(exp_log_dir)

    if FLAGS.use_envlogger:
        if not os.path.exists(traject_dir):
            os.makedirs(traject_dir)
            
    env = wrap_in_envlogger(env, traject_dir)
    
    # reset the environment
    env.reset()

    # decode the actions

    action_dict = {}
    action_dict['network'] = {"path": network_file}
    action_dict['workload'] = {"path": workload_file}
    
    # parse system
    parse_system(system_file, action_dict)

    action_dict_decoded = astraSim_helper.action_decoder_ga_astraSim(p)
    
    # change all variables decoded into action_dict
    for sect in action_dict_decoded:
        for key in action_dict_decoded[sect]:
            action_dict[sect][key] = action_dict_decoded[sect][key]

    # take a step
    step_type, reward, discount, info = env.step(action_dict)
    
    fitness_dict = {}
    fitness_dict["action"] = p
    fitness_dict["reward"] = reward
    fitness_dict["obs"] = info

    # Convert dictionary to dataframe
    fitness_df = pd.DataFrame([fitness_dict], columns=["action", "reward", "obs"])

    # check if exp_log_dir exists
    if not os.path.exists(exp_log_dir):
        os.makedirs(exp_log_dir)

    # write it to csv file append mode
    fitness_df.to_csv(os.path.join(exp_log_dir, "fitness.csv"), mode='a', header=False, index=False)
    
    
    return -1 * reward


def parse_system(system_file, action_dict):
    # parse system_file (above is the content) into dict
    action_dict['system'] = {}
    with open(system_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            key, value = line.strip().split(': ')
            action_dict['system'][key] = value


# parses knobs that we want to experiment with
def parse_knobs(knobs_spec):
    SYSTEM_KNOBS = {}
    NETWORK_KNOBS = {}

    with open(knobs_spec, 'r') as file:
        file_contents = file.read()
        parsed_dicts = {}

        # Evaluate the file contents and store the dictionaries in the parsed_dicts dictionary
        exec(file_contents, parsed_dicts)

        # Access the dictionaries
        SYSTEM_KNOBS = parsed_dicts['SYSTEM_KNOBS']
        NETWORK_KNOBS = parsed_dicts['NETWORK_KNOBS']
    
    return SYSTEM_KNOBS, NETWORK_KNOBS


def main(_):
    
    
    workload = FLAGS.workload
    layer_id = FLAGS.layer_id

    # encoding format: bounds have same order as modified parameters file
    ga = GA(
        func=AstraSim_optimization_function,
        n_dim=4, 
        size_pop=FLAGS.num_agents,
        max_iter=FLAGS.num_steps,
        prob_mut=FLAGS.prob_mutation,
        lb=[0, 0, 0, 0], 
        ub=[1, 1, 1, 1],
        precision=[1, 1, 1, 1],
    )

    """
    "scheduling-policy": {"FIFO", "LIFO"},
    "collective-optimization": {"localBWAware", "baseline"},
    "intra-dimension-scheduling": {"FIFO", "SCF"},
    "inter-dimension-scheduling": {"baseline", "themis"}
    """
    
    best_x, best_y = ga.run()
    
    
    # get directory names
    _, exp_log_dir = generate_run_directories()
    
    # check if exp_log_dir exists
    if not os.path.exists(exp_log_dir):
        os.makedirs(exp_log_dir)
    
    print(ga.all_history_Y)
    # Convert each array to a list and concatenate the resulting lists
    arr = [a.squeeze().tolist() for a in ga.all_history_Y]

    print(arr)
    
    Y_history = pd.DataFrame(arr)
    Y_history.to_csv(os.path.join(exp_log_dir, "Y_history.csv"))

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.savefig(os.path.join(exp_log_dir, "Y_history.png"))
    
    # save the best_x and best_y to a csv file
    best_x = pd.DataFrame(best_x)
    best_x.to_csv(os.path.join(exp_log_dir, "best_x.csv"))

    best_y = pd.DataFrame(best_y)
    best_y.to_csv(os.path.join(exp_log_dir, "best_y.csv"))


if __name__ == '__main__':
   app.run(main)
