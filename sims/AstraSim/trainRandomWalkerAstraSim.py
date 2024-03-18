import json
import time
import random
import pandas as pd
import numpy as np
import envlogger
import os
import sys
import argparse
os.sys.path.insert(0, os.path.abspath('../../'))
os.sys.path.insert(0, os.path.abspath('../../arch_gym'))
from arch_gym.envs import AstraSimWrapper, AstraSimEnv
from arch_gym.envs.envHelpers import helpers
from configs import arch_gym_configs

from absl import flags
from absl import app
from absl import logging

flags.DEFINE_string('workload', 'resnet18', 'Which AstraSim workload to run?')
flags.DEFINE_integer('num_steps', 30, 'Number of training steps.')
flags.DEFINE_integer('num_episodes', 1, 'Number of training episodes.')
flags.DEFINE_bool('use_envlogger', True, 'Use envlogger to log the data.')
flags.DEFINE_string('traject_dir', 'random_walker_trajectories', 'Directory to save the dataset.')
flags.DEFINE_string('summary_dir', "./all_logs/", 'Directory to save the dataset.')
flags.DEFINE_string('reward_formulation', 'cycles', 'Which reward formulation to use?')
flags.DEFINE_string('knobs', 'astrasim_220_example/knobs.py', "path to knobs spec file")
flags.DEFINE_string('network', 'astrasim_220_example/network_input.yml', "path to network input file")
flags.DEFINE_string('system', 'astrasim_220_example/system_input.json', "path to system input file")
flags.DEFINE_string('workload_file', 'astrasim_220_example/workload_cfg.json', "path to workload input file")
flags.DEFINE_bool('congestion_aware', True, "astra-sim congestion aware or not")
# FLAGS.workload_file = astrasim_220_example/workload_cfg.json if GENERATE_WORKLOAD = True
# FLAGS.workload_file = astrasim_220_example/workload-et/generated if GENERATE_WORKLOAD = False

FLAGS = flags.FLAGS

# define AstraSim version
VERSION = 2

def generate_random_actions(action_dict, system_knob, network_knob, workload_knob, dimension):
    print("DIMENSION: ", dimension)
    dicts = [(system_knob, 'system'), (network_knob, 'network'), (workload_knob, 'workload')]

    for dict_type, dict_name in dicts:
        for knob in dict_type:
            if knob == "dimensions-count":
                action_dict[dict_name]["dimensions-count"] = dimension
                continue
            if isinstance(dict_type[knob][0], set):
                if dict_type[knob][1] == "FALSE":
                    list_sorted = sorted(list(dict_type[knob][0]))
                    action_dict[dict_name][knob] = [random.choice(
                        list_sorted) for _ in range(dimension)]
                elif dict_type[knob][1] == "TRUE":
                    list_sorted = sorted(list(dict_type[knob][0]))
                    choice = random.choice(list_sorted)
                    action_dict[dict_name][knob] = [choice for _ in range(dimension)]
                else:
                    list_sorted = sorted(list(dict_type[knob][0]))
                    action_dict[dict_name][knob] = random.choice(list_sorted)
            else:
                if dict_type[knob][1] == "FALSE":
                    action_dict[dict_name][knob] = [random.randint(
                        dict_type[knob][0][0], dict_type[knob][0][1]) for _ in range(dimension)]
                elif dict_type[knob][1] == "TRUE":
                    choice = random.randint(dict_type[knob][0][0], dict_type[knob][0][1])
                    action_dict[dict_name][knob] = [choice for _ in range(dimension)]
                else:
                    action_dict[dict_name][knob] = random.randint(
                        dict_type[knob][0][0], dict_type[knob][0][1])

    return action_dict


def log_results_to_csv(filename, fitness_dict):
    # timestamp: date_hour_min_sec
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")

    df = pd.DataFrame([fitness_dict['reward']])
    df.insert(0, 'timestamp', timestamp)
    csvfile = os.path.join(filename, "rewards.csv")
    df.to_csv(csvfile, index=False, header=False, mode='a')

    df = pd.DataFrame([fitness_dict['action']])
    df.insert(0, 'timestamp', timestamp)
    csvfile = os.path.join(filename, "actions.csv")
    df.to_csv(csvfile, index=False, header=False, mode='a')

    df = pd.DataFrame([fitness_dict['obs']])
    df.insert(0, 'timestamp', timestamp)
    csvfile = os.path.join(filename, "observations.csv")
    df.to_csv(csvfile, index=False, header=False, mode='a')


# Random walker then random walker, else use other
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


def main(_):
    print("VERSION: ", VERSION)
    print(FLAGS.knobs)
    print(FLAGS.network)
    print(FLAGS.system)
    print(FLAGS.workload_file)

    settings_file_path = os.path.realpath(__file__)
    settings_dir_path = os.path.dirname(settings_file_path)
    proj_root_path = os.path.abspath(settings_dir_path)
    astrasim_archgym = os.path.join(proj_root_path, "astrasim-archgym")

    archgen_v1_knobs = os.path.join(astrasim_archgym, "dse/archgen_v1_knobs")
    knobs_spec = os.path.join(proj_root_path, FLAGS.knobs)
    networks_folder = os.path.join(archgen_v1_knobs, "templates/network")
    systems_folder = os.path.join(astrasim_archgym, "themis/inputs/system")
    workloads_folder = os.path.join(astrasim_archgym, "themis/inputs/workload")

    astrasim_helper = helpers()

    # parse knobs
    system_knob, network_knob, workload_knob = astrasim_helper.parse_knobs_astrasim(knobs_spec)
    if workload_knob == {}:
        GENERATE_WORKLOAD = "FALSE"
    else:
        GENERATE_WORKLOAD = "TRUE"

    # DEFINE NETWORK AND SYSTEM AND WORKLOAD
    if VERSION == 1:
        network_file = os.path.join(networks_folder, "4d_ring_fc_ring_switch.json")
        system_file = os.path.join(
            systems_folder, "4d_ring_fc_ring_switch_baseline.txt")
        workload_file = os.path.join(workloads_folder, "all_reduce/allreduce_0.65.txt")
    else:
        network_file = os.path.join(proj_root_path, FLAGS.network)
        system_file = os.path.join(proj_root_path, FLAGS.system)
        workload_file = os.path.join(proj_root_path, FLAGS.workload_file)

    env = AstraSimWrapper.make_astraSim_env(knobs_spec=knobs_spec, network=network_file, system=system_file, max_steps=FLAGS.num_steps,
                                            workload=workload_file, rl_form='random_walker', congestion_aware=FLAGS.congestion_aware)
    # env = AstraSimEnv.AstraSimEnv(rl_form='random_walker')

    # experiment name
    exp_name = str(FLAGS.workload)+"_num_steps_" + str(FLAGS.num_steps) + "_num_episodes_" + str(FLAGS.num_episodes)
    # set exp name to the timestamp
    # exp_name = str(int(time.time()))
    # append logs to base path
    log_path = os.path.join(FLAGS.summary_dir, 'random_walker_logs', FLAGS.reward_formulation, exp_name)
    # get the current working directory and append the exp name
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)
    # check if log_path exists else create it
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if FLAGS.use_envlogger:
        if not os.path.exists(traject_dir):
            os.makedirs(traject_dir)
    env = wrap_in_envlogger(env, traject_dir)

    start = time.time()

    step_results = {}

    # INITIATE action dict
    action_dict = {}

    # parse system, network, and workload
    action_dict['system'] = astrasim_helper.parse_system_astrasim(system_file, action_dict, VERSION)
    action_dict['network'] = astrasim_helper.parse_network_astrasim(network_file, action_dict, VERSION)

    if VERSION == 1:
        dimension = action_dict['network']["dimensions-count"]
    else:
        dimension = len(action_dict['network']["topology"])

    # only generate workload if knobs exist
    if GENERATE_WORKLOAD == "TRUE":
        action_dict['workload'] = astrasim_helper.parse_workload_astrasim(workload_file, action_dict, VERSION)
    else:
        action_dict['workload'] = {"path": workload_file}

    best_reward, best_observation, best_actions = 0.0, 0.0, {}

    for i in range(FLAGS.num_episodes):
        logging.info('Episode %r', i)

        for step in range(FLAGS.num_steps):
            # pass into generate_random_actions(dimension, knobs)
            if "dimensions-count" in network_knob:
                list_sorted = sorted(list(network_knob["dimensions-count"][0]))
                dimension = random.choice(list_sorted)
                action_dict['network']["dimensions-count"] = dimension
            action_dict = generate_random_actions(action_dict, system_knob, network_knob, workload_knob, dimension)
            print(f"{i} {step} DIMENSION: ", dimension)

            # step_result wrapped in TimeStep object
            step_result = env.step(action_dict)
            step_type, reward, discount, observation = step_result

            step_results['reward'] = [reward]
            step_results['action'] = action_dict
            step_results['obs'] = observation

            if reward and reward > best_reward:
                best_reward = reward
                best_observation = observation
                best_actions = action_dict

            log_results_to_csv(log_path, step_results)

    end = time.time()

    print("Best Reward: ", best_reward)
    print("Best Observation: ", best_observation)
    print("Best Parameters: ", best_actions)
    print("Total Time Taken: ", end - start)
    print("Total Useful Steps: ", env.useful_counter)


if __name__ == '__main__':
    app.run(main)
