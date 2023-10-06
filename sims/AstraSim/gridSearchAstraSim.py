import os
import sys
import pickle

from absl import app

os.sys.path.insert(0, os.path.abspath('../../'))
os.sys.path.insert(0, os.path.abspath('../../arch_gym'))

from configs import arch_gym_configs

from arch_gym.envs.envHelpers import helpers
from arch_gym.envs import AstraSimWrapper, AstraSimEnv
import envlogger
import numpy as np
import pandas as pd
import random
import time
import json

# systems: parse from file into json into generate_random_actions
"""
system_file content: 
scheduling-policy: LIFO
endpoint-delay: 1
active-chunks-per-dimension: 1
preferred-dataset-splits: 64
boost-mode: 1
all-reduce-implementation: direct_ring_halvingDoubling
all-gather-implementation: direct_ring_halvingDoubling
reduce-scatter-implementation: direct_ring_halvingDoubling
all-to-all-implementation: direct_direct_direct
collective-optimization: localBWAware
intra-dimension-scheduling: FIFO
inter-dimension-scheduling: baseline
"""
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
    

# action_type = specify 'network' or 'system
# new_params = parsed knobs from experiment file
def generate_random_actions(action_dict, system_knob, network_knob, args):
    dicts = [(system_knob, 'system'), (network_knob, 'network')]
    for dict_type, dict_name in dicts:
        i = 0
        for knob in dict_type.keys():
            if isinstance(dict_type[knob], set):
                action_dict[dict_name][knob] = list(dict_type[knob])[args[i]]
                i += 1
    
    return action_dict


def main(_):
    settings_file_path = os.path.realpath(__file__)
    settings_dir_path = os.path.dirname(settings_file_path)
    proj_root_path = os.path.abspath(settings_dir_path)

    astrasim_archgym = os.path.join(proj_root_path, "astrasim-archgym")

    # TODO: V1 SPEC:
    archgen_v1_knobs = os.path.join(astrasim_archgym, "dse/archgen_v1_knobs")
    knobs_spec = os.path.join(archgen_v1_knobs, "archgen_v1_knobs_spec.py")
    networks_folder = os.path.join(archgen_v1_knobs, "templates/network")
    systems_folder = os.path.join(astrasim_archgym, "themis/inputs/system")
    workloads_folder = os.path.join(astrasim_archgym, "themis/inputs/workload")

    # DEFINE NETWORK AND SYSTEM AND WORKLOAD
    network_file = "4d_ring_fc_ring_switch.json"
    system_file = os.path.join(systems_folder, "4d_ring_fc_ring_switch_baseline.txt")
    workload_file = "all_reduce/allreduce_0.65.txt"

    env = AstraSimWrapper.make_astraSim_env(rl_form='random_walker')
    # env = AstraSimEnv.AstraSimEnv(rl_form='random_walker')

    astrasim_helper = helpers()

    start = time.time()

    step_results = {}

    all_results = []
    best_reward, best_observation, best_actions = 0.0, 0.0, {}

    for sp in [0, 1]:
        for co in [0, 1]:
            for intra in [0, 1]:
                for inter in [0, 1]:
                    # INITIATE action dict
                    action_dict = {}
                    args = [sp, co, intra, inter]

                    # if path exists, use path, else parse the sub-dict
                    action_dict['network'] = {"path": network_file}
                    action_dict['workload'] = {"path": workload_file}
                    
                    # TODO: parse system 
                    parse_system(system_file, action_dict)

                    # TODO: parse knobs (all variables to change in action_dict)
                    system_knob, network_knob = parse_knobs(knobs_spec)

                    # pass into generate_random_actions(dimension, knobs)
                    action_dict = generate_random_actions(action_dict, system_knob, network_knob, args)

                    # with open("general_workload.txt", 'w') as file:
                    #     file.write(action["workload"]["value"])

                    # step_result wrapped in TimeStep object
                    step_result = env.step(action_dict)
                    step_type, reward, discount, observation = step_result
                    
                    step_results['reward'] = [reward]
                    step_results['action'] = action_dict
                    step_results['obs'] = observation

                    all_results.append((reward, observation))

                    if reward and reward > best_reward:
                        best_reward = reward
                        best_observation = observation
                        best_actions = action_dict

                    end = time.time()

    print("Best Reward: ", best_reward)
    print("Best Observation: ", best_observation)
    print("Best Parameters: ", best_actions)
    print("All Results: ", all_results)


if __name__ == '__main__':
   app.run(main)
