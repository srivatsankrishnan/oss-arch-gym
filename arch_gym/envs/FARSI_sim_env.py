import os
import sys

settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
os.sys.path.insert(0, settings_dir_path)

os.sys.path.insert(0, settings_dir_path + '/../../')
os.sys.path.insert(0, settings_dir_path + '/../../Project_FARSI')
os.sys.path.insert(0, settings_dir_path + '/../../Project_FARSI/data_collection/collection_utils')

from Project_FARSI import *
import home_settings
from top.main_FARSI import run_FARSI_only_simulation
from settings import config
import os
import itertools
# main function
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from visualization_utils.vis_hardware import *
import numpy as np
from specs.LW_cl import *
from specs.database_input import  *
import math
import matplotlib.colors as colors
#import pandas
import matplotlib.colors as mcolors
import pandas as pd
import argparse, sys
import data_collection.collection_utils.what_ifs.FARSI_what_ifs as wf



from configs import arch_gym_configs
import gym
from gym.utils import seeding
from envHelpers import helpers

from loggers import write_csv
import numpy as np

# ToDo: Have a configuration for Arch-Gym to manipulate this methods

import sys

import subprocess
import time
import re
import numpy

import random

class DRAMEnv(gym.Env):
    def __init__(self,
                reward_formulation = "power"):
        # Todo: Change the values if we normalize the observation space
        self.observation_space = gym.spaces.Box(low=0, high=1e10, shape=(1,3))
        self.action_space = gym.spaces.Box(low=0, high=8, shape=(10,))
        self.binary_name = arch_gym_configs.binary_name
        self.exe_path = arch_gym_configs.exe_path
        self.sim_config = arch_gym_configs.sim_config
        self.experiment_name = arch_gym_configs.experiment_name
        self.logdir = arch_gym_configs.logdir

        self.reward_formulation = reward_formulation
        self.max_steps = 100
        self.steps = 0
        self.max_episode_len = 10
        self.episode = 0
        self.reward_cap = sys.float_info.epsilon
        self.helpers = helpers()
        self.reset()

    def get_observation(self,outstream):
        '''
        converts the std out from DRAMSys to observation of energy, power, latency
        [Energy (PJ), Power (mW), Latency (ns)]
        '''
        obs = []

        keywords = ["Total Energy", "Average Power", "Total Time"]

        energy = re.findall(keywords[0],outstream)
        all_lines = outstream.splitlines()
        for each_idx in range(len(all_lines)):

            if keywords[0] in all_lines[each_idx]:
                obs.append(float(all_lines[each_idx].split(":")[1].split()[0])/1e9)
            if keywords[1] in all_lines[each_idx]:
                obs.append(float(all_lines[each_idx].split(":")[1].split()[0])/1e3)
            if keywords[2] in all_lines[each_idx]:
                obs.append(float(all_lines[each_idx].split(":")[1].split()[0])/1e9)

        obs = np.asarray(obs)
        print('[Environment] Observation:', obs)

        if(len(obs)==0):
             print(outstream)
        return obs

    def obs_to_dict(self, obs):
        obs_dict = {}
        obs_dict["Energy"] = obs[0]
        obs_dict["Power"] = obs[1]
        obs_dict["Latency"] = obs[2]

        return obs_dict

    def calculate_reward(self, power, latency):
        target_power = arch_gym_configs.target_power
        target_latency = arch_gym_configs.target_latency
        print("Power:", power, "Latency:", latency, "Target Power:", target_power, "Target Latency:", target_latency)
        #power_norm = max((power - target_power)/target_power, self.reward_cap)
        #latency_norm = max((latency-target_latency)/target_latency, self.reward_cap)
        if self.reward_formulation == "power":
            power_norm = target_power/abs(power-target_power)
            reward = power_norm
        elif self.reward_formulation == "latency":
            latency_norm = target_latency/abs((latency-target_latency))
            reward = latency_norm
        elif self.reward_formulation == "both":
            power_norm = target_power/abs(power-target_power)
            latency_norm = target_latency/abs((latency-target_latency))
            reward = power_norm*latency_norm

        # For RL agent, we want to maximize the reward
        if(arch_gym_configs.rl_agent):
            reward = 1/reward

        return reward

    def runDRAMEnv(self):
        # run simulation
        '''
        Method to launch the DRAM executables given an action
        '''
        exe_path = self.exe_path
        exe_name = self.binary_name
        config_name = self.sim_config
        exe_final = os.path.join(exe_path,exe_name)

        process = subprocess.Popen([exe_final, config_name],stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out, err = process.communicate()
        if err.decode() == "":
            outstream = out.decode()
        else:
            print(err.decode())
            sys.exit()

        obs = self.get_observation(outstream)
        obs = obs.reshape(1,3)

        return obs

    def step(self, action_dict):
        # where does action_dict come from

        # I believe  action_dict is the move
        # in this function, we apply the move (using actionToConfig) do I need to have this function or can I just direction modify
        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        self.steps += 1
        done = False


        status = self.actionToConfigs(action_dict)
        new_des = apply_move(move_)

        if(status):
            obs = self.runDRAMEnv()  # this run the simulation
            sim = eval(new_des)
        else:
            print("Error in writing configs")

        reward = self.calculate_reward(obs[0][1], obs[0][2])

        if(self.steps == 100):
            done = True
            print("Maximum steps per episodes reached!")
            self.reset()
            self.episode +=1

        print("Episode:", self.episode, " Rewards:", reward)
        return obs, reward, done, {}

    def reset(self):
        #print("Reseting Environment!")
        self.steps = 0
        return self.observation_space.sample()

    def actionToConfigs(self,action):

        '''
        Converts actions output from the agent to update the configuration files.
        '''
        write_ok = False

        if(type(action) == dict):
            write_ok = self.helpers.read_modify_write_dramsys(action)
        else:

            action_decoded = self.helpers.action_decoder_rl(action)
            write_ok = self.helpers.read_modify_write_dramsys(action_decoded)
        return write_ok




def run_FARSI_example():
    case_study = "simple_sim_run"
    file_prefix = config.FARSI_simple_sim_run_study_prefix
    current_process_id = 0
    total_process_cnt = 1
    #starting_exploration_mode = config.exploration_mode

    # -------------------------------------------
    # set result folder
    # -------------------------------------------
    result_home_dir_default = os.path.join(os.getcwd(), "data_collection/data/" + case_study)
    result_home_dir = os.path.join(config.home_dir, "data_collection/data/" + case_study)
    date_time = datetime.now().strftime('%m-%d_%H-%M_%S')
    result_folder = os.path.join(result_home_dir,
                                 date_time)

    # -------------------------------------------
    # set parameters
    # -------------------------------------------
    experiment_repetition_cnt = 1
    reduction = "most_likely"
    #workloads = {"audio_decoder", "edge_detection"}
    #workloads = {"audio_decoder"}
    #workloads = {"edge_detection"}
    #workloads = {"hpvm_cava"}
    workloads = {"partial_SOC_example_hard"}
    workloads = {"SOC_example_1p_2r"}
    tech_node_SF = {"perf":1, "energy":{"non_gpp":.064, "gpp":1}, "area":{"non_mem":.0374 , "mem":.079, "gpp":1}}   # technology node scaling factor
    db_population_misc_knobs = {"ip_freq_correction_ratio": 1, "gpp_freq_correction_ratio": 1,
                                "tech_node_SF":tech_node_SF,
                                "base_budget_scaling":{"latency":.5, "power":1, "area":1}}
    sw_hw_database_population = {"db_mode": "parse", "hw_graph_mode": "parse",
                                 "workloads": workloads, "misc_knobs":db_population_misc_knobs}

    # -------------------------------------------
    #  distribute the work
    # -------------------------------------------
    work_per_process = math.ceil(experiment_repetition_cnt / total_process_cnt)
    run_ctr = 0
    # -------------------------------------------
    # run the combination and collect the data
    # -------------------------------------------
    # -------------------------------------------
    # collect the exact hw sampling
    # -------------------------------------------
    accuracy_percentage = {}
    accuracy_percentage["sram"] = accuracy_percentage["dram"] = accuracy_percentage["ic"] = accuracy_percentage["gpp"] = accuracy_percentage[
        "ip"] = \
        {"latency": 1,
         "energy": 1,
         "area": 1,
         "one_over_area": 1}
    hw_sampling = {"mode": "exact", "population_size": 1, "reduction": reduction,
                   "accuracy_percentage": accuracy_percentage}




    db_input = database_input_class(sw_hw_database_population)
    print("hw_sampling:" + str(hw_sampling))
    print("budget set to:" + str(db_input.get_budget_dict("glass")))
    unique_suffix = str(total_process_cnt) + "_" + str(current_process_id) + "_" + str(run_ctr)
    dse_hndlr = run_FARSI_only_simulation(result_folder, unique_suffix, db_input, hw_sampling, sw_hw_database_population["hw_graph_mode"])
    run_ctr += 1

    # get the simulation object stats
    sim_dp = dse_hndlr.dse.so_far_best_sim_dp

    metrics_to_look_at = ["latency", "power", "area"]
    metric_value = {}
    for metric_name in metrics_to_look_at:
        metric_value[metric_name] =  sim_dp.dp.dp_stats.get_system_complex_metric(metric_name)
    obs = [[metrics_to_look_at["latency"], metrics_to_look_at["power"]]]

    """
    # write the results in the general folder
    result_dir_specific = os.path.join(result_folder, "result_summary")
    reason_to_terminate = "simple_sim_run"
    wf.write_one_results(dse_hndlr.dse.so_far_best_sim_dp, dse_hndlr.dse, reason_to_terminate, case_study, result_dir_specific,
                  unique_suffix,
                  file_prefix + "_" + str(current_process_id) + "_" + str(total_process_cnt))

    # write the results in the specific folder
    result_folder_modified = result_folder + "/runs/" + str(run_ctr) + "/"
    os.system("mkdir -p " + result_folder_modified)
    wf.copy_DSE_data(result_folder_modified)
    wf.write_one_results(dse_hndlr.dse.so_far_best_sim_dp, dse_hndlr.dse, reason_to_terminate, case_study,
                  result_folder_modified, unique_suffix,
                  file_prefix + "_" + str(current_process_id) + "_" + str(total_process_cnt))

    """
# For testing

if __name__ == "__main__":
    run_FARSI_example()
    dramObj = DRAMEnv()
    helpers = helpers()
    logs = []

    obs = dramObj.runDRAMEnv()







