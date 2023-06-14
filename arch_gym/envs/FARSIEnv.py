import os
import sys
from tkinter import N

settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
os.sys.path.insert(0, settings_dir_path)

os.sys.path.insert(0, settings_dir_path + '/../../')
os.sys.path.insert(0, settings_dir_path + '/../../')
os.sys.path.insert(0, settings_dir_path + '/../../Project_FARSI')
os.sys.path.insert(0, settings_dir_path + '/../../Project_FARSI/data_collection/collection_utils')


import _pickle as cPickle
from Project_FARSI import *
import home_settings
from top.main_FARSI import run_FARSI_only_simulation
from top.main_FARSI import run_FARSI
from top.main_FARSI import run_FARSI
from design_utils.components.hardware import *
from top.main_FARSI import set_up_FARSI_with_arch_gym
from settings import config
from design_utils.des_handler import move
from  design_utils.design import *
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

class FARSISimEnv(gym.Env):
    def __init__(self,
                reward_formulation = "power",
                workload = 'audio_decoder',
                rl_form = None,
                rl_algo = None,
                max_steps = 1,
                num_agents = 1,
                reward_scaling = 'false'):
        # Todo: Change the values if we normalize the observation space
        if rl_form is not None:
            if(rl_form != 'sa'): 
                print("Only SA currently implemented for FARSI!")
                exit()
            if workload == "audio_decoder":
                n_dim = 54
            elif workload == "hpvm_cava":
                n_dim = 36
            elif workload == "edge_detection":
                n_dim = 34
            self.observation_space = gym.spaces.Box(low=0, high=1e3, shape=(1,3))
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(n_dim,))
        else:  # leaving default prior to RL impl to make sure nothing breaks
            self.observation_space = gym.spaces.Box(low=0, high=1e3, shape=(1,3))
            self.action_space = gym.spaces.Box(low=0, high=8, shape=(10,))
        self.binary_name = arch_gym_configs.binary_name
        self.exe_path = arch_gym_configs.exe_path
        self.sim_config = arch_gym_configs.sim_config
        self.experiment_name = arch_gym_configs.experiment_name
        self.logdir = arch_gym_configs.logdir

        # RL params
        self.rl_form = rl_form
        self.rl_algo = rl_algo
        self.num_agents = num_agents
        self.reward_scaling = reward_scaling

        self.reward_formulation = reward_formulation
        self.workload = workload
        self.max_steps = max_steps
        self.steps = 0
        self.max_episode_len = 10
        self.episode = 0
        self.reward_cap = sys.float_info.epsilon
        self.set_env()
        self.helpers = helpers()
        self.reset()

    # might need to be replaced
    def set_env(self):
        # setting up parammeters
        
        # Select workload of possible ones in FARSI
        # workloads = {"hpvm_cava"}
        # workloads = {"edge_detection"}
        # workloads = {"audio_decoder"}
        workloads = {self.workload}

        SA_depth = 1 # don't touch this. This is a variable that is built into the simulated annealing. If set to 1,
                     # it provides a simple sampling of the environmnet
        base_budget_scaling = {"latency": 1, "power": 1, "area": 1}

        # database parameters (not comprehensive. Also distruted across some csv files)
        freq_range = [1, 4, 6, 8]
        config.transformation_selection_mode = "random"
        config.SA_depth = SA_depth
        ip_loop_unrolling = {"incr": 2, "max_spawn_ip": 17, "spawn_mode": "geometric"}
        ip_freq_range = freq_range
        mem_freq_range = freq_range
        ic_freq_range = freq_range
        tech_node_SF = {"perf":1, "energy":{"non_gpp":.064, "gpp":1}, "area":{"non_mem":.0374 , "mem":.07, "gpp":1}}   # technology node scaling factor
        db_population_misc_knobs = {"ip_freq_correction_ratio": 1, "gpp_freq_correction_ratio": 1,
                                    "ip_spawn": {"ip_loop_unrolling": ip_loop_unrolling, "ip_freq_range": ip_freq_range},
                                    "mem_spawn": {"mem_freq_range":mem_freq_range},
                                    "ic_spawn": {"ic_freq_range":ic_freq_range},
                                    "tech_node_SF":tech_node_SF,
                                    "base_budget_scaling":base_budget_scaling,
                                    "queue_available_size":[1, 2, 4, 8, 16],
                                    "burst_size_options":[1024],
                                    "task_spawn":{"parallel_task_cnt":2, "serial_task_cnt":3}}

        sw_hw_database_population = {"db_mode": "parse", "hw_graph_mode": "generated_from_scratch",
                                     "workloads": workloads, "misc_knobs": db_population_misc_knobs}
        reduction = "most_likely"
        accuracy_percentage = {}
        accuracy_percentage["sram"] = accuracy_percentage["dram"] = accuracy_percentage["ic"] = accuracy_percentage[
            "gpp"] = accuracy_percentage["ip"] = \
            {"latency": 1,
             "energy": 1,
             "area": 1,
             "one_over_area": 1}

        hw_sampling = {"mode": "exact", "population_size": 1, "reduction": reduction,
                       "accuracy_percentage": accuracy_percentage}

        db_input = database_input_class(sw_hw_database_population)
        unique_suffix = "1"  # this doesn't matter
        case_study = "simple_run"
        result_folder = ""
        config.heuristic_type = "simple_greedy_one_sample"
        # run FARSI
        self.dse_hndlr = set_up_FARSI_with_arch_gym(result_folder, unique_suffix, case_study, db_input, hw_sampling,
                              sw_hw_database_population["hw_graph_mode"])
        #return dse_hndlr

        self.cur_ex_dp = self.dse_hndlr.dse.init_ex_dp
        self.cur_sim_dp = self.dse_hndlr.dse.eval_design(self.cur_ex_dp, self.dse_hndlr.dse.database)

    """ 
    def sample_with_FARSI():
        init_ex_dp = self.dse_handler.dse.init_ex_dp

        # exploration does one simple sampling
        self.dse_handler.prepare_for_exploration(boost_SOC, starting_exploration_mode, init_ex_dp)
        self.dse_handler.explore()
    """

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
    
    def calculate_reward(self, sim_dp):

        """
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
        """
        
        #reward = sim_dp.dp_stats.dist_to_goal(self.reward_formulation.split(" "), config.metric_sel_dis_mode)
        reward = sim_dp.dp_stats.dist_to_goal(["power", "area", "latency"], config.metric_sel_dis_mode)
        
        # For RL agent, we want to maximize the reward
        if(arch_gym_configs.rl_agent):
            reward = 1/reward
        
        # some algo (ACO) will throw error if reward is 0. So set it to a very small number
        if (reward == 0):
            reward = 1e-5

        return reward

    def runDRAMEnv(self):
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

    def decode_action_greedy_style(self, action_encoded, cur_ex_dp):
        action = cPickle.loads(action_encoded["action"])

        for blck in cur_ex_dp.get_blocks():
            if blck.instance_name == action.get_block_ref().instance_name:
                selected_block = blck
                break

        selected_krnl = action.get_kernel_ref()
        action_decoded = move(action.transformation_name, action.transformation_sub_name, action.batch_mode, action.get_dir(), action.get_metric(), selected_block, selected_krnl, action.krnel_prob_dict_sorted)

        # set destination block
        if not isinstance(action.get_des_block(), str):
            action_decoded.set_dest_block(action.get_des_block())

        # set destination block
        if not isinstance(action.get_tasks(), str):
            move_tasks = []
            for tsk in cur_ex_dp.get_tasks():
                if tsk.name in [task.name for task in action.get_tasks()]:
                    move_tasks.append(tsk)
            action_decoded.set_tasks(move_tasks)
        action_decoded.set_validity(action.is_valid(), action.validity_meta_data)

        return action_decoded




    def step(self, action_decoded):
        
        self.steps += 1
        if self.rl_form is not None:
            action_decoded = self.actionToConfigs(action_decoded)
            
        #action_decoded = self.action_decode_FARSI(action_encoded)
        pe_allocation = action_decoded["pe_allocation"]
        mem_allocation = action_decoded["mem_allocation"]
        bus_allocation = action_decoded["bus_allocation"]

        pe_to_bus_connection =  action_decoded["pe_to_bus_connection"]
        bus_to_bus_connection = action_decoded["bus_to_bus_connection"]
        bus_to_mem_connection =action_decoded["bus_to_mem_connection"]

        task_to_pe_mapping_list = action_decoded["task_to_pe_mapping"]
        task_to_mem_mapping_list = action_decoded["task_to_mem_mapping"]

        pe_number_name_encoding = action_decoded["pe_number_name_encoding"]
        mem_number_name_encoding = action_decoded["mem_number_name_encoding"]
        ic_number_name_encoding = action_decoded["ic_number_name_encoding"]
        task_number_name_encoding = action_decoded["task_number_name_encoding"]


        system_validity = True
        # instantiate all the hardware blocks
        #-----------------------
        # generate allocation
        #-----------------------
        pes =[]
        for el in pe_allocation:
            if el == -1:
                pes.append(-1)
                continue
            pe_name = pe_number_name_encoding[el]
            pe_ = self.dse_hndlr.database.get_block_by_name(pe_name)
            pe_instance = self.dse_hndlr.database.sample_similar_block(pe_)
            ordered_SOCsL = sorted(self.dse_hndlr.database.SOCsL, key=lambda SOC: SOC.get_budget("latency"))
            pe_instance.set_SOC(ordered_SOCsL[0].type, self.dse_hndlr.database.SOC_id)
            pes.append(pe_instance)

        mems =[]
        for el in mem_allocation:
            if el == -1:
                mems.append(-1)
                continue
            mem_name = mem_number_name_encoding[el]
            mem_ = self.dse_hndlr.database.get_block_by_name(mem_name)
            mem_instance = self.dse_hndlr.database.sample_similar_block(mem_)
            mem_instance= self.dse_hndlr.database.copy_SOC(mem_instance, mem_)
            ordered_SOCsL = sorted(self.dse_hndlr.database.SOCsL, key=lambda SOC: SOC.get_budget("latency"))
            mem_instance.set_SOC(ordered_SOCsL[0].type, self.dse_hndlr.database.SOC_id)
            mems.append(mem_instance)

        ics =[]
        for el in bus_allocation:
            if el == -1:
                ics.append(-1)
                continue
            ic_name = ic_number_name_encoding[el]
            ic_ = self.dse_hndlr.database.get_block_by_name(ic_name)
            ic_instance = self.dse_hndlr.database.sample_similar_block(ic_)
            ordered_SOCsL = sorted(self.dse_hndlr.database.SOCsL, key=lambda SOC: SOC.get_budget("latency"))
            ic_instance.set_SOC(ordered_SOCsL[0].type, self.dse_hndlr.database.SOC_id)
            ics.append(ic_instance)

        #-----------------------
        # generate topology
        #-----------------------
        for idx, connected_bus_idx in enumerate(pe_to_bus_connection):
            connected_bus_idx = int(connected_bus_idx)
            # system checks
            bus_exist = not(ics[connected_bus_idx] == -1)
            pe_exist = not(pes[idx] == -1)
            connection_exist = not(connected_bus_idx == -1)
            if not connection_exist:
                continue
            if (not(bus_exist) or not(pe_exist)) and connection_exist:
                system_validity = False
                break
            elif (not(bus_exist) or not(pe_exist))  and not(connection_exist):
                continue
            pes[idx].connect(ics[connected_bus_idx])

        for idx, connected_bus_idx in enumerate(bus_to_bus_connection):
            connected_bus_idx = int(connected_bus_idx)
            bus_exist = not(ics[idx] == -1)
            connected_bus_exist = not(ics[connected_bus_idx] == -1)
            connection_exist = not(connected_bus_idx == -1)
            if not connection_exist:
                continue
            if (not(bus_exist) or not(connected_bus_exist)) and connection_exist:
                system_validity = False
                break
            elif (not(bus_exist) or not(connected_bus_exist))  and not(connection_exist):
                continue
            ics[idx].connect(ics[connected_bus_idx])

        mems_seen = []
        for idx, connected_mem_idx in enumerate(bus_to_mem_connection):
            connected_mem_idx = int(connected_mem_idx)
            # system checks
            bus_exist = not(ics[idx] == -1)
            mem_exist = not(mems[connected_mem_idx] == -1)
            connection_exist = not(connected_mem_idx == -1)
            if not connection_exist:
                continue
            if (not(bus_exist) or not(mem_exist)) and connection_exist:
                system_validity = False
                break
            elif  (not(bus_exist) or not(mem_exist))  and not(connection_exist):
                continue

            ics[idx].connect(mems[connected_mem_idx])

        # other system checks
        for mem in mems:
            if mem == -1:
                continue
            ics_ = [el for el in mem.get_neighs() if el.type =="ic"]
            if len(ics_) > 1:
                system_validity = False
                break

        found_negative_pe = False
        for pe in pes:
            if not(pe == -1) and found_negative_pe:
                system_validity = False
                break
            if pe == -1:
                found_negative_pe = True

        found_negative_mem = False
        for mem in mems:
            if not(mem== -1) and found_negative_mem:
                system_validity = False
                break
            if mem == -1:
                found_negative_mem = True

        if all([pe == - 1 for pe in pes]):
            system_validity = False
        if all([mem == - 1 for mem in mems]):
            system_validity = False


        if not system_validity:
            obs = [10e3, 10e3, 10e3]
            # some algo (ACO) will throw error if reward is 0. So set it to a very small number
            reward = 10e3
            done = False
            #print("Obs: " + str(obs))
            #return obs, reward, done, {}

        #-----------------------
        # generate mapping
        #-----------------------
        tasks =[]
        for task_idx in range(0, len(task_to_pe_mapping_list)):
            task_name = task_number_name_encoding[task_idx]
            task_instance_ = self.dse_hndlr.database.get_task_by_name(task_name)
            tasks.append(task_instance_)


        for task_idx, pe_mapping in enumerate(task_to_pe_mapping_list):
            pe_mapping=int(pe_mapping)
            if pe_mapping == -1:
                system_validity = False
                break
            pe = pes[pe_mapping]
            #get_work_ratio = self.database.get_block_work_ratio_by_task_dir
            task = tasks[task_idx]

            # check pe compatibility
            compatible_blocks = [el.instance_name_without_id for el in self.dse_hndlr.database.find_all_compatible_blocks("pe", [task])]
            if not pe.instance_name_without_id in compatible_blocks:
                system_validity = False
                break
            pe.load_improved(task, task)

        for task_idx, mem_mapping in enumerate(task_to_mem_mapping_list):
            mem_mapping = int(mem_mapping)
            if mem_mapping == -1:
                system_validity = False
                break
            mem = mems[mem_mapping]
            #get_work_ratio = self.database.get_block_work_ratio_by_task_dir
            task = tasks[task_idx]
            for task_child in task.get_children():
                mem.load_improved(task, task_child)  # load memory with tasks

        if not system_validity:
            obs = [10e3, 103, 103]
            # some algo (ACO) will throw error if reward is 0. So set it to a very small number
            reward = 100
            done = False
            #print("Obs: " + str(obs))
            #return obs, reward, done, {}


        status = True
        try:
            # generate a hardware graph and load read mem and ic
            hardware_graph = HardwareGraph(pes[0])
            new_ex_dp = ExDesignPoint(hardware_graph)
            self.dse_hndlr.dse.dh.load_tasks_to_read_mem_and_ic(new_ex_dp)
            new_ex_dp.hardware_graph.update_graph()
            new_ex_dp.hardware_graph.pipe_design()

        except Exception as e:
            """
            # if the error is already something that we are familiar with
            # react appropriately, otherwise, simply raise it.
            if e.__class__.__name__ in errors_names:
                print("Error: " + e.__class__.__name__)
                # TODOs
                # for now, just return the previous design, but this needs to be fixed immediately
                #raise e
            elif e.__class__.__name__ in exception_names:
                print("Exception: " + e.__class__.__name__)
                new_ex_dp_res = cur_ex_dp_copy
                action.set_validity(False)
            else:
                raise e
            """
            status = False


        if status:
            self.cur_ex_dp = new_ex_dp
            try:
                self.cur_sim_dp = self.dse_hndlr.dse.eval_design(self.cur_ex_dp, self.dse_hndlr.dse.database)
            except Exception as e:
                obs = [10e3, 10e3, 10e3]
                # some algo (ACO) will throw error if reward is 0. So set it to a very small number
                reward = 10e3
                done = False
                print("Obs: " + str(obs))
                self.steps += 1
                if self.rl_form is not None:
                    obs = np.asarray(obs)
                    obs = obs.reshape(1,3)
                #return obs, reward, done, {}
            #get obsv
            metrics_to_look_at = ["latency", "power", "area"]
            metric_value = {}
            for metric_name in metrics_to_look_at:
                metric_value[metric_name] =  self.cur_sim_dp.dp.dp_stats.get_system_complex_metric(metric_name)

            obs = [list(metric_value["latency"].values())[0], metric_value["power"], metric_value["area"] ]
            reward = self.calculate_reward(self.cur_sim_dp)
        else:
            obs = [1000.0, 1000.0, 1000.0]
            # some algo (ACO) will throw error if reward is 0. So set it to a very small number
            reward = -1*10e3
            done = False

        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        done = False

        """
        if(status):
            obs = self.runDRAMEnv()
        else:
            print("Error in writing configs")
        """
        #reward = self.calculate_reward(obs[0][0], obs[0][1])

        if(self.steps == self.max_steps):
            done = True
            print("Maximum steps per episodes reached!")
            self.reset()
            self.episode +=1

        # In case of RL we need to convert obs to np array and reshape to expected 
        if self.rl_form is not None:
            obs = np.asarray(obs)
            obs = obs.reshape(1,3)
            reward = 1.0/reward
        
        print("Episode:", self.episode, " Rewards:", reward)
        print("Obs: " + str(obs))
        #print("Action: " + str(action_decoded))
        return obs, reward, done, {}


    def step_greedy_style(self, action_encoded):
        cur_ex_dp_copy = cPickle.loads(cPickle.dumps(self.cur_ex_dp, -1))
        new_ex_dp = cPickle.loads(cPickle.dumps(self.cur_ex_dp, -1))
        action = self.decode_action(action_encoded, new_ex_dp)
        des_tup = [new_ex_dp, self.cur_sim_dp]
        status = True
        try:
            self.dse_hndlr.dse.dh.unload_read_mem(new_ex_dp)    # unload read memories
            action.validity_check()  # call after unload rad mems, because we need to check the scenarios where
            new_ex_dp_res, _ = self.dse_hndlr.dse.dh.apply_move(des_tup, action)
            action.set_before_after_designs(new_ex_dp, new_ex_dp_res)
            new_ex_dp_res.sanity_check()  # sanity check
            action.sanity_check()
            self.dse_hndlr.dse.dh.load_tasks_to_read_mem_and_ic(new_ex_dp_res)  # loading the tasks on to memory and ic
            new_ex_dp_res.hardware_graph.pipe_design()
            new_ex_dp_res.sanity_check()
            # add something here for status
        except Exception as e:
            # if the error is already something that we are familiar with
            # react appropriately, otherwise, simply raise it.
            if e.__class__.__name__ in errors_names:
                print("Error: " + e.__class__.__name__)
                # TODOs
                # for now, just return the previous design, but this needs to be fixed immediately
                new_ex_dp_res = cur_ex_dp_copy
                #raise e
            elif e.__class__.__name__ in exception_names:
                print("Exception: " + e.__class__.__name__)
                new_ex_dp_res = cur_ex_dp_copy
                action.set_validity(False)
            else:
                raise e
            status = False


        if not (action.get_transformation_name() == "identity") or action.is_valid():
            if status:
                self.cur_ex_dp = new_ex_dp_res
                self.cur_sim_dp = self.dse_hndlr.dse.eval_design(self.cur_ex_dp, self.dse_hndlr.dse.database)


        #get obsv
        metrics_to_look_at = ["latency", "power", "area"]
        metric_value = {}
        for metric_name in metrics_to_look_at:
            metric_value[metric_name] =  self.cur_sim_dp.dp.dp_stats.get_system_complex_metric(metric_name)
        print(metric_value)
        obs = [list(metric_value["latency"].values())[0], metric_value["power"], metric_value["area"] ]

        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        self.steps += 1
        done = False

        """
        if(status):
            obs = self.runDRAMEnv()
        else:
            print("Error in writing configs")
        """
        #reward = self.calculate_reward(obs[0][0], obs[0][1])
        reward = self.calculate_reward(self.cur_sim_dp)

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

    def actionToConfigs(self,action_decoded):

        '''
        Converts actions output from the RL agent to binned actions representation that FARSI can understand.
        '''
        dummy_env = None
        design_space_mode = "limited"  # ["limited", "comprehensive"]
        SOC_design_space = self.helpers.gen_SOC_design_space(dummy_env, design_space_mode, dse=self.dse_hndlr.dse)
        encoding_dictionary = self.helpers.gen_SOC_encoding(dummy_env, SOC_design_space, dse=self.dse_hndlr.dse)
        action_binned = self.helpers.action_mapper_FARSI(action_decoded, encoding_dictionary)            
        action_decoded = self.helpers.action_decoder_FARSI(action_binned, encoding_dictionary)
        return action_decoded
    


# For testing

if __name__ == "__main__":
    
    dramObj = DRAMEnv()
    helpers = helpers()
    logs = []

    obs = dramObj.runDRAMEnv()

    
 
     
    


