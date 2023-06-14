import os
import sys

settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
os.sys.path.insert(0, settings_dir_path)

os.sys.path.insert(0, settings_dir_path + '/../../')
os.sys.path.insert(0, settings_dir_path + '/../../sims/DRAM/binary/DRAMSys_Proxy_Model')
print(os.sys.path)
from DRAMSys_Proxy_Model import DRAMSysProxyModel
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
                reward_formulation = "power",
                cost_model = "proxy_model"):
        # Todo: Change the values if we normalize the observation space
        self.observation_space = gym.spaces.Box(low=0, high=1e10, shape=(1,3))
        self.action_space = gym.spaces.Box(low=0, high=8, shape=(10,))
        self.binary_name = arch_gym_configs.binary_name
        self.exe_path = arch_gym_configs.exe_path
        self.sim_config = arch_gym_configs.sim_config
        self.experiment_name = arch_gym_configs.experiment_name
        self.logdir = arch_gym_configs.logdir

        self.cost_model = cost_model

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
        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        self.steps += 1
        done = False

        if self.cost_model == "simulator":
            status = self.actionToConfigs(action_dict)

            if(status):
                obs = self.runDRAMEnv()
            else:
                print("Error in writing configs")
        elif self.cost_model == "proxy_model":

            proxy_model = DRAMSysProxyModel()
            obs = proxy_model.run_proxy_model(action_dict)
        
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
    


# For testing

if __name__ == "__main__":
    
    dramObj = DRAMEnv(cost_model="proxy_model")
    helpers = helpers()
    logs = []

    obs = dramObj.runDRAMEnv()

    
 
     
    


