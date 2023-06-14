import os
import sys
import csv

settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
os.sys.path.insert(0, settings_dir_path)

os.sys.path.insert(0, settings_dir_path + '/../../')

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
                 rl_form: str = 'tdm',
                 rl_algo: str = 'ppo',
                 max_steps: int = 1,
                 num_agents: int = 1,
                 reward_formulation: str = 'latency',
                 reward_scaling: str = 'false',
                 ):
        if(rl_form == 'tdm'):
            # observation space  for TDM
            # Simulator output: [Energy (PJ), Power (mW), Latency (ns)]
            # TDM Action = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]
            # Observation to agent at each time step
            # [Energy (PJ), Power (mW), Latency (ns), param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]

            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)
            # Agent is Time division multiplexed. Hence it will take only one of the many discrete actions
            self.action_space = gym.spaces.Discrete(8)
            if (rl_algo == 'sac'):
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        elif(rl_form == 'sa'):
            self.observation_space = gym.spaces.Box(low=0, high=1e3, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(10,))
        elif(rl_form == 'macme'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(3,))] * num_agents
            
            self.action_space = [
                
                # page policy agent
                gym.spaces.Discrete(4),
                
                # scheduler_mapper agent
                gym.spaces.Discrete(3),
                
                # schedulerbuffer_mapper agent
                gym.spaces.Discrete(3),
                
                # request_buffer_size_mapper agent
                gym.spaces.Discrete(8),
                
                # respqueue_mapper agent
                gym.spaces.Discrete(2),
                
                # refreshpolicy_mapper agent
                gym.spaces.Discrete(2),
                
                # refreshmaxpostponed_mapper agent 
                gym.spaces.Discrete(4),

                # refreshmaxpulledin_mapper
                gym.spaces.Discrete(4),

                # arbiter_mapper
                gym.spaces.Discrete(3),

                # max_active_transactions_mapper
                gym.spaces.Discrete(8),
            ]
        elif (rl_form == 'macme_continuous'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(3,))] * num_agents
            self.action_space = [
                # page policy agent
                gym.spaces.Box(low=-1, high=1, shape=(1,)),
                
                # scheduler_mapper agent
                gym.spaces.Box(low=-1, high=1, shape=(1,)),
                
                # schedulerbuffer_mapper agent
                gym.spaces.Box(low=-1, high=1, shape=(1,)),

                # request_buffer_size_mapper agent
                gym.spaces.Box(low=-1, high=1, shape=(1,)),
                
                # respqueue_mapper agent
                gym.spaces.Box(low=-1, high=1, shape=(1,)),

                # refreshpolicy_mapper agent
                gym.spaces.Box(low=-1, high=1, shape=(1,)),
                
                # refreshmaxpostponed_mapper agent
                gym.spaces.Box(low=-1, high=1, shape=(1,)),
                
                # refreshmaxpulledin_mapper agent
                gym.spaces.Box(low=-1, high=1, shape=(1,)),
                
                # arbiter_mapper agent
                gym.spaces.Box(low=-1, high=1, shape=(1,)),
                
                # max_active_transactions_mapper agent
                gym.spaces.Box(low=-1, high=1, shape=(1,)),
            ]
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-0.001, high=0.001, shape=(10,))
        self.rl_form = rl_form
        self.num_agents = num_agents
        self.binary_name = arch_gym_configs.binary_name
        self.exe_path = arch_gym_configs.exe_path
        self.sim_config = arch_gym_configs.sim_config
        self.experiment_name = arch_gym_configs.experiment_name
        self.logdir = arch_gym_configs.logdir

        self.reward_form = reward_formulation
        self.reward_scaling = reward_scaling

        print("[DEBUG][Reward Scaling]", self.reward_scaling)
        self.max_steps = max_steps
        self.steps = 0
        self.max_episode_len = 10
        self.episode = 0
        self.reward_cap = 1e3
        self.helpers = helpers()
        self.algorithm = "GA"
        self.goal_latency = 2e7
        self.prev_latency = 1e9
        self.prev_power = 0
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
                obs.append(float(all_lines[each_idx].split(":")[1].split()[0])/1e6)
        
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

        # Old Reward Calculation
        #power_norm = max((power - target_power)/target_power, self.reward_cap)
        #latency_norm = max((latency-target_latency)/target_latency, self.reward_cap)

        # Izzet's suggested reward
        #power_norm = max(-(power - target_power)/target_power, self.reward_cap)
        #latency_norm = max(-(latency - target_latency)/target_latency, self.reward_cap)

        # product of % increase as reward
        #power_norm = (self.prev_power - power)/self.prev_power
        #latency_norm = (self.prev_latency - latency)/self.prev_latency
        #latency_norm = (self.prev_latency - latency)/latency

        if(self.reward_form == 'power'):
            power_norm = target_power/abs((power-target_power))
            reward = power_norm
            if self.reward_scaling == 'true':
                # 350 seems to be the max reward for power
                reward = reward/350
        elif(self.reward_form == 'latency'):
            latency_norm = target_latency/abs((latency-target_latency))
            reward = latency_norm
            if self.reward_scaling == 'true':
                # 350 seems to be the max reward for latency
                reward = reward/20
        elif(self.reward_form == 'both'):
            power_norm = target_power/abs((power-target_power))
            latency_norm = target_latency/abs((latency-target_latency))
            if self.reward_scaling == 'true':
                # 350 seems to be the max reward for power
                power_norm = power_norm/350
                # 350 seems to be the max reward for latency
                latency_norm = latency_norm/20
            reward = power_norm * latency_norm
        else:
            print("Reward formulation not recognized. Exiting...")
            exit()

        
        if reward > self.reward_cap:
            reward = self.reward_cap

        print("Reward:", reward)
        
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            reward = [np.copy(reward)] * self.num_agents
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
        obs = obs.reshape(3,)
        
        return obs

    def step(self, action_dict):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        self.steps += 1
        done = False
        
        status = self.actionToConfigs(action_dict)

        if(status):
            obs = self.runDRAMEnv()
        else:
            print("Error in writing configs")
        print(obs)
        
        reward = self.calculate_reward(obs[1], obs[2])
        
        if(self.steps == self.max_steps):
            done = True
            print("Maximum steps per episodes reached!")
            self.reset()
            self.episode +=1
        
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            obs = [obs.copy()] * self.num_agents

        print("Episode:", self.episode, " Rewards:", reward)
        return obs, reward, done, {}

    def reset(self):
        #print("Reseting Environment!")
        self.steps = 0
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            return [self.observation_space[0].sample()]* self.num_agents
        else:
            # return zeros of shape of observation space
            return np.zeros(self.observation_space.shape)
            #return self.observation_space.sample()

    def actionToConfigs(self,action):

        '''
        Converts actions output from the agent to update the configuration files.
        '''
        write_ok = False

        if(type(action) == dict):
            write_ok = self.helpers.read_modify_write_dramsys(action)
        else:
            print("[Env][Action]", action)
            action_decoded = self.helpers.action_decoder_rl(action, self.rl_form)
            write_ok = self.helpers.read_modify_write_dramsys(action_decoded)
        return write_ok
    


# For testing

if __name__ == "__main__":
    
    dramObj = DRAMEnv()
    helpers = helpers()
    logs = []

    obs = dramObj.runDRAMEnv()

    
 
     
    


