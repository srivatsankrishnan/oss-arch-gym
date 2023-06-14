#!/usr/bin/env python3
from audioop import mul
import multiprocessing
from sims.Timeloop import simulate_timeloop, process_params
from configs import arch_gym_configs
from envHelpers import helpers
import pandas as pd
import math
import time
import os

import gym
import numpy as np

settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
os.sys.path.insert(0, settings_dir_path)
os.sys.path.insert(0, settings_dir_path + '/../../configs')

os.sys.path.insert(0, settings_dir_path + '/../../sims/Timeloop')



class TimeloopEnv(gym.Env):
    def __init__(self, script_dir=None, output_dir=None, arch_dir=None,
                 mapper_dir=None, workload_dir=None, target_val=None,
                 num_cores=None, reward_formulation='latency',
                 rl_form='sa', reward_scaling = 'False',
                 max_steps = 100, traj_dir=None):

        param_obj = process_params.TimeloopConfigParams(arch_gym_configs.timeloop_parameters)
        self.param_sizes = param_obj.get_param_size()

        self.rl_form = rl_form
        self.reward_scaling = reward_scaling
        
        if self.rl_form == 'sa':
            self.action_space = gym.spaces.Box(low=-1, high=1, shape = (len(self.param_sizes),))
            self.observation_space = gym.spaces.Box(low=-1, high=1e3, shape=(3,))
        else:
            print("Invalid RL Formulation")
            self.action_space = gym.spaces.Box(low=-1, high=1, shape = (len(self.param_sizes),))
            self.observation_space = gym.spaces.Box(low=-1, high=1e10, shape=(3,))
            #exit(1)
        self.steps = 0
        self.episode = 0
        self.max_steps = max_steps
        
        self.timeloop_script = script_dir
        self.timeloop_output = output_dir
        self.timeloop_arch = arch_dir
        self.timeloop_mapper = mapper_dir
        self.timeloop_workload = workload_dir
        self.traj_dir = os.path.join(traj_dir, 'trajectory.csv')
        self.target_val = target_val
        self.cores = num_cores
        self.reward_formulation = reward_formulation

        print("Reward formulation: ", self.reward_formulation)
        
        if script_dir is None:
            self.timeloop_script = arch_gym_configs.timeloop_scriptdir
        if output_dir is None:
            self.timeloop_output = arch_gym_configs.timeloop_outputdir
        if arch_dir is None:
            self.timeloop_arch = arch_gym_configs.timeloop_archdir
        if mapper_dir is None:
            self.timeloop_mapper = arch_gym_configs.timeloop_mapperdir
        if workload_dir is None:
            self.timeloop_workload = arch_gym_configs.timeloop_workloaddir
        if target_val is None:
            self.target_val = np.array([arch_gym_configs.target_energy,
                                        arch_gym_configs.target_area,
                                        arch_gym_configs.target_cycles])
        if num_cores is None:
            self.cores = int(arch_gym_configs.timeloop_numcores)
        

        self.cores = self.cores//8    # 8 threads per timeloop run

        self.cumulative_reward = 0

        self.helpers = helpers()

        # Batch mode directories
        self.timeloop_script_batch = []
        self.timeloop_output_batch = []
        self.timeloop_arch_batch = []

    def step(self, action_params):
        '''Take an action in a timestep'''
        
        fitness_hist = {}
        # Assumes that the action here is the modified architecture parameters for now
        self.steps += 1
        
        # bin the actions
        action_binned = self.helpers.action_mapper_timeloop(action_params, self.param_sizes)
        # decode the binned actions to dictionary
        action_params = self.helpers.decode_timeloop_action(action_binned)
        obs = self.run_timeloop(action_params)
        done = True
        reward = self.calculate_reward(obs)
        self.cumulative_reward += reward

        
        # For Rl take the inverse since we want to maximize the reward (inverse of the objective)
        if self.rl_form is not None:
            if reward == 1500:
                reward = -1*reward
            else:
                # log the reward
                fitness_hist['reward'] = reward
                fitness_hist['obs'] = obs
                fitness_hist['action'] = action_params
                
                # normalize the observation space for RL (To avoid exploding gradients)
                obs[0] = obs[0]/1e3
                obs[1] = obs[1]   # Don't normalize area
                obs[2] = obs[2]/1e6

                reward = 1/reward

        self.log_fitness_to_csv(fitness_hist)        
        return obs, reward, done, {}

    def step_multiagent(self, action_params):
        '''Take one action for multiple agents in each timestep'''
        self.steps += 1

        # create copies of all the directories
        for agent_ids in range(len(action_params)):
            s, o, a = self.helpers.create_timeloop_dirs(
                agent_ids, self.timeloop_script, self.timeloop_output, self.timeloop_arch)

            self.timeloop_script_batch.append(s)
            self.timeloop_output_batch.append(o)
            self.timeloop_arch_batch.append(a)

        obs_batch = self.run_timeloop_batch(action_params)

        # Calculate the reward for each agent
        rewards = []
        for obs in obs_batch:
            rewards.append(self.calculate_reward(obs))
        print(obs)
        print(rewards)


        done = True
        return obs_batch, rewards, done, {}

    def reset(self):
        '''Reset the environment and associated variables'''
        print("Resetting Environment")

        # All unused?
        self.steps = 0
        self.cumulative_reward = 0
        obs = self.observation_space.sample()

        return obs

    def reset_multiagent(self):
        '''Resets the multi-agent environment and associated variables'''
        self.helpers.remove_dirs(self.timeloop_script_batch)
        self.helpers.remove_dirs(self.timeloop_output_batch)
        self.helpers.remove_dirs(self.timeloop_arch_batch)

        self.timeloop_script_batch = []
        self.timeloop_output_batch = []
        self.timeloop_arch_batch = []

        # All unused?
        self.steps = 0
        self.cumulative_reward = 0
        obs = self.observation_space.sample()

        return obs

    def run_timeloop(self, arch_params):
        '''Invokes the timeloop scripts'''
        energy, area, cycles = simulate_timeloop.simulate_timeloop(self.timeloop_script, self.timeloop_output,
                                                                   self.timeloop_arch, self.timeloop_mapper, self.timeloop_workload, arch_params)

        obs = np.array([energy, area, cycles])

        return obs

    def run_timeloop_batch(self, multi_arch_params):
        '''Invokes the timeloop scripts in batch mode for all agents'''

        obs = []
        pool_params = []

        pool = multiprocessing.Pool(self.cores)

        for agent in range(len(multi_arch_params)):
            params = (self.timeloop_script_batch[agent], self.timeloop_output_batch[agent],
                      self.timeloop_arch_batch[agent], self.timeloop_mapper,
                      self.timeloop_workload, multi_arch_params[agent])
            pool_params.append(params)

        energy, area, cycles = zip(*pool.starmap(simulate_timeloop.simulate_timeloop, pool_params))

        for e, a, c in zip(energy, area, cycles):
            o = np.array([e, a, c])
            obs.append(o)

        return obs

    def calculate_reward(self, obs):
        '''
        Calculates the reward based on the current observation
        '''

        #reward = 1e-3
        if (self.reward_formulation == 'energy'):
            if obs[0] == -1.0 or obs[1] == -1.0 or obs[2] == -1.0:
                reward = 500*(len(self.target_val))
            else:
                reward = max(((obs[0] - self.target_val[0])/self.target_val[0]), 0)
        elif (self.reward_formulation == 'area'):
            if obs[0] == -1.0 or obs[1] == -1.0 or obs[2] == -1.0:
                reward =  500*(len(self.target_val))
            else:
                reward = max(((obs[1] - self.target_val[1])/self.target_val[1]), 0)
        elif (self.reward_formulation == 'latency'):
            if obs[0] == -1.0 or obs[1] == -1.0 or obs[2] == -1.0:
                reward = 500*(len(self.target_val))
            else:
                reward = max(((obs[2] - self.target_val[2])/self.target_val[2]), 0)
        elif (self.reward_formulation == 'joint'):
            if obs[0] == -1.0 or obs[1] == -1.0 or obs[2] == -1.0:
                reward = 500*(len(self.target_val))
            else:
                reward = 0
                reward += max(((obs[0] - self.target_val[0])/self.target_val[0]), 0)
                reward += max(((obs[1] - self.target_val[1])/self.target_val[1]), 0)
                reward += max(((obs[2] - self.target_val[2])/self.target_val[2]), 0)

        # some algo (ACO) will throw error if reward is 0. So set it to a very small number
        if (reward == 0):
            reward = 1e-5
        
        return reward

    def log_fitness_to_csv(self, fitness_dict):
        df = pd.DataFrame([fitness_dict])
        df.to_csv(self.traj_dir, index=False, header=False, mode='a')

        