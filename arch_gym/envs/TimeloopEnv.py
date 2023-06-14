#!/usr/bin/env python3
from audioop import mul
import multiprocessing
from sims.Timeloop import simulate_timeloop, process_params
from configs import arch_gym_configs
from envHelpers import helpers
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


MAX_EPISODE_LENGTH = 10
MAX_STEPS = 100


class TimeloopEnv(gym.Env):
    def __init__(self, script_dir=None, output_dir=None, arch_dir=None,
                 mapper_dir=None, workload_dir=None, target_val=None,
                 num_cores=None, reward_formulation=None):

        param_obj = process_params.TimeloopConfigParams(arch_gym_configs.timeloop_parameters)
        param_sizes = param_obj.get_param_size()

        # print(param_sizes)

        self.action_space = gym.spaces.MultiDiscrete(param_sizes)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1e10, shape=(3,))

        self.steps = 0
        self.episode = 0
        self.max_steps = MAX_STEPS
        self.max_episode_len = MAX_EPISODE_LENGTH

        self.timeloop_script = script_dir
        self.timeloop_output = output_dir
        self.timeloop_arch = arch_dir
        self.timeloop_mapper = mapper_dir
        self.timeloop_workload = workload_dir
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
        # Assumes that the action here is the modified architecture parameters for now
        self.steps += 1
        obs = self.run_timeloop(action_params)
        done = True
        reward = self.calculate_reward(obs)
        self.cumulative_reward += reward

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

        if obs[0] == -1.0 or obs[1] == -1.0 or obs[2] == -1.0:
            # return a very high number since timeloop failed
            # assumes you can achieve at max 99% improvement on each metric
            return 500*(len(self.target_val))

        reward = 1e-3
        if (self.reward_formulation == 'energy'):
            reward = max(((obs[0] - self.target_val[0])/self.target_val[0]), 0)
        elif (self.reward_formulation == 'area'):
            reward = max(((obs[1] - self.target_val[1])/self.target_val[1]), 0)
        elif (self.reward_formulation == 'latency'):
            reward = max(((obs[2] - self.target_val[2])/self.target_val[2]), 0)
        elif (self.reward_formulation == 'joint'):
            reward += max(((obs[0] - self.target_val[0])/self.target_val[0]), 0)
            reward += max(((obs[1] - self.target_val[1])/self.target_val[1]), 0)
            reward += max(((obs[2] - self.target_val[2])/self.target_val[2]), 0)

        # some algo (ACO) will throw error if reward is 0. So set it to a very small number
        if (reward == 0):
            reward = 1e-5
        return reward
