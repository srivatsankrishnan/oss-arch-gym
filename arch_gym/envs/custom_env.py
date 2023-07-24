import gym
from gym import spaces
import numpy as np
from absl import flags
import os



class CustomEnv(gym.Env):
    def __init__(self, max_steps=10):
        super(CustomEnv, self).__init__()
        self.observation_space = spaces.Dict({"energy": spaces.Box(0, 1, (1,)), "area": spaces.Box(0, 1, (1,)), "latency": spaces.Box(0, 1, (1,))})
        
        self.action_space = spaces.Dict({"num_cores": spaces.Discrete(15), "freq": spaces.Box(low = 0.5, high = 3, dtype = float),
        "mem_type": spaces.Discrete(3), "mem_size": spaces.Discrete(65)})
        # mem_type_list = ['DRAM', 'SRAM', 'Hybrid']

        self.max_steps = max_steps
        self.counter = 0
        self.energy = 0
        self.area = 0
        self.latency = 0
        self.initial_state = np.array([self.energy, self.area, self.latency])
        self.observation = None
        self.done = False


    def reset(self):
        return self.initial_state

    def step(self, action):
        num_cores = action['num_cores']
        freq = action['freq']
        mem_type = action['mem_type']
        mem_size = action['mem_size'] 

        action = np.array([num_cores, freq, mem_type, mem_size])

        if (self.counter == self.max_steps):
            self.done = True
            print("Maximum steps reached")
            self.reset()
        else:
            self.counter += 1
        
        self.energy += num_cores*1 + freq*2 + mem_size*3
        self.area += num_cores*2 + freq*3 + mem_size*1
        self.latency += num_cores*3 + freq*3 + mem_size*1

        observation = np.array([self.energy, self.area, self.latency])
        ideal_values = np.array([4, 2.0, 1, 32])
        print("custom env", observation)
        
        self.observation = observation
        # print(action)
        # print(ideal_values)
        reward = -np.linalg.norm(action - ideal_values)

        print("custom env rew", reward)
       
        return observation, reward, self.done, {}

    def render(self, mode='human'):
        print (f'Energy: {self.energy}, Area: {self.area}, Latency: {self.latency}')
    # def main(self):
        
    #     # observation = self.reset()
    #     # print("The initial observation is {}".format(observation))
    #     action = self.action_space.sample()
    #     print("The taken action is {}".format(action))
    #     obs, reward, done, info = self.step(action)
    #     print("The outcome is {}".format(reward))