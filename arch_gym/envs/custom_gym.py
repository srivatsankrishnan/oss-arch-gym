import gym
import numpy as np
from gym import Env, spaces

class ExampleEnv(Env):
    def __init__(self):
        super(ExampleEnv, self).__init__()

        self.energy = 0.0;
        self.area = 0.0;
        self.latency = 0.0;
    
        self.ideal = {'num_cores':4, 'freq':2.0, 'mem_type':'SRAM', 'mem_size':32}

        self.observation_shape = (3,)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float32)

        self.action_space = spaces.Tuple((
            spaces.Discrete(8), # num_cores
            spaces.Box(low=0.5, high=2.5, shape=(1,), dtype=np.float32), # freq
            spaces.Discrete(3), # mem_type
            spaces.Discrete(64) # mem_size
        ))


    def step(self, action):
        num_cores, freq, mem_type, mem_size = action

        self.energy = abs(self.ideal['num_cores'] - num_cores)
        self.area = abs(self.ideal['freq'] - freq)
        self.latency = abs(self.ideal['mem_size'] - mem_size)

        if(action[2] == 1):
            reward = -np.sqrt(self.energy**2 + self.area**2 + self.latency**2)
        else:
            reward = -np.sqrt(self.energy**2 + self.area**2 + self.latency**2) - 10

        observation = [self.energy, self.area, self.latency]

        done = True

        return observation, reward, done, {}

    def reset(self):
        self.energy = 0.0;
        self.area = 0.0;
        self.latency = 0.0;
        return np.array([self.energy, self.area, self.latency], dtype=np.float32)

    def render(self, mode='human'):
        print("Energy: ", self.energy, "Area: ", self.area, "Latency: ", self.latency)