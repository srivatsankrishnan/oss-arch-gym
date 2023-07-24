from absl import flags, app
import gym 
import numpy as np
from gym import Env, spaces


class SimpleArch(Env):
    def __init__(self):
        super(SimpleArch, self).__init__()
        
        self.energy = 0.0; self.area = 0.0; self.latency = 0.0
        
        
        # set ideal architecture parameters
        self.ideal = np.array([4, 2.0, 1, 32])   #the ideal values are those of [num_cores, freq, mem_type, mem_size]

        
        # define the observation space: Energy, Area, Latency
        self.observation_shape = (3,)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=self.observation_shape, dtype=np.float32)
        
        # define the action space
        self.action_space = spaces.Tuple((
            spaces.Discrete(8),           # num_cores
            spaces.Box(low=0.5, high=2.5, shape=(1,), dtype=np.float32),  # freq
            spaces.Discrete(3),           # mem_type  ## 0, 1, 2 would correspond to DRAM, SRAM, Hybrid
            spaces.Discrete(64)           # mem_size
        ))
        
    def reset(self):
        self.energy = 0.0; self.area = 0.0; self.latency = 0.0
        return np.array([self.energy, self.area, self.latency], dtype=np.float32)
    
    def step(self, action):
        # Extract the action values
        num_cores, freq, mem_type, mem_size = action

        # Compute the new state based on the action
        # these state values may be calculated using any random formulae for now
        self.energy += 1
        self.area += 1
        self.latency += 1


        # Compute the negative of Euclidean distance as the reward
        reward = -np.linalg.norm(self.ideal - action)


        # Update the observation
        observation = [self.energy, self.area, self.latency]

        # Set done to True since we have a simple environment with a single episode
        done = True

        # Return the new observation, reward, done flag, and additional information (empty dict in this case)
        return observation, reward, done, {}

    def render(self, mode='human'):
        print (f'Energy: {self.energy}, Area: {self.area}, Latency: {self.latency}')



def main(_):
    env = SimpleArch()
    env.reset()
    action = env.action_space.sample()
    print (f'Action: {action}')
    env.render()
    obs, reward, done, info = env.step(action)
    print(reward)

if __name__ == '__main__':
    app.run(main)