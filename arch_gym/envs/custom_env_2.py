#from absl import flags, app
import numpy as np
from gym import Env, spaces


class SimpleArch(Env):
    def __init__(self, max_steps=10):
        super(SimpleArch, self).__init__()
        
        #setting initial values
        self.energy = 0.0
        self.area = 0.0
        self.latency = 0.0

        self.max_steps = max_steps
        self.counter = 0
        self.observation = None
        self.done = False
        
        
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
        num_cores = action['num_cores']
        freq = action['freq']
        mem_type = action['mem_type']
        mem_size = action['mem_size'] 

        action = np.array([num_cores, freq, mem_type, mem_size])

        # Compute the new state based on the action
        # these state values may be calculated using any random formulae for now
        self.energy += 1
        self.area += 1
        self.latency += 1


        # Compute the negative of Euclidean distance as the reward
        reward = -np.linalg.norm(self.ideal - action)

        if (self.counter >= self.max_steps):
            self.done = True
            print("Maximum steps reached")
            self.reset()
        else:
            self.counter += 1


        # Update the observation
        observation = np.array([self.energy, self.area, self.latency])

        # Return the new observation, reward, done flag, and additional information (empty dict in this case)
        return observation, reward, self.done, {}

    def render(self, mode='human'):
        print (f'Energy: {self.energy}, Area: {self.area}, Latency: {self.latency}')



#def main(_):
    #env = SimpleArch()
    #env.reset()
    #action = env.action_space.sample()
    #print (f'Action: {action}')
    #env.render()
    #obs, reward, done, info = env.step(action)
    #print(reward)

#if __name__ == '__main__':
    #app.run(main)