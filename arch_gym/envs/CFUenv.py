import sys
import numpy as np

from gym import Env, spaces
import subprocess

class SimpleArch(Env):
    def __init__(self, reward_type, target_vals):
        super(SimpleArch, self).__init__()

        self.rewardType = reward_type
        self.target_val = target_vals

        # define the observation space: cycles, cells
        self.observation_shape = (2,)
        self.observation_space = spaces.Box(low=0, high=sys.maxsize, shape=self.observation_shape, dtype=np.int32)
        
        # define the action space
        # ! Maybe for all the true false ones, multibinary would be better?
        self.action_space = spaces.Tuple((
            spaces.Discrete(2),     
            spaces.Discrete(2),     
            spaces.Discrete(11),
            spaces.Discrete(2),    
            spaces.Discrete(11),
            spaces.Discrete(2),
            spaces.Discrete(4),
            spaces.Discrete(2),
            spaces.Discrete(2),     
            spaces.Discrete(2)
        ))

        #clearing the history file
        open('History', 'w').close()

    # No point in our application to have this function, really
    def reset(self):
        self.energy = 0.0; self.area = 0.0; self.latency = 0.0

    # How many options for csr Plugin config?
    def step(self, action):

        # store input to a text file, to be used by the dse script
        file = open('CFU_log', 'w')
        s = ''
        for a in action:
            s += str(a) + ' '
        file.write(s)
        file.close()
        
        #Store the action into the history file
        file = open('History', 'a')
        file.write(s + '\n')
        file.close()

        # Calling dse function through another script, in the symbiflow environment
        subprocess.run(['. ../../sims/CFU-Playground/env/conda/bin/activate cfu-symbiflow && python dse_run.py && conda activate arch-gym'], shell = True, executable='/bin/bash')

        #Read output from the script
        file = open('CFU_log', 'r')
        output = file.read()
        file.close()

        #Store output, this would need to be more comprehensive later
        file = open('History', 'a')
        file.write(output + '\n')
        file.close()
        
        output = output.split()
        observation = [int(output[0]), int(output[1])]
        reward = self.calculate_reward(observation)
        return observation, reward, True, {}
    
    def calculate_reward(self, obs):
        reward = 1e-3
        if(self.rewardType == 'cells'):
            reward = max(((obs[0] - self.target_val[0])/self.target_val[0]), 0)
        elif (self.reward_formulation == 'cycles'):
            reward = max(((obs[1] - self.target_val[1])/self.target_val[1]), 0)
        # assuming the "algo" gives error when reward is 0. So i'll also give the reward a small value...
        if (reward == 0):
            reward = 1e-5

        return reward

    def render(self, mode='human'):
        print('Hi')