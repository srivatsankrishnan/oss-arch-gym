import sys
import numpy as np

from gym import Env, spaces
import subprocess
import os

class SimpleArch(Env):
    def __init__(self, reward_type, target_vals):
        super(SimpleArch, self).__init__()

        self.rewardType = reward_type
        self.target_val = target_vals

        # define the observation space: cycles, cells
        self.observation_shape = (2,)
        self.observation_space = spaces.Box(low=0, high=sys.maxsize, shape=self.observation_shape, dtype=np.int32)
        
        # define the action space
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

    # No point in our application to have this function, really
    def reset(self):
        self.energy = 0.0; self.area = 0.0; self.latency = 0.0

    # How many options for csr Plugin config?
    def step(self, action):

        #ensure that relative paths work
        Caller_wd = os.getcwd()        
        os.chdir(os.path.dirname(__file__))

        # store input to a text file, to be used by the dse script
        file = open('CFU_log', 'w')
        s = ''
        for a in action:
            s += str(a) + ' '
        file.write(s)
        file.close()

        #Calling dse function through another script, in the symbiflow environment
        subprocess.run(['. ../../sims/CFU-Playground/env/conda/bin/activate cfu-symbiflow && python CFU_run.py'], shell = True, executable='/bin/bash')

        #Read output from the script
        file = open('CFU_log', 'r')
        output = file.read()
        file.close()

        #Go bath to the original directory
        os.chdir(Caller_wd)

        output = output.split()
        observation = [float(output[0]), int(output[1])]
        reward = self.calculate_reward(observation)
        return observation, reward, True, {}
    
    def calculate_reward(self, obs):

        reward = 1e-3
        if(self.rewardType == 'cells'):
            reward = max(((obs[0] - self.target_val[0])/self.target_val[0]), 0)
        elif (self.reward_formulation == 'cycles'):
            reward = max(((obs[1] - self.target_val[1])/self.target_val[1]), 0)

        # assuming the algo gives error when reward is 0.
        if (reward == 0):
            reward = 1e-5

        return reward

    def render(self, mode='human'):
        print('Hi')