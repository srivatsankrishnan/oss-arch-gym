import gym
import random
from gym import spaces
import numpy as np

import math


print("Import Successful!")

class RandomParameterEnv(gym.Env):
    def __init__(self, natural=False):
        """
        Initialize environment
        """

        # Parameters to control
        self.action_space = spaces.Box(-5.0, 0.0, shape=(2,))  # parameter 1, parameter2
      

        # observation features. Two randomly generated parameters
        self.observation_space = spaces.Box(-1e5, 1e5, shape=(1,2))  # validation accuracy
        
        self.state = np.random.random_sample(size=self.observation_space.shape)
        
        self.stepN = 0
        self.episode = 0
        self.cum_reward = 0
        self.maxStepN = 1000
        self.steps = 0
        # Start the first game
        self.reset()

    def step(self, action):
        """
        Step function for the environment
        """

        self.stepN = self.stepN + 1
        self.steps = self.steps + 1
        p1, p2 = action

        state = self.random_walk(p1,p2)
        reward = self.compute_reward()
        self.cum_reward = self.cum_reward+reward

        if (state[0][0]<=500 and state[0][1] <= 400):
            done = True
        else:
            done = False

        if(self.steps == self.maxStepN):
            self.steps = 0
            state = self.reset()
            self.episode = self.episode + 1
        info = {"step":self.stepN, "reward":reward, "episode":self.episode}
        
        return state, reward, done, info

    def reset(self):
        self.state =  np.zeros(shape=self.state.shape)

        return self.state

    def compute_reward(self):

        return np.random.random()

    def random_walk(self,p1, p2):

        new_state = np.array([np.random.random()*p1*np.random.random()*p2, np.random.random()*p1 + np.random.random()*p2])
        return np.reshape(new_state,(1,2))

    def render(self):
        print("Step:\t"+ str(self.stepN)+" \t Reward:\t"+str(self.cum_reward//self.stepN))


        






