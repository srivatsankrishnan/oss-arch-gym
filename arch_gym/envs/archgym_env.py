import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import math
import time
import sys
import scipy

import os
import datetime

class ArchGymEnv(gym.Env):
    def __init__(self):
        # Will vary based on the comp arch simulator used.
        NUM_INPUT = 1000
        self.observation_space = spaces.Box(low=-100000, high=1000000, shape=(1,NUM_INPUT))
        self.action_space = spaces.Discrete(25)
        self.goal = 0
        self.stepN = 0
        self.episodeN = 0



    def getGoal(self):

        raise NotImplementedError

    def computeReward(self):

        raise NotImplementedError

    def _step(self,action):

        raise NotImplementedError

    def _reset(self):
        # determining the conditions for reset?

        raise NotImplementedError







