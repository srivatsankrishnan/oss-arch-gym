import gym
from gym.utils import seeding
import numpy as np


class simpleEnv(gym.Env):
  LF_MIN = 1
  RT_MAX = 10

  move_left  = 0
  move_right = 1

  max_steps = 10

  reward_away = -2
  reward_step = -1
  reward_goal = 10

  metadata = {
  "render.modes": ["human"]
  }


  def __init__(self):
    self.action_space = gym.spaces.Discrete(2)
    self.observation_space = gym.spaces.Discrete(self.RT_MAX + 1)

    self.goal = int((self.LF_MIN + self.RT_MAX)/2)
    self.init_positions = list(range(self.LF_MIN, self.RT_MAX))
    self.init_positions.remove(self.goal)

    self.seed()

    self.reset()

  def reset(self):
    self.position = np.random.choice(self.init_positions)
    
    self.count = 0

    self.state = self.position
    self.reward = 0
    self.done = False
    self.info = {}

    return self.state

  def step(self, action):
    if self.done:
      print("Episode Done!")
    elif (self.count == self.max_steps):
      self.done = True
    else:
      self.count += 1
    
    if(action == self.move_left):
      if(self.position == self.LF_MIN):
        self.reward = self.reward_away
      else:
        self.position -= 1

      if(self.position == self.goal):
        self.reward = self.reward_goal
        self.done = True
      elif(self.position < self.goal):
        self.reward = self.reward_away
      else:
        self.reward = self.reward_step

    if(action == self.move_right):
      if(self.position == self.RT_MAX):
        self.reward = self.reward_away
      else:
        self.position += 1

      if(self.position == self.goal):
        self.reward = self.reward_goal
        self.done = True
      elif(self.position > self.goal):
        self.reward = self.reward_away
      else:
        self.reward = self.reward_step

    self.state = self.position
    self.info["dist"] = self.goal - self.position

    try:
      assert self.observation_space.contains(self.state)
    except AssertionError:
      print("INVALID STATE",self.state)

    return [self.state, self.reward, self.done, self.info]

    def render(self):
      s = "position: {:2d} reward: {:2d} info: {}"
      print(s.format(self.position, self.reward, self.info))

    def seed(self):
      self.np_random, seed = seeding.np_random(seed)
      return[seed]

    def close(self):
      pass





