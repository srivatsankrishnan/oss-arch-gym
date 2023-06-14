import envlogger
from envlogger.testing import catch_env
from envlogger import reader
import numpy as np

import os
import sys

# get full base path
base_path = os.path.dirname(os.path.abspath(__file__))
print(base_path)

# get the directory_path 
log_path = os.path.join(base_path, 'logs')

# check if the path exists and if not create it
if not os.path.exists(log_path):
    os.makedirs(log_path)

env = catch_env.Catch()



with envlogger.EnvLogger(
    env, data_directory=log_path) as env:

  env.reset()
  for step in range(100000):
    action = np.random.randint(low=0, high=3)
    timestep = env.step(action)