# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wraps an OpenAI Gym environment to be used as a dm_env environment."""
import sys
from typing import Any, Dict, List, Optional

from acme import specs
from acme import types
from acme import wrappers
import dm_env
import gym
from gym import spaces
import numpy as np
import tree

from DRAMEnv_RL import DRAMEnv
from envHelpers import helpers


class DRAMSysEnvWrapper(dm_env.Environment):
  """Environment wrapper for OpenAI Gym environments."""

  # Note: we don't inherit from base.EnvironmentWrapper because that class
  # assumes that the wrapped environment is a dm_env.Environment.

  def __init__(self, environment: gym.Env,
               env_wrapper_sel: str= 'macme'):

    self._environment = environment
    self._reset_next_step = True
    self._last_info = None
    self.helper = helpers()
    self.env_wrapper_sel = env_wrapper_sel


    # Convert action and observation specs.
    obs_space = self._environment.observation_space
    act_space = self._environment.action_space
    print("obs_space: ", obs_space)
    print("act_space: ", act_space)
    self._observation_spec = _convert_to_spec(obs_space, name='observation')
    self._action_spec = _convert_to_spec(act_space, name='action')

  def reset(self) -> dm_env.TimeStep:
    """Resets the episode."""
    self._reset_next_step = False
    observation = self._environment.reset()
    # Reset the diagnostic information.
    self._last_info = None
    return dm_env.restart(observation)

  
  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    """Steps the environment."""
    if self._reset_next_step:
      return self.reset()
    if(self.env_wrapper_sel=='macme' or self.env_wrapper_sel=='macme_continuous'):
      agents_action  = []
      for each_agent_action in action:
        agents_action.append(each_agent_action.item())
      observation, reward, done, info = self._environment.step(agents_action)
    else:
      observation, reward, done, info = self._environment.step(action)
    self._reset_next_step = done
    self._last_info = info

    # Convert the type of the reward based on the spec, respecting the scalar or
    # array property.
    reward = tree.map_structure(
        lambda x, t: (  # pylint: disable=g-long-lambda
            t.dtype.type(x)
            if np.isscalar(x) else np.asarray(x, dtype=t.dtype)),
        reward,
        self.reward_spec())

    if done:
      truncated = info.get('TimeLimit.truncated', False)
      if truncated:
        return dm_env.truncation(reward, observation)
      return dm_env.termination(reward, observation)
    return dm_env.transition(reward, observation)

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_spec

  def action_spec(self) -> types.NestedSpec:
    return self._action_spec

  def reward_spec(self):
    if self.env_wrapper_sel == 'macme' or self.env_wrapper_sel == 'macme_continuous':
      return [specs.Array(shape=(), dtype=float, name='reward')] * self._environment.num_agents
    else:
      return specs.Array(shape=(), dtype=float, name='reward')

  def get_info(self) -> Optional[Dict[str, Any]]:
    """Returns the last info returned from env.step(action).
    Returns:
      info: dictionary of diagnostic information from the last environment step
    """
    return self._last_info

  @property
  def environment(self) -> gym.Env:
    """Returns the wrapped environment."""
    return self._environment

  def __getattr__(self, name: str):
    if name.startswith('__'):
      raise AttributeError(
          "attempted to get missing private attribute '{}'".format(name))
    return getattr(self._environment, name)

  def close(self):
    self._environment.close()


def _convert_to_spec(space: gym.Space,
                     name: Optional[str] = None) -> types.NestedSpec:
  """Converts an OpenAI Gym space to a dm_env spec or nested structure of specs.
  Box, MultiBinary and MultiDiscrete Gym spaces are converted to BoundedArray
  specs. Discrete OpenAI spaces are converted to DiscreteArray specs. Tuple and
  Dict spaces are recursively converted to tuples and dictionaries of specs.
  Args:
    space: The Gym space to convert.
    name: Optional name to apply to all return spec(s).
  Returns:
    A dm_env spec or nested structure of specs, corresponding to the input
    space.
  """
  if isinstance(space, spaces.Discrete):
    return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

  elif isinstance(space, spaces.Box):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=space.low,
        maximum=space.high,
        name=name)

  elif isinstance(space, spaces.MultiBinary):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=0.0,
        maximum=1.0,
        name=name)

  elif isinstance(space, spaces.MultiDiscrete):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=np.zeros(space.shape),
        maximum=space.nvec - 1,
        name=name)

  elif isinstance(space, spaces.Tuple):
    return tuple(_convert_to_spec(s, name) for s in space.spaces)

  elif isinstance(space, spaces.Dict):
    return {
        key: _convert_to_spec(value, key)
        for key, value in space.spaces.items()
    }
  elif isinstance(space, list):
    return [_convert_to_spec(s, name) for s in space]

  else:
    raise ValueError('Unexpected gym space: {}'.format(space))

def make_dramsys_env(seed: int = 12234,
                    rl_form = 'macme',
                    reward_formulation = 'power',
                    reward_scaling = 'false',
                    max_steps: int = 1,
                    num_agents: int = 10) -> dm_env.Environment:
  """Returns DRAMSys environment."""
  print("[DEBUG][Seed]", seed)
  print("[DEBUG][RL Form]", rl_form)
  print("[DEBUG][Max Steps]", max_steps)
  print("[DEBUG][Num Agents]", num_agents)
  print("[DEBUG][Reward Formulation]", reward_formulation)
  print("[DEBUG][Reward Scaling]", reward_scaling)
  environment = DRAMSysEnvWrapper(
    DRAMEnv(
      rl_form = rl_form,
      max_steps = max_steps,
      num_agents = num_agents,
      reward_formulation = reward_formulation,
      reward_scaling = reward_scaling
    ),
    env_wrapper_sel = rl_form
  )
  environment = wrappers.SinglePrecisionWrapper(environment)
  if(rl_form == 'sa' or rl_form == 'tdm'):
    environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
  return environment
