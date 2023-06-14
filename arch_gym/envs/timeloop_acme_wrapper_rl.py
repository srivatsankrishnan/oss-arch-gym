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

from typing import Any, Dict, List, Optional

from acme import specs
from acme import types
from acme import wrappers
import dm_env
import gym
from gym import spaces
import numpy as np
import tree

from TimeloopEnv_RL import TimeloopEnv
from envHelpers import helpers


class TimeloopEnvWrapper(dm_env.Environment):
    """Environment wrapper for OpenAI Gym environments."""

    # Note: we don't inherit from base.EnvironmentWrapper because that class
    # assumes that the wrapped environment is a dm_env.Environment.

    def __init__(self,
                 environment: gym.Env,
                 maximize_action: bool = False,
                 convert_multi_discrete: bool = True,
                 multi_agent: bool = False):

        self._environment = environment
        self._reset_next_step = True
        self._last_info = None
        self.maximize_action = maximize_action
        self.convert_multi_discrete = convert_multi_discrete
        self.multi_agent = multi_agent
        self.helper = helpers()

        # Convert action and observation specs.
        obs_space = self._environment.observation_space
        act_space = self._environment.action_space
        self._observation_spec = self._convert_to_spec(
            obs_space, name='observation')
        # self._action_spec = self._convert_to_spec(act_space, name='action')
        # self._action_spec_clipped = self._convert_spec(self._action_spec)
        self._action_spec = self._convert_spec(
            self._convert_to_spec(act_space, name='action'))
        self._action_spec_unclipped = self._convert_to_spec(
            act_space, name='action')

        # print(self._action_spec_clipped)

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        self._reset_next_step = False

        if self.multi_agent:
            observation = self._environment.reset_multiagent()
        else:
            observation = self._environment.reset()
        # Reset the diagnostic information.
        self._last_info = None
        return dm_env.restart(observation)

    def _scale_nested_action(self,
                             nested_action: types.NestedArray,
                             nested_spec: types.NestedSpec
                             ) -> types.NestedArray:
        """Converts a canonical nested action back to the given nested action spec."""

        def _scale_action(action: np.ndarray, spec: specs.Array):
            """Converts a single canonical action back to the given action spec."""

            if isinstance(spec, specs.BoundedArray):

                # Get scale and offset of output action spec.
                scale = spec.maximum - spec.minimum
                offset = spec.minimum

                # lip the action.
                action = np.clip(action, -1.0, 1.0)

                # Map action to [0, 1].
                action = 0.5 * (action + 1.0)

                # Map action to [spec.minimum, spec.maximum].
                action *= scale
                action += offset

                return action

        return tree.map_structure(_scale_action, nested_action, nested_spec)

    def step(self, action: types.NestedArray) -> dm_env.TimeStep:
        """Steps the environment."""

        if self._reset_next_step:
            return self.reset()

        if self.multi_agent:
            action_dict = np.array(
                [self.helper.decode_timeloop_action(i) for i in action])

            observation, reward, done, info = self._environment.step_multiagent(
                action_dict)
            
            reward = np.array(reward)
            
        else:
            
            #action_dict = self.helper.decode_timeloop_action(action)

            observation, reward, done, info = self._environment.step(
                action)

        if self.maximize_action:
            reward = -reward

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

    def _convert_spec(self, nested_spec: types.NestedSpec) -> types.NestedSpec:
        """Converts all bounded specs in nested spec to the canonical scale."""

        def _convert_single_spec(spec: specs.Array) -> specs.Array:
            """Converts a single spec to canonical if bounded."""
            if isinstance(spec, specs.BoundedArray):
                return spec.replace(
                    minimum=-np.ones(spec.shape), maximum=np.ones(spec.shape))
            else:
                return spec

        return tree.map_structure(_convert_single_spec, nested_spec)

    def _convert_to_spec(self, space: gym.Space,
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
            if self.convert_multi_discrete:
                # Convert multi-discrete type to float array.
                return specs.BoundedArray(
                    shape=space.shape,
                    dtype=np.float,
                    minimum=np.zeros(space.shape),
                    maximum=space.nvec - 1,
                    name=name)
            else:
                # Do not convert
                return specs.BoundedArray(
                    shape=space.shape,
                    dtype=space.dtype,
                    minimum=np.zeros(space.shape),
                    maximum=space.nvec - 1,
                    name=name)

        elif isinstance(space, spaces.Tuple):
            return tuple(self._convert_to_spec(s, name) for s in space.spaces)

        elif isinstance(space, spaces.Dict):
            return {
                key: self._convert_to_spec(value, key)
                for key, value in space.spaces.items()
            }

        else:
            raise ValueError('Unexpected gym space: {}'.format(space))


def make_timeloop_env(seed: int = 12234,
                    max_steps: int = 100,
                    maximize_action: bool = False,
                    convert_multi_discrete: bool = True,
                    multi_agent: bool = False,
                    script_dir: str = None,
                    arch_dir: str = None,
                    mapper_dir: str = None,
                    workload_dir: str = None,
                    output_dir: str = None,
                    target_val: list = None,
                    reward_formulation: str = None,
                    reward_scaling: str = 'false',
                    rl_form: str = None,
                    traj_dir: str = None,
                    ) -> dm_env.Environment:
    """Returns Timeloop environment."""
    env = TimeloopEnv(
        script_dir= script_dir,
        arch_dir= arch_dir, 
        mapper_dir= mapper_dir,
        workload_dir= workload_dir,
        output_dir= output_dir,
        target_val= target_val,
        reward_formulation= reward_formulation,
        rl_form = rl_form,
        max_steps=max_steps,
        reward_scaling=reward_scaling,
        traj_dir= traj_dir,
    )
    environment = TimeloopEnvWrapper(
        env, maximize_action, convert_multi_discrete, multi_agent)
    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment
