# Standard imports for every respective environment wrapper
import numpy as np
import tree
import dm_env
import gym
from   gym         import spaces
from   acme        import specs
from   acme        import types
from   acme        import wrappers
from   typing      import Any, Dict, Optional
from   OpenLaneEnv import OpenLaneEnv

class OpenLaneEnvWrapper(dm_env.Environment):
  """Deepmind Environment wrapper for OpenAI Gym environments."""

  def __init__(self, environment: gym.Env):
      self._environment     = environment 
      self._reset_next_step = True
      self._last_info       = None

      # Convert Gym observation & action spaces into Deepmind's acme specs
      self._observation_spec = _convert_to_spec(self._environment.observation_space, name="observation")
      self._action_spec      = _convert_to_spec(self._environment.action_space     , name="action"     )

  def reset(self) -> dm_env.TimeStep:
      """Resets the episode."""
      self._reset_next_step = False
      self._last_info       = None                       # reset the diagnostic info 
      observation           = self._environment.reset()  # reset the underlying Gym environment
      return dm_env.restart(observation)                 # returns initial dm_env.TimeStep at restart using obs

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
      """Steps in the environment."""
      if self._reset_next_step:
          return self.reset()

      # Step in the Gym environment with the action 
      observation, reward, done, info = self._environment.step(action)
       
      # Set corresponding class variables returned from step
      self._reset_next_step           = done
      self._last_info                 = info

      # Convert the reward type to conform  to structure specified by self.reward_spec(), respecting scalar or array property.
      reward = tree.map_structure(
          lambda x, t: (  
              t.dtype.type(x)
              if np.isscalar(x) else np.asarray(x, dtype=t.dtype)),
          reward,
          self.reward_spec())

      # If episode complete, return appropriate cause (i.e., timesteps max limit reached or natural episode end)  
      if done:
          truncated = info.get("TimeLimit.truncated", False)
          if truncated:  
              # Episode concluded because max timesteps reached
              return dm_env.truncation(reward, observation)
          else:
              # Episode concluded  
              return dm_env.termination(reward, observation)

      # Episode continuing, provide agent with reward + obs returned from applied action when step taken in environment
      return dm_env.transition(reward, observation)

  def observation_spec(self) -> types.NestedSpec:
      """Retrieve the specification for environment observations.
      
      Returns:
          types.NestedSpec: Specification detailing the format of observations.
      """
      return self._observation_spec
  
  def action_spec(self) -> types.NestedSpec:
      """Retrieve the specification for valid agent actions.
      
      Returns:
          types.NestedSpec: Specification detailing the format of actions.
      """
      return self._action_spec
     
  def reward_spec(self):
      """Retrieve the specification for environment rewards.
  
      Returns:
          specs.Array: Specification detailing the format of rewards.
      """
      return specs.Array(shape=(), dtype=float, name='reward')

  def get_info(self) -> Optional[Dict[str, Any]]:
      """Returns the last info returned from env.step(action).

      Returns:
        info: Dictionary of diagnostic information from the last environment step.
      """
      return self._last_info

  @property
  def environment(self) -> gym.Env:
      """Return the wrapped Gym environment.
      
      Returns:
          gym.Env: The underlying environment.
      """
      return self._environment
  
  def __getattr__(self, name: str):
      """Retrieve attributes from the wrapped environment.
  
      Args:
          name (str): Name of the attribute to retrieve.
  
      Returns:
          Any: The value of the attribute from the wrapped environment.
  
      Raises:
          AttributeError: If attempting to access a private attribute.
      """
      if name.startswith('__'):
          raise AttributeError(
              f"Attempted to get missing private attribute '{name}'")
      return getattr(self._environment, name)
  
  def close(self):
      """Close and clean up the wrapped environment."""
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

  else:
    raise ValueError('Unexpected gym space: {}'.format(space))


def make_OpenLaneEnvironment(
        seed              : int  = 12345  ,
        rtl_top           : str  = "Top"  ,
        reward_formulation: str  = "POWER",
        reward_scaling    : bool = False  ,
        max_steps         : int  = 1      , 
        num_agents        : int  = 1      ,
        rl_form                  = None   ,
        rl_algo                  = None
        ) -> dm_env.Environment:

    """Returns instance of OpenLane Gym Environment wrapped around DM Environment."""
    environment = OpenLaneEnvWrapper(
                      OpenLaneEnv(
                          rtl_top            = rtl_top           ,
                          reward_formulation = reward_formulation,
                          reward_scaling     = reward_scaling    ,
                          max_steps          = max_steps         ,
                          num_agents         = num_agents        ,
                          rl_form            = rl_form           ,
                          rl_algo            = rl_algo
                      )
                  )
      
    # Set obs, reward, and action spec precision of the environment and clipping if needed
    environment = wrappers.SinglePrecisionWrapper(environment)
    if(rl_form == 'sa' or rl_form == 'tdm'):
      environment = wrappers.CanonicalSpecWrapper(environment, clip=True)

    return environment
