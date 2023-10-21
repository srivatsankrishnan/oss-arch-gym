import gym

class OpenLaneEnv(gym.Env):
    """OpenLane Gym Environment."""
    def __init__(self                              ,
                 rtl_top           : str  = "Top"  ,
                 reward_formulation: str  = "POWER",
                 reward_scaling    : bool = False  ,
                 max_steps         : int  = 1      , 
                 num_agents        : int  = 1      ,
                 rl_form                  = None   ,
                 rl_algo                  = None   ):

        # Call base class constructor
        super(OpenLaneEnv, self).__init__()

        # Assign basic class variables 
        self.rtl_top            = rtl_top             # name of RTL top module & Verilog dir
        self.reward_formulation = reward_formulation
        self.reward_scaling     = reward_scaling
        self.max_steps          = max_steps
        self.num_agents         = num_agents
        self.rl_form            = rl_form
        self.rl_algo            = rl_algo
        self.curr_step          = 0
        self.observation        = None
        self.reward             = None
        self.done               = False
        self.info               = {}  # We do not currently make use of this metadata, but return it to DM Env Wrapper

        # Construct action and observation spaces for environment
        # TODO: Revisit definitions below based on params selected
        self.action_space = gym.spaces.Dict({
            "pdk"      : gym.spaces.Discrete(2),
            "synthesis": gym.spaces.Discrete(2),
            "floorplan": gym.spaces.Discrete(2),
            "placement": gym.spaces.Discrete(2),
            "cts"      : gym.spaces.Discrete(2),
            "route"    : gym.spaces.Discrete(2)
            })
        self.observation_space = gym.spaces.Dict({
            "power"       : gym.spaces.Box(low=0, high=1e10, shape=(1,)),
            "performance" : gym.spaces.Box(low=0, high=1e10, shape=(1,)),
            "area"        : gym.spaces.Box(low=0, high=1e10, shape=(1,))
            })

        # Reset environment upon construction
        self.reset()

    def reset(self):
        self.curr_step = 0
        return self.observation_space.sample()  # Return random sample from observation space on reset 

    def step(self, action):
        """ Step in the Gym environment with the action.

        Returns:
            observation, reward, done, info. 
        """
        if (self.curr_step == self.max_steps):
            print("Max number of steps reached: episode complete.")
            self.observation = self.reset()  # set curr_step back to zero & return random sample from observation space
            self.reward      = self.calculate_reward(self.observation)
            self.done        = True
            self.info        = {}
        else:
            self.observation = self.run_OpenLane(action)
            self.reward      = self.calculate_reward(self.observation)
            self.done        = False
            self.info        = {}
            self.curr_step   = self.curr_step + 1

        return self.observation, self.reward, self.done, self.info

    def run_OpenLane(self, action):
        """ Run OpenLane RTL to GDS flow with parameters specified by agent.

        Returns:
            observation: PPA of design.
        """
        # reward_action = action["reward_formulation"]
        pdk_action    = action["pdk"]
        synth_actions = action["synthesis"]
        fp_actions    = action["floorplan"]
        place_action  = action["placement"]
        cts_action    = action["cts"]
        route_action  = action["route"]

        # TODO: Invoke OpenLane with params above

        return self.get_observation()
        
    def get_observation(self):
        """ Gets observation (i.e. PPA) from OpenLane physically implemented design.

        Returns:
            observation: PPA of design.
        """

        # TODO: Get PPA from OpenLane logs
        observation                = {}
        observation["power"]       = 0 
        observation["performance"] = 0
        observation["area"]        = 0

        return self.observation_space.sample()  # return random sample for now

    def calculate_reward(self, observation):
        """ Calculates the reward for the agent based on reward_formulation metric (i.e. power, performance, and/or area).

        Returns:
            reward.
        """
        power       = observation["power"]
        performance = observation["performance"]
        area        = observation["area"]

        if self.reward_formulation == "POWER":
            # TODO: Normalize
            return power
        elif self.reward_formulation == "PERFORMANCE":
            # TODO: Normalize
            return performance
        elif self.reward_formulation == "AREA":
            # TODO: Normalize
            return area
        elif self.reward_formulation ==  "ALL":
            # TODO: Normalize
            return power + performance + area
    
    def render(self):
        pass

    def close(self):
        pass
