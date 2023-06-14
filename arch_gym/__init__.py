from gym.envs.registration import register
'''
register(
    id='arch-gym-v0',
    entry_point='arch_gym.envs:ArchGymEnv-v0',
    max_episode_step=1000,
    reward_threshold=1000,
)
'''
register(
    id='random-env-v0',
    entry_point='arch_gym.envs:RandomParameterEnv',
    ) 

register(
    id="simpleEnv-v0",
    entry_point="arch_gym.envs:simpleEnv",
)

register(
    id="DRAMEnv-v0",
    entry_point="arch_gym.envs:DRAMEnv",
)