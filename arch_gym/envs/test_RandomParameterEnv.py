import gym
import sys
from RandomParameterEnv import RandomParameterEnv

from stable_baselines3 import PPO
from stable_baselines3 import A2C


def main():
    #env = gym.make('arch-gym:random-env-v0')
    # Todo Sri: Fix this to be compatible with gym make
    env = RandomParameterEnv()
    print(env.state[0][0])
    print(env.state.shape)
    print(env.action_space.shape)
    print(env.compute_reward())

    

    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="logs/")
    model.learn(total_timesteps=10000, tb_log_name="first_run")

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()
    #model = PPO('MlpPolicy', env, verbose=1)
    
    #print(env.state)
    #print(env.action_space[0])
    #print(env.compute_reward())

if __name__ == "__main__":
    main()
