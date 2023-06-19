import sys
import gym

sys.path.append("../../")
import arch_gym

def run_one_episode (env):
    env.reset()
    sum_reward = 0
    for i in range(env.max_steps):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        sum_reward += reward
        if done:
            break
    return sum_reward

env = gym.make("simpleEnv-v0")

sum_reward = run_one_episode(env)

history = []
for _ in range(10000):
    sum_reward = run_one_episode(env)
    history.append(sum_reward)
avg_sum_reward = sum(history) / len(history)
print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))
