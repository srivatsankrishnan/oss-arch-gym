
from absl import flags
from absl import app
import sys
import os.path

os.sys.path.insert(0, os.path.abspath('../../'))


from arch_gym.envs.custom_env import CustomEnv

flags.DEFINE_integer('num_steps', 4, 'Number of training steps')
flags.FLAGS(sys.argv)
steps = flags.FLAGS.num_steps
print(steps)
env = CustomEnv()
observation = env.reset()


def main(_):
    i = 1
    while not i > steps:
        env.render()
        action = env.action_space.sample()
        print("The taken action is {}".format(action))
        obs, reward, done, info = (env.step(action))
        print("The reward is {}".format(reward))
        i += 1

if __name__ == '__main__':
    app.run(main)