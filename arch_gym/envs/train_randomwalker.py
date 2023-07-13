from absl import flags, app
import numpy as np
from CFUenv import SimpleArch  

flags.DEFINE_integer('num_steps', 10, 'Number of training steps.')
FLAGS = flags.FLAGS

def main(_):
    num_steps = FLAGS.num_steps
    env = SimpleArch('cells', [1000, 1000])
    env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        print (f'Action: {action}')
        obs, reward, done, info = env.step(action)
        print(reward)

if __name__ == '__main__':
    app.run(main)