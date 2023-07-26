from absl import flags, app
import numpy as np
from CFUPlaygroundEnv import SimpleArch  

flags.DEFINE_integer('num_steps', 10, 'Number of training steps.')
FLAGS = flags.FLAGS

def main(_):
    env = SimpleArch('cells', [1000, 1000], max_steps=FLAGS.num_steps)
    env.reset()
    for _ in range(FLAGS.num_steps):
        print("________________________________________________________\n",
              "________________________________________________________\n",
              "________________________________________________________\n",
              "________________________________________________________\n")
        print("ITERATION NUMBER: ",_+1," OUT OF: ",FLAGS.num_steps)
        print("________________________________________________________\n",
              "________________________________________________________\n",
              "________________________________________________________\n",
              "________________________________________________________\n")
        action = env.action_space.sample()
        
        obs, reward, done, info = env.step(action)

if __name__ == '__main__':
    app.run(main)