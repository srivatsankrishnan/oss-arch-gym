from absl import flags, app
import sys

sys.path.append('../../arch_gym/envs')
from CFUPlaygroundEnv import CFU_PlaygroundEnv

flags.DEFINE_integer('num_steps', 10, 'Number of training steps.')
FLAGS = flags.FLAGS

def main(_):
    env = CFU_PlaygroundEnv('cells', [1000, 1000], max_steps=FLAGS.num_steps)
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
        #action = (0, 0, 2, 1 ,9, 0, 2, 0 ,1, 0)
        
        obs, reward, done, info = env.step(action)

if __name__ == '__main__':
    app.run(main)