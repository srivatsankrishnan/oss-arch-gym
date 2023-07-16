from absl import flags, app
import numpy as np
from CFUenv import SimpleArch  

flags.DEFINE_integer('num_steps', 10, 'Number of training steps.')
FLAGS = flags.FLAGS

# clearing the file of its previous content
with open('Env_logfile','w') as file:
    pass

def main(_):
    num_steps = FLAGS.num_steps
    env = SimpleArch('cells', [1000, 1000])
    env.reset()
    K = 3
    for _ in range(K):
        print("________________________________________________________\n",
              "________________________________________________________\n",
              "________________________________________________________\n",
              "________________________________________________________\n")
        print("ITERATION NUMBER: ",_+1," OUT OF: ",K)
        print("________________________________________________________\n",
              "________________________________________________________\n",
              "________________________________________________________\n",
              "________________________________________________________\n")
        action = env.action_space.sample()
        #print (f'Action: {action}')
        envlog_file = open('Env_logfile','a')
        #writing the iteration number into the log file
        envlog_file.write(str(_)+',')
        envlog_file.close()
        obs, reward, done, info = env.step(action)
        #print(reward)

if __name__ == '__main__':
    app.run(main)