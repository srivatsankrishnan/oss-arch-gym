from absl import flags, app
import os
import envlogger
from envlogger.testing import catch_env   
import sys

sys.path.append('../../arch_gym/envs')
import CFUPlayground_wrapper

flags.DEFINE_string('workload', 'micro_speech', 'workload the processor is being optimized for')
flags.DEFINE_integer('num_steps', 1, 'Number of training steps.')
flags.DEFINE_bool('use_envlogger', True, 'Use envlogger to log the data.') 
flags.DEFINE_string('traject_dir', 'random_walker_trajectories', 'Directory to save the dataset.')
flags.DEFINE_string('summary_dir', ".", 'Directory to save the dataset.')
flags.DEFINE_string('reward_formulation', 'both', 'The kind of reward we are optimizing for')

FLAGS = flags.FLAGS

envdm = catch_env.Catch()


def wrap_in_envlogger(env,envlogger_dir):
   metadata = {
      'agent_type' : 'RandomWalker',
      'num_steps': FLAGS.num_steps,
      'env_type': type(env).__name__,
   }  
   if FLAGS.use_envlogger:
      env = envlogger.EnvLogger(env, data_directory=envlogger_dir, max_episodes_per_file=1000, metadata=metadata)
      return env
   else:
      return env

def main(_):
   env = CFUPlayground_wrapper.make_cfuplaygroundEnv(target_vals = [1000, 1000],rl_form='random_walker', reward_type = FLAGS.reward_formulation, max_steps = FLAGS.num_steps, workload = FLAGS.workload)
   # experiment name 
   exp_name =  FLAGS.workload + "_num_steps_" + str(FLAGS.num_steps) + "_reward_type+" + FLAGS.reward_formulation
   # append logs to base path
   log_path = os.path.join(FLAGS.summary_dir, 'random_walker_logs', FLAGS.reward_formulation, exp_name)
    # get the current working directory and append the exp name
   traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)
    # check if log_path exists else create it
   if not os.path.exists(log_path):
        os.makedirs(log_path)
   if FLAGS.use_envlogger:
        if not os.path.exists(traject_dir):
            os.makedirs(traject_dir)
   env = wrap_in_envlogger(env, traject_dir)
   env.reset()

   for step in range(FLAGS.num_steps):
      print("________________________________________________________\n",
             "________________________________________________________\n",
             "________________________________________________________\n",
             "________________________________________________________\n")
      print("ITERATION NUMBER: ",step+1," OUT OF: ",FLAGS.num_steps)
      print("________________________________________________________\n",
             "________________________________________________________\n",
             "________________________________________________________\n",
             "________________________________________________________\n")
      # generate random actions
      action = env.action_space.sample()

      obs, reward, done, info = env.step(action)

   env.close()
      
if __name__ == '__main__':
   app.run(main)
