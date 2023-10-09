import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.sys.path.insert(0, os.path.abspath('../../'))

from sko.GA                   import GA
from arch_gym.envs.DRAMEnv    import DRAMEnv
from arch_gym.envs            import dramsys_wrapper
from arch_gym.envs.envHelpers import helpers
from absl                     import flags
from absl                     import app
from absl                     import logging
import envlogger


flags.DEFINE_bool('use_envlogger', False, 'Whether to use envlogger.')
flags.DEFINE_string('reward_formulation', 'power', 'Reward formulation to use')
flags.DEFINE_string('workload', 'stream.stl', 'Workload trace file')
flags.DEFINE_integer('num_iter', 100, 'Number of training steps.')
flags.DEFINE_integer('num_agents', 32, 'Number of agents.')
flags.DEFINE_float('prob_mutation', 0.1, 'Probability of mutation.')
flags.DEFINE_string('traject_dir','ga_trajectories', 'Directory to save the dataset.')
flags.DEFINE_string('summary_dir', '.', 'Directory to save the summary.')

FLAGS = flags.FLAGS

def wrap_in_envlogger(env, envlogger_dir):
    metadata = {
        'agent_type': 'RandomWalker',
        'num_steps': FLAGS.num_iter,
        'env_type': type(env).__name__,
    }
    if FLAGS.use_envlogger:
        logging.info('Wrapping environment with EnvironmentLogger...')
        env = envlogger.EnvLogger(env,
                                  data_directory=envlogger_dir,
                                  max_episodes_per_file=1000,
                                  metadata=metadata)
        logging.info('Done wrapping environment with EnvironmentLogger.')
        return env
    else:
        return env


def generate_run_directories():
    # Construct the exp name from seed and num_iter
    exp_name = FLAGS.workload + "_num_iter_" + str(FLAGS.num_iter) + "_num_agents_" + str(FLAGS.num_agents) + "_prob_mut_" + str(FLAGS.prob_mutation)
  
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)
    
    # log directories for storing exp csvs
    exp_log_dir = os.path.join(FLAGS.summary_dir,"ga_logs",FLAGS.reward_formulation, exp_name)

    return traject_dir, exp_log_dir
    
def dram_optimization_function(p):
    '''
    This function is used to optimize the DRAM parameters. The default objective is to minimize. If you have a reward/fitness formulation
    that is to be maximized, you can simply return -1 * your_reward.
    '''
    rewards = []
    print("Agents Action", p)
    # instantiate the environment and the helpers
    env = dramsys_wrapper.make_dramsys_env(reward_formulation = FLAGS.reward_formulation)
    dram_helper = helpers()

    traject_dir, exp_log_dir = generate_run_directories()
    
    env = wrap_in_envlogger(env, FLAGS.summary_dir)
    
    if FLAGS.use_envlogger:
        if not os.path.exists(traject_dir):
            os.makedirs(traject_dir)
    env.reset()

    # decode the actions
    action_dict = dram_helper.action_decoder_ga(p)
            
    # take a step in the environment
    _, reward, done, info = env.step(action_dict)

    fitness_dict = {}
    fitness_dict["action"] = action_dict
    fitness_dict["reward"] = reward
    fitness_dict["obs"] = info

    # Convert dictionary to dataframe
    fitness_df = pd.DataFrame([fitness_dict], columns=["action", "reward", "obs"])

    # check if exp_log_dir exists
    if not os.path.exists(exp_log_dir):
        os.makedirs(exp_log_dir)

    # write it to csv file append mode
    fitness_df.to_csv(os.path.join(exp_log_dir, "fitness.csv"), mode='a', header=False, index=False)
    rewards.append(reward)
    
    return -1 * reward
    

def main(_):

    ga = GA(
        func=dram_optimization_function, 
        n_dim=10, 
        size_pop=FLAGS.num_agents,
        max_iter=FLAGS.num_iter,
        prob_mut=FLAGS.prob_mutation,
        lb=[0, 0, 0, 1, 0, 0, 1, 1, 0, 1], 
        ub=[3, 2, 2, 8, 1, 1, 8, 8, 2, 128], 
        precision=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
    )

    best_x, best_y = ga.run()

    # get directory names
    _, exp_log_dir = generate_run_directories()

    # check if exp_log_dir exists
    if not os.path.exists(exp_log_dir):
        os.makedirs(exp_log_dir)
    
    Y_history = pd.DataFrame(ga.all_history_Y)
    Y_history.to_csv(os.path.join(exp_log_dir, "Y_history.csv"))

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.savefig(os.path.join(exp_log_dir, "Y_history.png"))

if __name__ == '__main__':
   app.run(main)
