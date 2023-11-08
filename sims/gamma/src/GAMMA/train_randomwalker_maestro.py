import os
import sys

from absl import flags
from absl import app
from absl import logging

os.sys.path.insert(0, os.path.abspath('../../../../'))
os.sys.path.insert(0, os.path.abspath('../../../../arch_gym'))

from configs import arch_gym_configs
from arch_gym.envs.envHelpers import helpers
from arch_gym.envs import maestero_wrapper
import envlogger
import numpy as np
import pandas as pd


flags.DEFINE_string('workload', 'resnet18', 'Which DRAMSys workload to run?')
flags.DEFINE_integer('num_steps', 4, 'Number of training steps.')
flags.DEFINE_integer('num_episodes', 1, 'Number of training episodes.')
flags.DEFINE_bool('use_envlogger', False, 'Use envlogger to log the data.') 
flags.DEFINE_string('traject_dir', 
                    'random_walker_trajectories', 
            'Directory to save the dataset.')
flags.DEFINE_string('summary_dir', ".", 'Directory to save the dataset.')
flags.DEFINE_string('reward_formulation', 'latency', 'Which reward formulation to use?')
FLAGS = flags.FLAGS


def generate_random_actions(dimension):
    print(dimension)

    lb=[0, 0,  dimension['S']-1, dimension['R']-1, 1, 1, 1, 1, 0, dimension['S']-1, dimension['R']-1, 1, 1, 1, 1, 0,1],
    ub=[(2**32)-1, 3, dimension['S'], dimension['R'],
        dimension['K'], dimension['C'], dimension['X'],
        dimension['Y'], 3, dimension['S'], dimension['R'],
        dimension['K'], dimension['C'], dimension['X'], dimension['Y'], (2**32)-1, 1024]

    # generate random action with lower bound of lb and upper bound of ub.
    action = np.random.randint(lb, ub)

    return action[-1]

def log_fitness_to_csv(filename, fitness_dict):
        df = pd.DataFrame([fitness_dict['reward']])
        csvfile = os.path.join(filename, "fitness.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

        df = pd.DataFrame([fitness_dict['action']])
        csvfile = os.path.join(filename, "actions.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

        df = pd.DataFrame([fitness_dict['obs']])
        csvfile = os.path.join(filename, "observations.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

def wrap_in_envlogger(env, envlogger_dir):
    metadata = {
        'agent_type': 'RandomWalker',
        'num_steps': FLAGS.num_steps,
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

def main(_):
    env = maestero_wrapper.make_maestro_env(rl_form='random_walker', reward_formulation=FLAGS.reward_formulation)
    
    maestro_helper = helpers()
    
    fitness_hist = {}

    # experiment name 
    exp_name = str(FLAGS.workload)+"_num_steps_" + str(FLAGS.num_steps) + "_num_episodes_" + str(FLAGS.num_episodes)

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
    
    dimension,_ = maestro_helper.get_dimensions(FLAGS.workload, layer_id=2)

    for i in range(FLAGS.num_episodes):
        logging.info('Episode %r', i)
        for step in range(FLAGS.num_steps):
            # generate random actions
            action = generate_random_actions(dimension)
            
            obs, reward, c, info = env.step(action)
            
            print("Reward: ", reward)
            print("Action: ", action)
            print("Observation: ", obs)
            print("Info: ", info)
            
            #skip if the observation is the first observation
            if (obs.__dict__['_name_'] == 'FIRST'):
                print("First observation")
                continue
            else:
                fitness_hist['reward'] = [reward]
                fitness_hist['action'] = action[-1]
                fitness_hist['obs'] = info
            
                log_fitness_to_csv(log_path, fitness_hist)



if __name__ == '__main__':
   app.run(main)