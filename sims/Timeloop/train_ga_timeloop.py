import configparser
import copy
import argparse
import os
import sys
import time
import envlogger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.sys.path.insert(0, os.path.abspath('../../'))

from arch_gym.envs.TimeloopEnv import TimeloopEnv
from arch_gym.envs.timeloop_acme_wrapper import make_timeloop_env
from arch_gym.envs.envHelpers import helpers
from process_params import TimeloopConfigParams
from sko.GA import GA
import configs.arch_gym_configs as arch_gym_configs

from absl import app
from absl import flags
from absl import logging


# get the base directory from the file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR: ", BASE_DIR)

flags.DEFINE_string("script", str(BASE_DIR) + "/script", "Path to the script")
flags.DEFINE_string("output", str(BASE_DIR) + "/output", "Path to the output")
flags.DEFINE_string("arch", str(BASE_DIR) + "/arch", "Path to the arch")
flags.DEFINE_string("mapper", str(BASE_DIR) + "/mapper", "Path to the mapper")
flags.DEFINE_string("workload", str(BASE_DIR) + "/layer_shapes/AlexNet", "Path to the workload")
flags.DEFINE_string("summary_dir", ".", "Path to the log")
flags.DEFINE_string("params_file", str(BASE_DIR) + "/parameters.ini", "Path to the parameters file")

flags.DEFINE_string("runtime", "docker", "Runtime to use: docker or singularity")

flags.DEFINE_bool('use_envlogger', False, 'Whether to use envlogger.')
flags.DEFINE_string('reward_formulation', 'power', 'Reward formulation to use')

flags.DEFINE_integer('num_iter', 10, 'Number of training steps.')
flags.DEFINE_integer('num_agents', 16, 'Number of agents.')
flags.DEFINE_float('prob_mutation', 0.1, 'Probability of mutation.')
flags.DEFINE_string('traject_dir','ga_trajectories', 'Directory to save the dataset.')
flags.DEFINE_float('target_energy', 20444.2, 'Target energy value.')
flags.DEFINE_float('target_area', 1.7255, 'Target area value.')
flags.DEFINE_float('target_cycles', 6308563, 'Target cycles value.')

FLAGS = flags.FLAGS

def log_fitness_to_csv(filename, fitness_dict):
        df = pd.DataFrame([fitness_dict['reward']])
        csvfile = os.path.join(filename, "fitness.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

        # append to csv
        df = pd.DataFrame([fitness_dict])
        csvfile = os.path.join(filename, "trajectory.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

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


def step_fn(unused_timestep, unused_action, unused_env):
    return {'timestamp': time.time()}


def generate_run_directories():
    
    # experiment name
    if "AlexNet" in FLAGS.workload:
        workload = "AlexNet"
    elif "resnet" in FLAGS.workload:
        workload = "ResNet"
    elif "mobilenet" in FLAGS.workload:
        workload = "mobilenet"

    # Construct the exp name from seed and num_iter
    exp_name = workload + "_num_iter_" + str(FLAGS.num_iter) + "_num_agents_" + str(FLAGS.num_agents) + "_prob_mut_" + str(FLAGS.prob_mutation)
  
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)
    
    # log directories for storing exp csvs
    exp_log_dir = os.path.join(FLAGS.summary_dir,"ga_logs",FLAGS.reward_formulation, exp_name)

    return traject_dir, exp_log_dir


def timeloop_optimization_function(x):
    '''Single-threaded optimization function'''

    rewards = []
    fitness_hist = {}
    target_value = [FLAGS.target_energy, FLAGS.target_area, FLAGS.target_cycles]
    timeloop_env = TimeloopEnv(script_dir= FLAGS.script, arch_dir=FLAGS.arch, 
                            mapper_dir=FLAGS.mapper, workload_dir=FLAGS.workload,
                            output_dir=FLAGS.output, target_val=target_value,
                            reward_formulation=FLAGS.reward_formulation)
                            

    env_wrapper = make_timeloop_env(env=timeloop_env)
    envhelper = helpers()

    traject_dir, exp_log_dir = generate_run_directories()
    
    env = wrap_in_envlogger(env_wrapper, FLAGS.summary_dir)
    
    if FLAGS.use_envlogger:
        if not os.path.exists(traject_dir):
            os.makedirs(traject_dir)

    for i in range(len(x)):
        # clean up after the run
        env.reset()

        _, reward, _, info = env.step(x[i])
        rewards.append(reward)
        
        action_dict = envhelper.decode_timeloop_action(x[i])
        
        fitness_hist['reward'] = reward
        fitness_hist['action'] = action_dict
        fitness_hist['obs'] = info

        log_fitness_to_csv(exp_log_dir, fitness_hist)
    
    # Minimizes the reward as we wish to be closer to the target value
    return rewards


def timeloop_optimization_function_parallel(x):
    '''Multi-threaded optimization function'''

    # multi_actions = [helpers.decode_timeloop_action(i) for i in x]

    timeloop_env = TimeloopEnv(script_dir=SCRIPT_DIR, arch_dir=ARCH_DIR,
                               mapper_dir=MAPPER_DIR, workload_dir=WORKLOAD_DIR,
                               output_dir=OUTPUT_DIR, target_val=TARGET_VALUE)
    envhelper = helpers()
    
    env_wrapper = make_timeloop_env(env=timeloop_env, multi_agent=True)

    traject_dir, exp_log_dir = generate_run_directories()
    
    env = wrap_in_envlogger(env, FLAGS.traject_dir)
    
    if FLAGS.use_envlogger:
        if not os.path.exists(traject_dir):
            os.makedirs(traject_dir)
    
    # clean up before the run
    env.reset()

    _, rewards, _, _ = env.step(x)
    env.reset()

    return rewards


def main(_):
    if FLAGS.runtime == "docker":
        from timeloop_wrapper import TimeloopWrapper
    elif FLAGS.runtime == "singularity":
        from timeloop_wrapper_singularity import TimeloopWrapper

    wrapper = TimeloopWrapper()
    param_obj = TimeloopConfigParams(FLAGS.params_file)
    param_sizes = param_obj.get_param_size()

    # get directory names
    _, exp_log_dir = generate_run_directories()
    
    if not os.path.exists(exp_log_dir):
        os.makedirs(exp_log_dir)

    print("param_sizes: ", param_sizes)
    #import sys
    #sys.exit(0)
    # initialize GA
    ga = GA(timeloop_optimization_function,
            n_dim=len(param_sizes),
            size_pop=16,
            max_iter=FLAGS.num_iter,
            lb=[1 for _ in param_sizes],
            ub=param_sizes,
            precision=[1.0 for _ in param_sizes],
            prob_mut=FLAGS.prob_mutation)

    best_points, best_dist = ga.run()

    # Log all experiment meta data 
    envhelper = helpers()    
    print("Exp log dir: ", exp_log_dir)
    # check if exp_log_dir exists
    if not os.path.exists(exp_log_dir):
        os.makedirs(exp_log_dir)
    
    Y_history = pd.DataFrame(ga.all_history_Y)
    Y_history.to_csv(os.path.join(exp_log_dir, "ga_timelooptest.csv"))

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.savefig(os.path.join(exp_log_dir, "Y_history.png"))

    # log the best parameters for the best fitness
    best_parms = pd.DataFrame(pd.json_normalize(
        envhelper.decode_timeloop_action(best_points)))
    best_parms.to_csv(os.path.join(exp_log_dir, "best_parms.csv"))
    
if __name__ == '__main__':
   app.run(main)