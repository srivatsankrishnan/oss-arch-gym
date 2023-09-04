from concurrent import futures
import grpc
import portpicker
import sys
import os


from absl import flags
from absl import app
from absl import logging

os.sys.path.insert(0, os.path.abspath('../../'))

# from configs import arch_gym_configs
# from arch_gym.envs.envHelpers import helpers

import envlogger
import numpy as np
import pandas as pd


from vizier._src.algorithms.evolution.nsga2 import NSGA2Survival, create_nsga2
from arch_gym.envs import customenv_wrapper
from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service import vizier_server
from vizier.service import vizier_service_pb2_grpc

# flags.DEFINE_string('workload', 'stream.stl', 'Which DRAMSys workload to run?')
flags.DEFINE_integer('num_steps', 50, 'Number of training steps.')
flags.DEFINE_integer('num_episodes', 2, 'Number of training episodes.')
flags.DEFINE_string('traject_dir', 
                    'NSGA2_trajectories', 
            'Directory to save the dataset.')
flags.DEFINE_bool('use_envlogger', False, 'Use envlogger to log the data.')  
flags.DEFINE_string('summary_dir', '.', 'Directory to save the summary.')
flags.DEFINE_string('reward_formulation', 'power', 'Which reward formulation to use?')
flags.DEFINE_integer('population_size', 100, 'hyperparameter1 for NSGA2')
flags.DEFINE_integer('eviction_limit', 3, 'hyperparameter2 for NSGA2')

FLAGS = flags.FLAGS

def log_fitness_to_csv(filename, fitness_dict):
    """Wraps the environment in envlogger

    Args:
        env (gym.Env): gym environment
        envlogger_dir (str): path to the directory where the data will be logged
    """

        
    df = pd.DataFrame([fitness_dict['reward']])
    csvfile = os.path.join(filename, "fitness.csv")
    df.to_csv(csvfile, index=False, header=False, mode='a')

    # append to csv
    df = pd.DataFrame([fitness_dict])
    csvfile = os.path.join(filename, "trajectory.csv")
    df.to_csv(csvfile, index=False, header=False, mode='a')

def wrap_in_envlogger(env, envlogger_dir):
    """Wraps the environment in envlogger

    Args:
        env (gym.Env): gym environment
        envlogger_dir (str): path to the directory where the data will be logged
    """
    metadata = {
        'agent_type': 'NSGA2',
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
    """Trains the custom environment using random actions for a given number of steps and episodes 
    """

    env = customenv_wrapper.make_custom_env(max_steps=FLAGS.num_steps)
    fitness_hist = {}
    problem = vz.ProblemStatement()
    problem.search_space.select_root().add_int_param(name='num_cores', min_value = 1, max_value = 12)
    problem.search_space.select_root().add_float_param(name='freq', min_value = 0.5, max_value = 3)
    problem.search_space.select_root().add_categorical_param(name='mem_type', feasible_values =['DRAM', 'SRAM', 'Hybrid'])
    problem.search_space.select_root().add_discrete_param(name='mem_size', feasible_values=[0, 32, 64, 128, 256, 512])

    problem.metric_information.append(
        vz.MetricInformation(
            name='Reward', goal=vz.ObjectiveMetricGoal.MAXIMIZE))



    study_config = vz.StudyConfig.from_problem(problem)
    # study_config.algorithm = vz.Algorithm.NSGA2

    # nsga2_evolution = NSGA2Survival(target_size = FLAGS.target_size, eviction_limit= FLAGS.eviction_limit)
    nsga2_designer = create_nsga2(problem, population_size = FLAGS.population_size, eviction_limit= FLAGS.eviction_limit )
    port = portpicker.pick_unused_port()
    address = f'localhost:{port}'

    # Setup server.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))

    # Setup Vizier Service.
    servicer = vizier_server.VizierService()
    vizier_service_pb2_grpc.add_VizierServiceServicer_to_server(servicer, server)
    server.add_secure_port(address, grpc.local_server_credentials())

    # Start the server.
    server.start()

    clients.environment_variables.service_endpoint = address  # Server address.
    study = clients.Study.from_study_config(
        study_config, owner='owner', study_id='example_study_id')

     # experiment name 
    exp_name = "_num_steps_" + str(FLAGS.num_steps) + "_num_episodes_" + str(FLAGS.num_episodes)

    # append logs to base path
    log_path = os.path.join(FLAGS.summary_dir, 'NSGA2_logs', FLAGS.reward_formulation, exp_name)

    # get the current working directory and append the exp name
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)

    # check if log_path exists else create it
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if FLAGS.use_envlogger:
        if not os.path.exists(traject_dir):
            os.makedirs(traject_dir)
    env = wrap_in_envlogger(env, traject_dir)

    # for i in range(flags.FLAGS.num_episodes):
    max_reward = float('-inf')
    env.reset()
    count = 0
    suggestions = nsga2_designer.suggest(count=flags.FLAGS.num_steps)
    for suggestion in suggestions:
        count += 1
        num_cores = str(suggestion.parameters['num_cores'])
        freq = str(suggestion.parameters['freq'])
        mem_type_dict = {'DRAM':0, 'SRAM':1, 'Hybrid':2}
        mem_type = str(mem_type_dict[str(suggestion.parameters['mem_type'])])
        mem_size = str(suggestion.parameters['mem_size'])
        
        action = {"num_cores":float(num_cores), "freq": float(freq), "mem_type":float(mem_type), "mem_size": float(mem_size)}
        
        print("Suggested Parameters for num_cores, freq, mem_type, mem_size are :", num_cores, freq, mem_type, mem_size)
        done, reward, info, obs = (env.step(action))
        fitness_hist['reward'] = reward
        fitness_hist['action'] = action
        fitness_hist['obs'] = obs
        if count == FLAGS.num_steps:
            done = True
        log_fitness_to_csv(log_path, fitness_hist)
        print("Observation: ",obs)
        
        final_measurement = vz.Measurement({'Reward': reward})
        suggestion = suggestion.to_trial()
        suggestion.complete(final_measurement)
    


   

if __name__ == '__main__':
   app.run(main)