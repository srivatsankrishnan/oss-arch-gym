from concurrent import futures
import grpc
import portpicker
import sys
import os

sys.path.append('../../arch_gym/envs')
import CFUPlayground_wrapper
from absl import flags, app, logging

import envlogger
from envlogger.testing import catch_env   

import numpy as np
import pandas as pd


from vizier._src.algorithms.designers import grid
from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service import vizier_server
from vizier.service import vizier_service_pb2_grpc

flags.DEFINE_string('workload', 'micro_speech', 'workload the processor is being optimized for')
flags.DEFINE_integer('num_steps', 1, 'Number of training steps.')
flags.DEFINE_string('traject_dir', 'grid_search_trajectories', 'Directory to save the dataset.')
flags.DEFINE_bool('use_envlogger', True, 'Use envlogger to log the data.')  
flags.DEFINE_string('summary_dir', '.', 'Directory to save the summary.')
flags.DEFINE_string('reward_formulation', 'both', 'Which reward formulation to use?')
FLAGS = flags.FLAGS

def log_fitness_to_csv(filename, fitness_dict):
    """Logs fitness history to csv file

    Args:
        filename (str): path to the csv file
        fitness_dict (dict): dictionary containing the fitness history
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
        'agent_type': 'GridSearch',
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

    env = CFUPlayground_wrapper.make_cfuplaygroundEnv(target_vals = [1000, 1000],rl_form='GRID-SEARCH', reward_type = FLAGS.reward_formulation, max_steps = FLAGS.num_steps, workload = FLAGS.workload)
   
    fitness_hist = {}
    problem = vz.ProblemStatement()
    problem.search_space.select_root().add_int_param(name='Bypass', min_value = 0, max_value = 1)
    problem.search_space.select_root().add_int_param(name='CFU_enable', min_value = 0, max_value = 1)
    problem.search_space.select_root().add_int_param(name='Data_cache_size', min_value = 0, max_value = 10)
    problem.search_space.select_root().add_int_param(name='Hardware_Divider', min_value = 0, max_value = 1)
    problem.search_space.select_root().add_int_param(name='Instruction_cache_size', min_value = 0, max_value = 10)
    problem.search_space.select_root().add_int_param(name='Hardware_Multiplier', min_value = 0, max_value = 1)
    problem.search_space.select_root().add_int_param(name='Branch_predictor_type', min_value = 0, max_value = 3)
    problem.search_space.select_root().add_int_param(name='Safe_mode_enable', min_value = 0, max_value = 1)
    problem.search_space.select_root().add_int_param(name='Single_Cycle_Shifter', min_value = 0, max_value = 1)
    problem.search_space.select_root().add_int_param(name='Single_Cycle_Multiplier', min_value = 0, max_value = 1)
    
    problem.metric_information.append(
        vz.MetricInformation(
            name='Reward', goal=vz.ObjectiveMetricGoal.MAXIMIZE))

    
    study_config = vz.StudyConfig.from_problem(problem)
    study_config.algorithm = vz.Algorithm.GRID_SEARCH

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
    exp_name = FLAGS.workload+ "_num_steps_" + str(FLAGS.num_steps) + "_reward_type+" + FLAGS.reward_formulation

    # append logs to base path
    log_path = os.path.join(FLAGS.summary_dir, 'grid_search_logs', FLAGS.reward_formulation, exp_name)

    # get the current working directory and append the exp name
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)

    # check if log_path exists else create it
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if FLAGS.use_envlogger:
        if not os.path.exists(traject_dir):
            os.makedirs(traject_dir)
    env = wrap_in_envlogger(env, traject_dir)

    count = 0
    env.reset()
    suggestions = study.suggest(count=flags.FLAGS.num_steps)
    for suggestion in suggestions:
        count += 1
        
        action = {"Bypass": int(float(str(suggestion.parameters['Bypass']))),
            "CFU_enable": int(float(str(suggestion.parameters['CFU_enable']))),
            "Data_cache_size": int(float(str(suggestion.parameters['Data_cache_size']))),
            "Hardware_Divider": int(float(str(suggestion.parameters['Hardware_Divider']))),
            "Instruction_cache_size": int(float(str(suggestion.parameters['Instruction_cache_size']))),
            "Hardware_Multiplier": int(float(str(suggestion.parameters['Hardware_Multiplier']))),
            "Branch_predictor_type": int(float(str(suggestion.parameters['Branch_predictor_type']))),
            "Safe_mode_enable": int(float(str(suggestion.parameters['Safe_mode_enable']))),
            "Single_Cycle_Shifter": int(float(str(suggestion.parameters['Single_Cycle_Shifter']))),     
            "Single_Cycle_Multiplier": int(float(str(suggestion.parameters['Single_Cycle_Multiplier'])))}
        
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