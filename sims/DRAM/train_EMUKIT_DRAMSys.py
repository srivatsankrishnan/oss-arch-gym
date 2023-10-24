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


from vizier._src.algorithms.designers.emukit import EmukitDesigner
from arch_gym.envs import dramsys_wrapper
from arch_gym.envs.envHelpers import helpers
from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service import vizier_server
from vizier.service import vizier_service_pb2_grpc

flags.DEFINE_string('workload', 'stream.stl', 'Which DRAMSys workload to run?')
flags.DEFINE_integer('num_steps', 100, 'Number of training steps.')
flags.DEFINE_integer('num_episodes', 2, 'Number of training episodes.')
flags.DEFINE_string('traject_dir', 
                    'EMUKIT_trajectories', 
            'Directory to save the dataset.')
flags.DEFINE_bool('use_envlogger', False, 'Use envlogger to log the data.')  
flags.DEFINE_string('summary_dir', '.', 'Directory to save the summary.')
flags.DEFINE_string('reward_formulation', 'power', 'Which reward formulation to use?')
flags.DEFINE_integer('num_random_sample', 100, 'hyperparameter for emukit')
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
        'agent_type': 'EMUKIT',
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

    env = dramsys_wrapper.make_dramsys_env()
    
    dram_helper = helpers()
    
    fitness_hist = {}
    problem = vz.ProblemStatement()
    

    problem.search_space.select_root().add_int_param(name='pagepolicy', min_value = 0, max_value = 3)
    problem.search_space.select_root().add_int_param(name='scheduler', min_value = 0, max_value = 2)
    problem.search_space.select_root().add_int_param(name='schedulerbuffer', min_value = 0, max_value = 2)
    problem.search_space.select_root().add_discrete_param(name='reqest_buffer_size', feasible_values=[1, 2, 4, 8, 16, 32, 64, 128])
    problem.search_space.select_root().add_int_param(name='respqueue', min_value = 0, max_value = 1)
    problem.search_space.select_root().add_int_param(name='refreshpolicy', min_value = 0, max_value = 1)
    problem.search_space.select_root().add_discrete_param(name='refreshmaxpostponed', feasible_values=[1, 2, 4, 8])
    problem.search_space.select_root().add_discrete_param(name='refreshmaxpulledin', feasible_values=[1, 2, 4, 8])
    # problem.search_space.select_root().add_int_param(name='powerdownpolicy', min_value = 0, max_value = 2)
    problem.search_space.select_root().add_int_param(name='arbiter', min_value = 0, max_value = 2)
    problem.search_space.select_root().add_discrete_param(name='maxactivetransactions', feasible_values=[1, 2, 4, 8, 16, 32, 64, 128])

    problem.metric_information.append(
        vz.MetricInformation(
            name='Reward', goal=vz.ObjectiveMetricGoal.MAXIMIZE))

   


    study_config = vz.StudyConfig.from_problem(problem)
    emukit_designer = EmukitDesigner(problem, num_random_samples= FLAGS.num_random_sample)


    

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
    log_path = os.path.join(FLAGS.summary_dir, 'emukit_logs', FLAGS.reward_formulation, exp_name)

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
    
    count = 0
    suggestions = emukit_designer.suggest(count=flags.FLAGS.num_steps)
    
    for suggestion in suggestions:
        count += 1
        
        PagePolicy = str(suggestion.parameters['pagepolicy'])
        Scheduler = str(suggestion.parameters['scheduler'])
        SchedulerBuffer = str(suggestion.parameters['schedulerbuffer'])
        RequestBufferSize = str(suggestion.parameters['reqest_buffer_size'])
        RespQueue = str(suggestion.parameters['respqueue'])
        RefreshPolicy = str(suggestion.parameters['refreshpolicy'])
        RefreshMaxPostponed = str(suggestion.parameters['refreshmaxpostponed'])
        RefreshMaxPulledin = str(suggestion.parameters['refreshmaxpulledin'])
        Arbiter = str(suggestion.parameters['arbiter'])
        MaxActiveTransactions = str(suggestion.parameters['maxactivetransactions'])

        vizier_actions = [PagePolicy, Scheduler, SchedulerBuffer, RequestBufferSize, RespQueue, RefreshPolicy, RefreshMaxPostponed,
                   RefreshMaxPulledin, Arbiter, MaxActiveTransactions]
        
        
        action = np.asarray(vizier_actions)
        action_dict = dram_helper.action_decoder_ga(action)

        # print("Suggested Parameters for num_cores, freq, mem_type, mem_size are :", num_cores, freq, mem_type, mem_size)
        done, reward, info, obs = (env.step(action_dict))
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