import os
import sys

from absl import flags
from absl import app
from absl import logging

os.sys.path.insert(0, os.path.abspath('../../'))
#from configs import arch_gym_configs
#from arch_gym.envs.envHelpers import helpers
#from arch_gym.envs import dramsys_wrapper
#import envlogger
import numpy as np
import pandas as pd

from concurrent import futures
import grpc
import portpicker

from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service import vizier_server
from vizier.service import vizier_service_pb2_grpc


os.sys.path.insert(0, os.path.abspath('../../'))
from arch_gym.envs.custom_env_2 import SimpleArch


flags.DEFINE_string('workload', 'some random stuff', 'Which DRAMSys workload to run?')
flags.DEFINE_integer('num_steps', 1000, 'Number of training steps.')
flags.DEFINE_integer('num_episodes', 1, 'Number of training episodes.')
flags.DEFINE_string('traject_dir', 
                    'random_walker_trajectories', 'Directory to save the dataset.') 
flags.DEFINE_bool('use_envlogger', False, 'Use envlogger to log the data.')
flags.DEFINE_string('summary_dir', '.', 'Directory to save the summary.')
flags.DEFINE_string('reward_formulation', 'power', 'Which reward formulation to use?')

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
    # pass your environment to it.
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
    env = SimpleArch()     #importing custom env here
    env.reset()

    #dram_helper = helpers()       
    
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

    #if FLAGS.use_envlogger:                     
        #if not os.path.exists(traject_dir):
            #os.makedirs(traject_dir)
    #env = wrap_in_envlogger(env, traject_dir)


    problem = vz.ProblemStatement()

    problem.search_space.select_root().add_int_param(name = 'num_cores', min_value=0, max_value=10)
    problem.search_space.select_root().add_float_param(name = 'freq', min_value=0, max_value=5)
    problem.search_space.select_root().add_discrete_param(name = 'mem_type', feasible_values = [0,1,2])
    problem.search_space.select_root().add_discrete_param(name = 'mem_size', feasible_values = [0,16,32,64,128,256])


    # Our goal is to maximize reward, and thus find the set of action values which correspond to the maximum reward
    problem.metric_information.append(
        vz.MetricInformation(
            name='Reward', goal=vz.ObjectiveMetricGoal.MAXIMIZE))


    #def evaluate(num_cores, freq, mem_type, mem_size):
    ##Random formulae for now
    #energy = num_cores + freq
    #area = num_cores * freq
    #latency = num_cores + mem_size
    #return (energy, area, latency)

    study_config = vz.StudyConfig.from_problem(problem)

    #SETTING THE ALGORITHM
    study_config.algorithm = vz.Algorithm.QUASI_RANDOM_SEARCH

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

    suggestions = study.suggest(count=FLAGS.num_steps)
    
    for i in range(FLAGS.num_episodes):

        logging.info('Episode %r', i)
        count = 1
        for suggestion in suggestions:              

            num_cores = suggestion.parameters['num_cores']
            freq =  suggestion.parameters['freq']
            mem_type =  suggestion.parameters['mem_type']
            mem_size =  suggestion.parameters['mem_size']


            print("\n")
            print(count)
            print('Suggested Parameters (num_cores, freq, mem_type, mem_size):', num_cores, freq, mem_type, mem_size)


            # generate random actions
            action = [num_cores, freq, mem_type, mem_size]
            print (f'Action: {action}')


            # decode the actions                              
            #action_dict = dram_helper.action_decoder_ga(action)

            #_, reward, c, info = env.step(action_dict)
            obs, reward, done, info = env.step(action)
            env.render()  #prints the observation which is (energy, area, latency)
            print(f'Reward: {reward}')

            final_measurement = vz.Measurement({'Reward': reward})

            suggestion.complete(final_measurement)
            count += 1

            fitness_hist['reward'] = reward
            fitness_hist['action'] = action
            #fitness_hist['obs'] = info
            fitness_hist['obs'] = obs

            log_fitness_to_csv(log_path, fitness_hist)

        count = 1
        for optimal_trial in study.optimal_trials():
            optimal_trial = optimal_trial.materialize()
            print("\n")
            print(count)
            print("Optimal Trial Suggestion and Objective:", optimal_trial.parameters,
                    optimal_trial.final_measurement)
            count += 1


if __name__ == '__main__':
   app.run(main)

