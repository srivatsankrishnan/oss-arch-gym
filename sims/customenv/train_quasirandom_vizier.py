import os
import sys

from absl import flags
from absl import app
from absl import logging

os.sys.path.insert(0, os.path.abspath('../../'))
#from configs import arch_gym_configs
#from arch_gym.envs.envHelpers import helpers
#from arch_gym.envs import dramsys_wrapper
from arch_gym.envs import customenv_wrapper_2
import envlogger

import numpy as np
import pandas as pd

from concurrent import futures
import grpc
import portpicker

from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service import vizier_server
from vizier.service import vizier_service_pb2_grpc

from typing import Optional, Sequence

from vizier import algorithms as vza
from vizier import pythia
#from vizier.algorithms import designers 


from vizier._src.algorithms.designers import quasi_random
#from vizier._src.algorithms.designers import random
#from vizier._src.algorithms.designers import emukit
#from vizier._src.algorithms.designers import grid


#from vizier._src.algorithms.testing import test_runners  
#from vizier.testing import test_studies                  
#from absl.testing import absltest

os.sys.path.insert(0, os.path.abspath('../../'))
from arch_gym.envs.custom_env_2 import SimpleArch


flags.DEFINE_string('workload', 'custom_env_workload', 'Which DRAMSys workload to run?')
flags.DEFINE_integer('num_steps', 1000, 'Number of training steps.')
flags.DEFINE_integer('num_episodes', 1, 'Number of training episodes.')
flags.DEFINE_string('traject_dir', 
                    'quasi_random_trajectories', 'Directory to save the dataset.')
flags.DEFINE_bool('use_envlogger', False, 'Use envlogger to log the data.') 
flags.DEFINE_string('summary_dir', '.', 'Directory to save the summary.')
flags.DEFINE_string('reward_formulation', 'power', 'Which reward formulation to use?')

#Hyperparameter flags
flags.DEFINE_integer('skip_points', 0, 'hyperparameter1 for quasi_random')
flags.DEFINE_integer('num_points_generated', 0, 'hyperparameter2 for quasi_random')
flags.DEFINE_bool('scramble', False, 'hyperparameter3 for quasi_random')

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
        'agent_type': 'QUASI_RANDOM_EI',
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
    env = customenv_wrapper_2.make_custom_env(max_steps=FLAGS.num_steps)   #importing custom env here

    #dram_helper = helpers()                          
    
    fitness_hist = {}
                                                      
    # experiment name 
    exp_name = str(FLAGS.workload)+"_num_steps_" + str(FLAGS.num_steps) + "_num_episodes_" + str(FLAGS.num_episodes)

    # append logs to base path
    log_path = os.path.join(FLAGS.summary_dir, 'quasi_random_logs', FLAGS.reward_formulation, exp_name)

    # get the current working directory and append the exp name
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)

    # check if log_path exists else create it                      
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if FLAGS.use_envlogger:                      
        if not os.path.exists(traject_dir):
            os.makedirs(traject_dir)
    env = wrap_in_envlogger(env, traject_dir)


    problem = vz.ProblemStatement()

    problem.search_space.select_root().add_int_param(name = 'num_cores', min_value=0, max_value=10)
    problem.search_space.select_root().add_float_param(name = 'freq', min_value=0, max_value=5)
    problem.search_space.select_root().add_categorical_param(name='mem_type', feasible_values =['DRAM', 'SRAM', 'Hybrid'])
    problem.search_space.select_root().add_discrete_param(name = 'mem_size', feasible_values = [0,16,32,64,128,256])


    # Our goal is to maximize reward, and thus find the set of action values which correspond to the maximum reward
    #problem.metric_information.append(
        #vz.MetricInformation(
            #name='Reward', goal=vz.ObjectiveMetricGoal.MAXIMIZE))
    

    #study_config = vz.StudyConfig.from_problem(problem)
    #a = study_config.search_space
    #print(a)

    #SETTING THE ALGORITHM
    #study_config.algorithm = vz.Algorithm.QUASI_RANDOM_SEARCH
    
    #SETTING CUSTOM HYPERPARAMETERS, by importing the algorithm's class:
    #mydesigner = random.RandomDesigner(problem.search_space, seed=110)  #use this for RANDOM SEARCH
    #mydesigner = emukit.EmukitDesigner(problem, num_random_samples=10)  #use this for EMUKIT
    #mydesigner = grid.GridSearchDesigner(problem.search_space)  #use this for GRID SEARCH
    mydesigner = quasi_random.QuasiRandomDesigner(problem.search_space)  #use this for QUASI RANDOM SEARCH
    # setting the hyperparameters for quasi_random:
    mydesigner._halton_generator = quasi_random._HaltonSequence(len(problem.search_space.parameters),
                                                                skip_points=FLAGS.skip_points,
                                                                num_points_generated=FLAGS.num_points_generated,
                                                                scramble=FLAGS.scramble)

    #print(type(mydesigner))


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
    #study = clients.Study.from_study_config(study_config, owner='owner', study_id='example_study_id')

    #suggestions = study.suggest(count=FLAGS.num_steps)
    suggestions = mydesigner.suggest(count=FLAGS.num_steps)

    max_reward = float('-inf')
    env.reset()

    for i in range(FLAGS.num_episodes):

        logging.info('Episode %r', i)
        count = 1
        for suggestion in suggestions:                

            num_cores = str(suggestion.parameters['num_cores'])
            freq = str(suggestion.parameters['freq'])
            mem_type_dict = {'DRAM':0, 'SRAM':1, 'Hybrid':2}
            mem_type = str(mem_type_dict[str(suggestion.parameters['mem_type'])])
            mem_size = str(suggestion.parameters['mem_size'])


            print("\n")
            print(count)
            print('Suggested Parameters (num_cores, freq, mem_type, mem_size):', num_cores, freq, mem_type, mem_size)


            # generate action based on the suggested parameters
            action = {"num_cores":float(num_cores), "freq": float(freq), "mem_type":float(mem_type), "mem_size": float(mem_size)}
            #for alternative method to convert action values into float datatype, see earlier commits (earlier version of this file)
            print (f'Action: {action}')

            # decode the actions                              
            #action_dict = dram_helper.action_decoder_ga(action)
            #_, reward, c, info = env.step(action_dict)

            #obs, reward, done, info = env.step(action)
            print("\n")
            obsrew = env.step(action)
            print(obsrew)
            step_type, reward, discount, obs = obsrew

            env.render()  #prints the observation which is (energy, area, latency)
            print(f'Reward: {reward}')

            # loop added to store max reward and corresponding action:
            if reward > max_reward:
                max_reward = reward
                best_action = action

            #final_measurement = vz.Measurement({'Reward': reward})

            #print(type(suggestion))
            #convert the type from TrialSuggestion to Trial:
            #suggestion = suggestion.to_trial()
            #suggestion.complete(final_measurement)
            count += 1

            fitness_hist['obs'] = obs
            fitness_hist['reward'] = reward
            fitness_hist['action'] = action

            log_fitness_to_csv(log_path, fitness_hist)


        # custom loop to print the actions corresponding to max reward.   
        print("\n", "OPTIMAL ACTION AND CORRESPONDING REWARD ARE: ", "\n", "Best Action: ",
               best_action, "\n", "Max reward: ", max_reward)
    
if __name__ == '__main__':
   app.run(main)

## why are the rewards in "wrapper step rew" and "Reward:"" slightly different?