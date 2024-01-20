import os
import sys

from absl import flags
from absl import app
from absl import logging

os.sys.path.insert(0, os.path.abspath('../../'))
#from configs import arch_gym_configs
from arch_gym.envs.envHelpers import helpers
#from arch_gym.envs import dramsys_wrapper
from arch_gym.envs import FARSI_sim_wrapper
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


#from vizier._src.algorithms.designers import quasi_random
#from vizier._src.algorithms.designers import random
from vizier._src.algorithms.designers import emukit
#from vizier._src.algorithms.designers import grid


#from vizier._src.algorithms.testing import test_runners  
#from vizier.testing import test_studies                  
#from absl.testing import absltest

os.sys.path.insert(0, os.path.abspath('../../'))
from arch_gym.envs.FARSI_sim_env import DRAMEnv


flags.DEFINE_string('workload', 'edge_detection', 'Which workload to run?')
flags.DEFINE_integer('num_steps', 500, 'Number of training steps.')
flags.DEFINE_integer('num_episodes', 1, 'Number of training episodes.')
flags.DEFINE_string('traject_dir', 
                    'emukit_FARSIsim_trajectories', 'Directory to save the dataset.')
flags.DEFINE_bool('use_envlogger', False, 'Use envlogger to log the data.') 
flags.DEFINE_string('summary_dir', '.', 'Directory to save the summary.')
flags.DEFINE_string('reward_formulation', 'power', 'Which reward formulation to use?')

#Hyperparameter flags
flags.DEFINE_integer('num_random_samples', 50, 'hyperparameter1 for emukit')

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
        'agent_type': 'EMUKIT_FARSIsim_EI',
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
    """We use emukit algorithm on the action parameter space of FARSI simulator environment
        for a given number of steps and episodes, to find out the optimal actions to be taken.
    """                                                      
    # experiment name 
    exp_name = str(FLAGS.workload)+"_num_steps_" + str(FLAGS.num_steps) + "_num_episodes_" + str(FLAGS.num_episodes)

    # append logs to base path
    log_path = os.path.join(FLAGS.summary_dir, 'emukit__FARSI_sim_logs', FLAGS.reward_formulation, exp_name)

    # get the current working directory and append the exp name
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)

    # check if log_path exists else create it                      
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if FLAGS.use_envlogger:                      
        if not os.path.exists(traject_dir):
            os.makedirs(traject_dir)

    env = FARSI_sim_wrapper.make_FARSI_sim_env(reward_formulation = FLAGS.reward_formulation, 
                                               workload=FLAGS.workload,      
                                               max_steps=FLAGS.num_steps)   #importing FARSI env here
    env = wrap_in_envlogger(env, traject_dir)
    FARSI_sim_helper = helpers()
    design_space_mode = "limited"  # ["limited", "comprehensive"]
    SOC_design_space = FARSI_sim_helper.gen_SOC_design_space(env, design_space_mode)
    encoding_dictionary = FARSI_sim_helper.gen_SOC_encoding(env, SOC_design_space)  

    #print("\n", encoding_dictionary.keys(), "\n")
    #print("\n", encoding_dictionary["encoding_flattened_ub"], "\n")              

    fitness_hist = {}

    problem = vz.ProblemStatement()
    
    #To print the lower and upper bounds of the parameter names:
    #param_names = ["pe_allocation","mem_allocation","bus_allocation",
                       #"pe_to_bus_connection","bus_to_bus_connection","bus_to_mem_connection",
                       #"task_to_pe_mapping","task_to_mem_mapping"]
    #for i in range(len(param_names)):
        #print("hola")
        #print(encoding_dictionary[param_names[i]+"_lb"])
        #print(encoding_dictionary[param_names[i]+"_ub"])

    #Adding all parameters to the vizier problem space, in accordance with their lower boundaries (lb) and upper boundaries (ub)
    #The parameters are named as param_0, param_1, param_2, etc.
    for i in range(len(encoding_dictionary["encoding_flattened_lb"])):
        problem.search_space.select_root().add_int_param(name = "param_"+str(i+1), 
                                                         min_value = encoding_dictionary["encoding_flattened_lb"][i], 
                                                         max_value = encoding_dictionary["encoding_flattened_ub"][i])

    # Our goal is to maximize reward, and thus find the set of action values which correspond to the maximum reward
    problem.metric_information.append(
        vz.MetricInformation(
            name='Reward', goal=vz.ObjectiveMetricGoal.MAXIMIZE))
    

    #study_config = vz.StudyConfig.from_problem(problem)
    #SETTING THE ALGORITHM for study_config
    #study_config.algorithm = vz.Algorithm.QUASI_RANDOM_SEARCH
    
    #SETTING CUSTOM HYPERPARAMETERS, by importing the algorithm's class:
    #mydesigner = random.RandomDesigner(problem.search_space, seed=11)  #use this for RANDOM SEARCH
    mydesigner = emukit.EmukitDesigner(problem, num_random_samples=FLAGS.num_random_samples)  #use this for EMUKIT
    #mydesigner = grid.GridSearchDesigner(problem.search_space)  #use this for GRID SEARCH
    #mydesigner = quasi_random.QuasiRandomDesigner(problem.search_space)  #use this for QUASI RANDOM SEARCH
    # setting the hyperparameters for quasi_random:
    #mydesigner._halton_generator = quasi_random._HaltonSequence(len(problem.search_space.parameters),
                                                                #skip_points=FLAGS.skip_points,
                                                                #num_points_generated=FLAGS.num_points_generated,
                                                                #scramble=FLAGS.scramble)



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

    suggestions = mydesigner.suggest(count=FLAGS.num_steps)   # suggested parameters from mydesigner

    max_reward = float('-inf')

    env.reset()

    for i in range(FLAGS.num_episodes):

        logging.info('Episode %r', i)
        count = 1
        for suggestion in suggestions:  

            print("\n")
            print(count)

            #print("\n","Suggested parameters are: ",suggestion.parameters,"\n")   

            suggested_params = []
            for i in range(len(suggestion.parameters)):
                suggested_params.append(int(str(suggestion.parameters['param_'+str(i+1)])))

            #check_system = True
            # generate action based on the suggested parameters
            #action_encoded = FARSI_sim_helper.random_walk_FARSI_array_style(env, encoding_dictionary, check_system) ##random actions 
            action_encoded = suggested_params
            print (f'Action (suggested parameters): {action_encoded}')

            # decode the actions                              
            # serialize to convert to string/dictionary
            action_dict= FARSI_sim_helper.action_decoder_FARSI(action_encoded, encoding_dictionary)  
            #See the function action_decoder_FARSI() in envHelpers.py to see how action_encoded is decoded
            #print (f'Action: {action_dict}')


            #action_first_8_pairs = dict(list(action.items())[0: 8])
            action_dict_for_logging={}
            for key in action_dict.keys():
                if "encoding" not in key:
                    action_dict_for_logging[key] = action_dict[key]
            
            print (f'Action_dict_for_logging (suggested parameters): {action_dict_for_logging}',"\n")

            obsrew = env.step(action_dict)
            print("The step function returns: ", obsrew)
            step_type, reward, discount, obs = obsrew

            print(f'Reward: {reward}')
            # See the step() function in arch_gym/envs/FARSIEnv.py for details about the observation and reward being printed here.

            # Loop added to store max reward and corresponding action:
            if reward is None:
                reward = float('-inf')   # since some error gets thrown when reward is none (happens very rarely)
            if reward > max_reward:
                trial_no = count
                max_reward = reward
                best_action = action_dict_for_logging

            #final_measurement = vz.Measurement({'Reward': reward})   #vizier is asked to keep track of Reward, so as to maximise it

            #print(type(suggestion))
            #convert the type from TrialSuggestion to Trial:
            #suggestion = suggestion.to_trial()
            #suggestion.complete(final_measurement)
            count += 1

            #fitness_hist["obs"] = [metric.item() for metric in obs]
            fitness_hist["obs"] = obs
            fitness_hist['reward'] = reward
            fitness_hist['action'] = action_dict_for_logging

            log_fitness_to_csv(log_path, fitness_hist)


        # custom loop to print the actions corresponding to max reward.   
        print("\n", "OPTIMAL ACTION, CORRESPONDING REWARD and TRIAL NO. ARE: ", "\n", "Best Action: ",
               best_action, "\n", "Max reward: ", max_reward, "\n", "Trial no.: ", trial_no)
    
if __name__ == '__main__':
   app.run(main)