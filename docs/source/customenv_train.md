 Importing dependencies


```python
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

```

 Import customenv_wrapper - it converts custom environment to deepmind envlogger environment, vizier algorithm designer 


```python
from vizier._src.algorithms.designers.random import RandomDesigner
from arch_gym.envs import customenv_wrapper
from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service import vizier_server
from vizier.service import vizier_service_pb2_grpc

```

Define Flags - for taking in command line inputs 


```python
flags.DEFINE_string('workload_rs', 'stream.stl', 'Which DRAMSys workload to run?')
flags.DEFINE_integer('num_steps_rs', 100, 'Number of training steps.')
flags.DEFINE_integer('num_episodes_rs', 2, 'Number of training episodes.')
flags.DEFINE_string('traject_dir_rs', 
                    'random_search_trajectories', 
            'Directory to save the dataset.')
flags.DEFINE_bool('use_envlogger_rs', False, 'Use envlogger to log the data.')  
flags.DEFINE_string('summary_dir_rs', '.', 'Directory to save the summary.')
flags.DEFINE_string('reward_formulation_rs', 'power', 'Which reward formulation to use?')
flags.DEFINE_integer('seed', 110, 'random_search_hyperparameter')
FLAGS = flags.FLAGS
```

This function logs fitness history to csv file 


```python
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
```

This function wraps the environment in envlogger


```python
def wrap_in_envlogger(env, envlogger_dir):
    """Wraps the environment in envlogger

    Args:
        env (gym.Env): gym environment
        envlogger_dir (str): path to the directory where the data will be logged
    """
    metadata = {
        'agent_type': 'RandomSearch',
        'num_steps': FLAGS.num_steps_rs,
        'env_type': type(env).__name__,
    }
    if FLAGS.use_envlogger_rs:
        logging.info('Wrapping environment with EnvironmentLogger...')
        env = envlogger.EnvLogger(env,
                                  data_directory=envlogger_dir,
                                  max_episodes_per_file=1000,
                                  metadata=metadata)
        logging.info('Done wrapping environment with EnvironmentLogger.')
        return env
    else:
        return env


```

Main function trains the custom environment using random actions for a given number of steps and episodes 

We inititalise env by calling the custom environment wrapper. And then we setup the problem statement, which contains information about the search space and the metrics to optimize.


```python
def main(_):
    """Trains the custom environment using random actions for a given number of steps and episodes 
    """

    env = customenv_wrapper.make_custom_env(max_steps=FLAGS.num_steps_rs)
    fitness_hist = {}
    problem = vz.ProblemStatement()
    problem.search_space.select_root().add_int_param(name='num_cores', min_value = 1, max_value = 12)
    problem.search_space.select_root().add_float_param(name='freq', min_value = 0.5, max_value = 3)
    problem.search_space.select_root().add_categorical_param(name='mem_type', feasible_values =['DRAM', 'SRAM', 'Hybrid'])
    problem.search_space.select_root().add_discrete_param(name='mem_size', feasible_values=[0, 32, 64, 128, 256, 512])

    problem.metric_information.append(
        vz.MetricInformation(
            name='Reward', goal=vz.ObjectiveMetricGoal.MAXIMIZE))
```

The study configuration contains additional information, such as the algorithm to use and level of noise that we think the objective will have. To sweep through the hyperparameters of the algorithm, we access the algorithm through its designer.

NOTE - This part of the code is different for different algorithms.


```python
study_config = vz.StudyConfig.from_problem(problem)
random_designer = RandomDesigner(problem.search_space, seed = FLAGS.seed)
```

## Setting up the client
Starts a `study_client`, which can be either in **local mode (default)** or **distributed mode.**

**Local Mode:** The client has no `endpoint` set, and will implicitly create a local Vizier Service which will be shared across other clients in the same Python process. Studies will then be stored locally in a SQL database file located at `service.VIZIER_DB_PATH`.

**Distributed mode:** The service may be explicitly created, wrapped as a server in a separate process to accept requests from all other client processses. Details such as the `database_url`, `port`, `policy_factory`, etc. can be configured in the server's initializer.

All client processes (on a single machine or over multiple machines) will connect to this server via a globally specified `endpoint`.

## Client Parallelization
Regardless of whether the setup is local or distributed, we may simultaneously create multiple clients to work on the same study, useful for parallelizing evaluation workload.


```python
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
```

Define the directories for the logs to be saved.


```python
exp_name = "_num_steps_" + str(FLAGS.num_steps_rs) + "_num_episodes_" + str(FLAGS.num_episodes_rs)

    # append logs to base path
log_path = os.path.join(FLAGS.summary_dir_rs, 'random_search_logs', FLAGS.reward_formulation_rs, exp_name)

# get the current working directory and append the exp name
traject_dir = os.path.join(FLAGS.summary_dir_rs, FLAGS.traject_dir_rs, FLAGS.reward_formulation_rs, exp_name)

# check if log_path exists else create it
if not os.path.exists(log_path):
    os.makedirs(log_path)

if FLAGS.use_envlogger_rs:
    if not os.path.exists(traject_dir):
        os.makedirs(traject_dir)
env = wrap_in_envlogger(env, traject_dir)
```

## Obtaining suggestions
Start requesting suggestions from the server, for evaluating objectives. Suggestions can be made sequentially (`count=1`) or in batches (`count>1`).


```python
env.reset()
    
count = 0
suggestions = random_designer.suggest(count=FLAGS.num_steps_rs)

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
    if count == FLAGS.num_steps_rs:
        done = True

    log_fitness_to_csv(log_path, fitness_hist)
    print("Observation: ",obs)
    final_measurement = vz.Measurement({'Reward': reward})
    suggestion = suggestion.to_trial()
    suggestion.complete(final_measurement)
        
```

Calling the main function


```python
if __name__ == '__main__':
   app.run(main)
```
