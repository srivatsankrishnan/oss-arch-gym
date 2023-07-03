from concurrent import futures
import grpc
import portpicker

from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service import vizier_server
from vizier.service import vizier_service_pb2_grpc

#from dse_framework import dse

from gym_env import SimpleArch  

NUM_TRIALS = 1000

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

suggestions = study.suggest(count=NUM_TRIALS)

env = SimpleArch()
env.reset()
count = 1

for suggestion in suggestions:
  num_cores = suggestion.parameters['num_cores']
  freq =  suggestion.parameters['freq']
  mem_type =  suggestion.parameters['mem_type']
  mem_size =  suggestion.parameters['mem_size']

  print("\n")
  print(count)
  print('Suggested Parameters (num_cores, freq, mem_type, mem_size):', num_cores, freq, mem_type, mem_size)
  #action = env.action_space.sample()
  action = [num_cores, freq, mem_type, mem_size]
  print (f'Action: {action}')
  
  obs, reward, done, info = env.step(action)
  env.render()  #prints the observation which is (energy, area, latency)
  print(f'Reward: {reward}')

  #energy, area, latency = env.energy, env.area, env.latency
  #final_measurement = vz.Measurement({'energy': energy, 'area':area, 'latency': latency})
  final_measurement = vz.Measurement({'Reward': reward})


  suggestion.complete(final_measurement)
  count += 1

count = 1
for optimal_trial in study.optimal_trials():
  optimal_trial = optimal_trial.materialize()
  print("\n")
  print(count)
  print("Optimal Trial Suggestion and Objective:", optimal_trial.parameters,
        optimal_trial.final_measurement)
  count += 1
