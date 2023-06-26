from concurrent import futures
import grpc
import portpicker
import sys
import os

os.sys.path.insert(0, os.path.abspath('../../'))


from arch_gym.envs.custom_env import CustomEnv
from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service import vizier_server
from vizier.service import vizier_service_pb2_grpc
from vizier import algorithms as vza
from vizier.service.pyvizier import Algorithm

# from random_train import RandomDesigner
from absl import flags
# from dse_framework import dse



flags.DEFINE_integer('num_steps', 4, 'Number of training steps')
flags.FLAGS(sys.argv)
steps = flags.FLAGS.num_steps
print(steps)
env = CustomEnv()
observation = env.reset()

problem = vz.ProblemStatement()
problem.search_space.select_root().add_int_param(name='num_cores', min_value = 1, max_value = 12)
problem.search_space.select_root().add_float_param(name='freq', min_value = 0.5, max_value = 3)
problem.search_space.select_root().add_categorical_param(name='mem_type', feasible_values =['DRAM', 'SRAM', 'Hybrid'])
problem.search_space.select_root().add_discrete_param(name='mem_size', feasible_values=[0, 32, 64, 128, 256, 512])

problem.metric_information.append(
    vz.MetricInformation(
        name='energy', goal=vz.ObjectiveMetricGoal.MINIMIZE))

problem.metric_information.append(
    vz.MetricInformation(
        name='area', goal=vz.ObjectiveMetricGoal.MINIMIZE))

problem.metric_information.append(
    vz.MetricInformation(
        name='latency', goal=vz.ObjectiveMetricGoal.MINIMIZE))



study_config = vz.StudyConfig.from_problem(problem)
study_config.algorithm = vz.Algorithm.RANDOM_SEARCH
# RandomDesigner(search_space=problem.search_space)

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


suggestions = study.suggest(count=steps)
for suggestion in suggestions:
  num_cores = float(suggestion.parameters['num_cores'])
  freq = float(suggestion.parameters['freq'])
  mem_type_dict = {'SRAM':0, 'DRAM':1, 'Hybrid':2}
  mem_type = float(mem_type_dict[suggestion.parameters['mem_type']])
  mem_size = float(suggestion.parameters['mem_size'])
  action = {"num_cores":num_cores, "freq": freq, "mem_type":mem_type, "mem_size": mem_size}
  print("Suggested Parameters for num_cores, freq, mem_type, mem_size are :", num_cores, freq, mem_type, mem_size)
  obs, reward, done, info = (env.step(action))
  print(obs)
  #   final_measurement = vz.Measurement({'energy': energy, 'area': area, 'latency': latency})
#   suggestion.complete(final_measurement)


for optimal_trial in study.optimal_trials():
  optimal_trial = optimal_trial.materialize()
  print("Optimal Trial Suggestion and Objective:", optimal_trial.parameters,
        optimal_trial.final_measurement)
