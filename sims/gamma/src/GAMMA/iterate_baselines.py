import subprocess
import numpy as np
from itertools import product
import sys
import os
import yaml
import json
from datetime import date, datetime
os.sys.path.insert(0, os.path.abspath('../../../../'))
from configs import arch_gym_configs

from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string('algo', 'ga', 'Which Algorithm to run')
flags.DEFINE_string('workload', 'resnet18', 'Which workload to run')
flags.DEFINE_string('summary_dir', '', 'Directory to store the summary')
flags.DEFINE_integer('num_iter', 10, 'Number of iterations')
flags.DEFINE_string('reward_formulation', 'latency', 'Reward formulation to use')


# BO
flags.DEFINE_integer('rand_state', 0, 'Random state')

# GAa
flags.DEFINE_integer('num_agents', 10, 'Number of agents')
flags.DEFINE_float('prob_mutation', 0.1, 'Probability of mutation.')

# ACO
flags.DEFINE_integer('ant_count', 2, 'Number of ants')
flags.DEFINE_float('evaporation', 0.25, 'Evaporation rate')
flags.DEFINE_float('greediness', 0.25, 'Greedy rate')

def update_aco_agent_configs(agent_config, aco_hyperparams):
    print("Agent Config File", agent_config)
    print("Agent Hyperparams", aco_hyperparams)
    
    # read the yaml file
    with open(agent_config, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    data['DeepSwarm']['max_depth'] = aco_hyperparams["depth"]
    data['DeepSwarm']['aco']['ant_count'] = aco_hyperparams["ant_count"]
    data['DeepSwarm']['aco']['greediness'] = aco_hyperparams["greediness"]
    data['DeepSwarm']['aco']['pheromone']['evaporation'] = aco_hyperparams["evaporation"]
    
    # write back the yaml data to agent_config file
    with open(agent_config, "w") as stream:
        yaml.dump(data, stream, default_flow_style=True)

def run_task(task):

    if ("algo" in task.keys()):
        if (task["algo"] in ["ga", "bo", "aco", "rw", "rl"]):
            if (task["algo"] == "aco"):
                algo = "aco"
            elif (task["algo"] == "bo"):
                algo = "bo"
            elif (task["algo"] == "ga"):
                algo = "ga"
            elif (task["algo"] == "rl"):
                algo = "rl"
            elif(task["algo"] == "rw"):
                algo = "rw"
        else:
            print("This algorithm is not supported.")
            exit(0)
    else: 
        print("Need to provide an algorithm.")
        exit(0)
    
    workload = task['workload']
    if(algo == "ga"):
        prob_mut = task["prob_mut"]
        num_agents = task["num_agents"]
        num_iter = task["num_iter"]
        summary_dir = task["summary_dir"]
        reward_formulation = task["reward_formulation"]
        unqiue_ids = [algo, workload, str(prob_mut), str(num_agents)]
    elif (algo == "rw"):
        num_iter = task["num_iter"]
        summary_dir = task["summary_dir"]
        reward_formulation = task["reward_formulation"]
        unqiue_ids = [algo, workload]
    elif (algo == "bo"):
        num_iter = task["num_iter"]
        rand_state = task["rand_state"]
        summary_dir = task["summary_dir"]
        reward_formulation = task["reward_formulation"]
        unqiue_ids = [algo, workload, str(rand_state)]
    elif (algo == "aco"):
        num_iter = task["num_iter"]
        ant_count = task["ant_count"]
        evaporation = task["evaporation"]
        greediness = task["greediness"]
        summary_dir = task["summary_dir"]
        reward_formulation = task["reward_formulation"]
        depth = task["num_iter"]
        unqiue_ids = [algo, workload, str(ant_count), str(evaporation), str(greediness)]
    else:
        raise NotImplementedError

    if algo == "ga":
        print("train_ga_DRAMSys")
        cmd = "python train_ga_maestro.py " + \
            "--workload=" + str(workload) + " " \
            "--num_steps=" + str(num_iter) + " " \
            "--prob_mutation=" + str(prob_mut) + " "\
            "--num_agents=" + str(num_agents) + " "\
            "--summary_dir=" + str(summary_dir) + " "\
            "--reward_formulation=" + str(reward_formulation) 
        print("Shell Command", cmd)
        
    elif algo == "rw":
        print("train_randomwalker_maestro")
        cmd = "python train_randomwalker_maestro.py " + \
            "--workload=" + str(workload) + " " \
            "--num_steps=" + str(num_iter) + " " \
            "--summary_dir=" + str(summary_dir) + " "\
            "--reward_formulation=" + str(reward_formulation)
        print("Shell Command", cmd)
    elif algo == "bo":
        print("train_bo_maestro")
        cmd = "python train_bo_maestro.py " + \
            "--workload=" + str(workload) + " " \
            "--num_iter=" + str(num_iter) + " " \
            "--random_state=" + str(rand_state) + " " \
            "--summary_dir=" + str(summary_dir) + " "\
            "--reward_formulation=" + str(reward_formulation)
        print("Shell Command", cmd)
    elif algo == "aco":
        aco_agent_config_file = os.path.join(
                                arch_gym_configs.proj_root_path,
                                "settings",
                                arch_gym_configs.aco_config)
        aco_hyperparams = {"evaporation": evaporation,
                            "ant_count": ant_count,
                            "greediness": greediness,
                            "depth": depth}
        update_aco_agent_configs(aco_agent_config_file, aco_hyperparams)

        print("train_aco_maestro")
        cmd = "python train_aco_maestro.py " + \
            "--workload=" + str(workload) + " " \
            "--depth=" + str(num_iter) + " " \
            "--ant_count=" + str(ant_count) + " " \
            "--evaporation=" + str(evaporation) + " " \
            "--greediness=" + str(greediness) + " " \
            "--summary_dir=" + str(summary_dir) + " "\
            "--reward_formulation=" + str(reward_formulation)
        print("Shell Command", cmd)
    else:
        raise NotImplementedError
    
    # run the command
    os.system(cmd)
  
def generate_tasks():
    algorithms = ['ga', 'bo', 'rw']
    total_iterations = [3, 4]
    reward_formulations = ['latency', 'energy', 'area']
    workloads = ['resnet18', 'mobilenet_v2', 'alexnet', 'vgg16']

    tasks = []

    for algo in algorithms:
        for num_iter in total_iterations:
            for reward_formulation in reward_formulations:
                for workload in workloads:
                    if algo == 'ga':
                        num_agents_values = [16, 32, 64]
                        prob_mutation_values = [0.01, 0.05]
                        for num_agents, prob_mutation in product(num_agents_values, prob_mutation_values):
                            task = {
                                'algo': algo,
                                'workload': workload,
                                'num_iter': num_iter,
                                'summary_dir': FLAGS.summary_dir,
                                'reward_formulation': reward_formulation,
                                'num_agents': num_agents,
                                'prob_mut': prob_mutation
                            }
                            tasks.append(task)
                    elif algo == 'bo':
                        rand_state_values = [0, 1, 2, 3, 4, 5]
                        for rand_state in rand_state_values:
                            task = {
                                'algo': algo,
                                'workload': workload,
                                'num_iter': num_iter,
                                'summary_dir': FLAGS.summary_dir,
                                'reward_formulation': reward_formulation,
                                'rand_state': rand_state
                            }
                            tasks.append(task)
                    elif algo == 'aco':
                        ant_count_values = [16, 32]
                        evaporation_values = [0.1, 0.2]
                        greediness_values = [0.1, 0.2]
                        for ant_count, evaporation, greediness in product(ant_count_values, evaporation_values, greediness_values):
                            task = {
                                'algo': algo,
                                'workload': workload,
                                'num_iter': num_iter,
                                'summary_dir': FLAGS.summary_dir,
                                'reward_formulation': reward_formulation,
                                'ant_count': ant_count,
                                'evaporation': evaporation,
                                'greediness': greediness
                            }
                            tasks.append(task)
                    else:
                        task = {
                            'algo': algo,
                            'workload': workload,
                            'num_iter': num_iter,
                            'summary_dir': FLAGS.summary_dir,
                            'reward_formulation': reward_formulation
                        }
                        tasks.append(task)

    return tasks

def main(_):
    
    tasks = generate_tasks()

    for each_task in tasks:
        run_task(each_task)


if __name__ == '__main__':
   app.run(main)
