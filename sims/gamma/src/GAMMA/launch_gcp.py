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

# GA
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
  


def main(_):
    taskList = []

    if FLAGS.algo == "ga":
        task = {"algo": FLAGS.algo,
                "workload": FLAGS.workload, 
                "num_agents": FLAGS.num_agents, 
                "num_iter": FLAGS.num_iter, 
                "prob_mut": FLAGS.prob_mutation,
                'summary_dir': FLAGS.summary_dir,
                'reward_formulation': FLAGS.reward_formulation}
        taskList.append(task)
    elif FLAGS.algo == "rw":
        task = {"algo": FLAGS.algo,
                "workload": FLAGS.workload, 
                "num_iter": FLAGS.num_iter, 
                'summary_dir': FLAGS.summary_dir,
                'reward_formulation': FLAGS.reward_formulation}
        taskList.append(task)
    elif FLAGS.algo == "bo":
        task = {"algo": FLAGS.algo,
                "workload": FLAGS.workload, 
                "num_iter": FLAGS.num_iter, 
                "rand_state": FLAGS.rand_state,
                'summary_dir': FLAGS.summary_dir,
                'reward_formulation': FLAGS.reward_formulation}
        taskList.append(task)
    elif FLAGS.algo == "aco":
        task = {"algo": FLAGS.algo,
                "workload": FLAGS.workload, 
                "num_iter": FLAGS.num_iter, 
                "ant_count": FLAGS.ant_count,
                "evaporation": FLAGS.evaporation,
                "greediness": FLAGS.greediness,
                'summary_dir': FLAGS.summary_dir,
                'reward_formulation': FLAGS.reward_formulation}
        taskList.append(task)

    else:
        raise NotImplementedError


    for each_task in taskList:
        # update the workload in DRAMSys simulator
        run_task(each_task)


if __name__ == '__main__':
   app.run(main)
