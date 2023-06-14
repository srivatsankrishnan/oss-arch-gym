import subprocess
import numpy as np
from itertools import product
import sys
import os
import yaml
import json
from datetime import date, datetime
os.sys.path.insert(0, os.path.abspath('../../'))
from configs import arch_gym_configs

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string('algo', 'random_walk', 'Which Algorithm to run')

def update_sim_configs(sim_config, dram_sys_workload):
    # read a json file
    with open(sim_config, 'r') as f:
        data = json.load(f)
    # new value
    data['simulation']['tracesetup'][0]['name'] = dram_sys_workload
    
    # write back the json data to sim_config file
    with open(sim_config, 'w') as f:
        json.dump(data, f, indent=4)

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
    # Dict for storing the time to complete data
    time_to_complete = {}

    if ("algo" in task.keys()):
        if (task["algo"] in ["ga", "bo", "aco", "random_walk", "rl"]):
            if (task["algo"] == "aco"):
                algo = "aco"
            elif (task["algo"] == "bo"):
                algo = "bo"
            elif (task["algo"] == "ga"):
                algo = "ga"
            elif (task["algo"] == "rl"):
                algo = "rl"
            elif(task["algo"] == "random_walk"):
                algo = "random_walk"
        else:
            print("This algorithm is not supported.")
            exit(0)
    else: 
        print("Need to provide an algorithm.")
        exit(0)
    
    workload = task['workload']
    
    # Get date and time and construct unique directory name to store outputs
    now = datetime.now()
    now = str(now)
    date = now.split(" ")[0]
    time = now.split(" ")[1]

    if(algo == "ga"):
        prob_mut = task["prob_mut"]
        num_agents = task["num_agents"]
        num_iter = task["num_iter"]
        ga_hyperparams = {"prob_mut": prob_mut, "num_agents": num_agents}
        unqiue_ids = [algo, workload, str(prob_mut), str(num_agents), date, time]
    elif(algo == "bo"):
        rand_state = task["rand_state"]
        num_iter = task["num_iter"]
        unqiue_ids = [algo, workload, str(rand_state), str(num_iter), date, time]
    elif(algo == "aco"):
        evaporation = task["evaporation"]
        ant_count = task["ant_count"]
        greediness = task["greediness"]
        depth = task["depth"]
        unqiue_ids = [algo, workload, str(evaporation), str(ant_count), str(greediness), str(depth), date, time]
    elif (algo == "random_walk"):
        num_iter = task["num_iter"]
        unqiue_ids = [algo, workload, str(num_iter), date, time]
    
    # TODO: Create unique id with hyper params 
    #dir_name = '_'.join(unqiue_ids)
    #os.mkdir(dir_name)

    # Run the experiments

    # Run algo 
    if algo == "aco":
        aco_agent_config_file = os.path.join(
                                arch_gym_configs.proj_root_path,
                                "settings",
                                arch_gym_configs.aco_config)
        aco_hyperparams = {"evaporation": evaporation,
                            "ant_count": ant_count,
                            "greediness": greediness,
                            "depth": depth}
        update_aco_agent_configs(aco_agent_config_file, aco_hyperparams)
          
        print("train_aco_DRAMSys")
        cmd = "python train_aco_DRAMSys.py " + \
            "--evaporation=" + str(evaporation) + " " \
            "--workload=" + str(workload) + " " \
            "--ant_count=" + str(ant_count) + " "\
            "--greediness=" + str(greediness) + " "\
            "--depth="+ str(depth)
        print("Shell Command", cmd)
        
        # time the run
        start = datetime.now()
        os.system(cmd)
        end = datetime.now()

        identifier = [algo, workload, str(evaporation), str(ant_count), str(greediness), str(depth)]
        key = '_'.join(identifier)
        time_to_complete[key] = (end - start).total_seconds()

        # open and write time to complete to file
        time_to_complete_path = os.getcwd()
        with open(time_to_complete_path + "/time_to_complete.txt", "a") as f:
            for key in time_to_complete:
                f.write(key + ": " + str(time_to_complete[key]) + "\n")
    # Run BO
    elif algo == "bo":
        print("train_bo_DRAMSys")
        cmd = "python train_bo_DRAMSys.py " + \
            "--workload=" + str(workload) + " " \
            "--random_state=" + str(rand_state) + " "\
            "--num_iter=" + str(num_iter)
        print("Shell Command", cmd)

        # time the run
        start = datetime.now()
        os.system(cmd)
        end = datetime.now()

        identifier = [algo, workload, str(rand_state), str(num_iter)]
        key = '_'.join(identifier)
        time_to_complete[key] = (end - start).total_seconds()

        # open and write time to complete to file
        time_to_complete_path = os.getcwd()
        with open(time_to_complete_path + "/time_to_complete.txt", "a") as f:
            for key in time_to_complete:
                f.write(key + ": " + str(time_to_complete[key]) + "\n")
    # Run Random Walk
    elif algo == "random_walk":
        print("train_randomwalker_DRAMSys")
        cmd = "python train_randomwalker_DRAMSys.py " + \
            "--workload=" + str(workload) + " " \
            "--num_steps=" + str(num_iter)
        print("Shell Command", cmd)
            
        # time the run
        start = datetime.now()
        os.system(cmd)
        end = datetime.now()

        identifier = [algo, workload, str(num_iter)]
        key = '_'.join(identifier)
        time_to_complete[key] = (end - start).total_seconds()

        # open and write time to complete to file
        time_to_complete_path = os.getcwd()
        with open(time_to_complete_path + "/time_to_complete.txt", "a") as f:
            for key in time_to_complete:
                f.write(key + ": " + str(time_to_complete[key]) + "\n")
    # Run GA
    elif algo == "ga":
        print("train_ga_DRAMSys")
        cmd = "python train_ga_DRAMEnv.py " + \
            "--workload=" + str(workload) + " " \
            "--num_iter=" + str(num_iter) + " " \
            "--prob_mutation=" + str(prob_mut) + " "\
            "--num_agents=" + str(num_agents)
        print("Shell Command", cmd)
            
        # time the run
        start = datetime.now()
        os.system(cmd)
        end = datetime.now()

        identifier = [algo, workload, str(prob_mut),str(num_iter),str(num_agents)]
        key = '_'.join(identifier)
        time_to_complete[key] = (end - start).total_seconds()

        # open and write time to complete to file
        time_to_complete_path = os.getcwd()
        with open(time_to_complete_path + "/time_to_complete.txt", "a") as f:
            for key in time_to_complete:
                f.write(key + ": " + str(time_to_complete[key]) + "\n")
    else:
        print("Unsupport task formulation!!")
        raise NotImplementedError

def main(_):

    taskList = []
    sim_config = arch_gym_configs.sim_config
    
    if FLAGS.algo == "aco":
        for workload in arch_gym_configs.dram_sys_workload:
            for ant_count in arch_gym_configs.ant_count:
                for greediness in arch_gym_configs.greediness:
                    for evaporation in arch_gym_configs.evaporation:
                        for depth in arch_gym_configs.depth:
                            task = {"algo": FLAGS.algo, "workload": workload, "evaporation": evaporation, "ant_count": ant_count, "greediness": greediness, "depth": depth}
                            taskList.append(task)
    elif FLAGS.algo == "bo":
        for workload in arch_gym_configs.dram_sys_workload:
            for rand_state in arch_gym_configs.rand_state_bo:
                for num_iter in arch_gym_configs.num_iter_bo:
                    task = {"algo": FLAGS.algo, "workload": workload, "rand_state": rand_state, "num_iter": num_iter}
                    taskList.append(task)
    elif FLAGS.algo == "random_walk":
        for workload in arch_gym_configs.dram_sys_workload:
            for num_iter in arch_gym_configs.num_steps:
                task = {"algo": FLAGS.algo, "workload": workload, "num_iter": num_iter}
                taskList.append(task)
    elif FLAGS.algo == "ga":
        for workload in arch_gym_configs.dram_sys_workload:
            for num_agents in arch_gym_configs.num_agents:
                for num_iter in arch_gym_configs.num_iter_ga:
                    for prob_mut in arch_gym_configs.prob_mut:
                        task = {"algo": FLAGS.algo, "workload": workload, "num_agents": num_agents, "num_iter": num_iter, "prob_mut": prob_mut}
                        taskList.append(task)
    else:
        print(" Algorithm not supported!!")
        raise NotImplementedError
    print(taskList)
    
    for each_task in taskList:
        # update the workload in DRAMSys simulator
        update_sim_configs(sim_config, each_task["workload"])
        run_task(each_task)
    
if __name__ == '__main__':
   app.run(main)
