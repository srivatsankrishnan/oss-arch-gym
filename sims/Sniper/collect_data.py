from enum import unique
import os
import sys
os.sys.path.insert(0, os.path.abspath('../../'))
from configs import arch_gym_configs
from datetime import date, datetime
from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string('algo', 'bo', 'Which Algorithm to run')

def run_task(task):
    if ("algo" in task.keys()):
        if (task["algo"] in ["ga", "bo", "aco"]):
            if (task["algo"] == "aco"):
                algo = "aco"
            elif (task["algo"] == "bo"):
                algo = "bo"
            elif (task["algo"] == "ga"):
                algo = "ga"
            elif (task["algo"] == "rl"):
                algo = "rl"
        else:
            print("This algorithm is not supported.")
            exit(0)
    else: 
        print("Need to provide an algorithm.")
        exit(0)

    if ("workload" in task.keys()):
        if (task["workload"] in ["600", "602", "605", "620", "625", "641", "648", "657", "619", "638", "644"]):
            if (task["workload"] == "600"):
                workload = "600"
            elif (task["workload"] == "602"):
                workload = "602"
            elif (task["workload"] == "605"):
                workload = "605"
            elif (task["workload"] == "620"):
                workload = "620"
            elif (task["workload"] == "625"):
                workload = "625"
            elif (task["workload"] == "641"):
                workload = "641"
            elif (task["workload"] == "648"):
                workload = "648"
            elif (task["workload"] == "657"):
                workload = "657"
            elif (task["workload"] == "619"):
                workload = "619"
            elif (task["workload"] == "638"):
                workload = "638"
            elif (task["workload"] == "644"):
                workload = "644"
        else:
            print("This workload is not supported.")
            exit(0)
    else: 
        print("Need to provide a workload.")
        exit(0)

    # Get date and time and construct unique directory name to store outputs
    now = datetime.now()
    now = str(now)
    date = now.split(" ")[0]
    time = now.split(" ")[1]
    if(algo == "ga"):
        prob_mut = task["prob_mut"]
        num_agents = task["num_agents"]
        ga_hyperparams = {"prob_mut": prob_mut, "num_agents": num_agents}
        unqiue_ids = [algo, workload, str(prob_mut), str(num_agents), date, time]
    elif(algo == "bo"):
        rand_state = task["rand_state"]
        num_iter = task["num_iter"]
        unqiue_ids = [algo, workload, str(rand_state), str(num_iter), date, time]
    
    # TODO: Create unique id with hyper params 
    dir_name = '_'.join(unqiue_ids)
    os.mkdir(dir_name)

    # Write sniper workload var to arch gym configs 
    file = open("../../configs/arch_gym_configs.py", "r")
    replacement = ""
    for line in file:
        line = line.strip()
        if 'spec_workload = ' in line: 
            changes = 'spec_workload = ' + '"' + workload + '"'
            replacement = replacement + changes + "\n"
        else:
            replacement = replacement + line + "\n"

    file.close()
    fout = open("../../configs/arch_gym_configs.py", "w")
    fout.write(replacement)
    fout.close()
    
    time_to_complete = {}
    # Run algo 
    if algo == "aco":
        print("train_aco_Sniper")
        os.system("python train_aco_Sniper.py " + dir_name)
    elif algo == "ga":
        print("train_ga_Sniper")
        # time the runs
        start = datetime.now()
        os.system("python train_ga_Sniper.py " + dir_name)
        end = datetime.now()
        identifier = [algo, str(workload), str(prob_mut), str(num_agents)]
        key = '_'.join(identifier)
        time_to_complete[key] = (end - start).total_seconds()
    elif algo == "bo":
        print("train_bo_Sniper")
        cmd = "python train_bo_Sniper.py " + \
        "--random_state " + str(task["rand_state"]) + \
        " --num_iter " + str(task["num_iter"])
        print(cmd)
        start = datetime.now()
        os.system(cmd)
        end = datetime.now()
        identifier = [algo, str(workload), str(rand_state), str(num_iter)]
        key = '_'.join(identifier)
        time_to_complete[key] = (end - start).total_seconds()
    # open and write time to complete to file
    time_to_complete_path = os.getcwd()
    with open(time_to_complete_path + "/time_to_complete.txt", "a") as f:
        for key in time_to_complete:
            f.write(key + ": " + str(time_to_complete[key]) + "\n")

def main(_):
    taskList = []
    if(FLAGS.algo == "ga"):
        for workload in ["602"]: 
            for num_agent in arch_gym_configs.num_agents:
                for prob_mut in arch_gym_configs.prob_mut:
                    taskList.append({"algo": FLAGS.algo,
                    "workload": workload,
                    "prob_mut": prob_mut, 
                    "num_agents": num_agent}
                    )
    elif(FLAGS.algo == "bo"):
        for workload in ["602"]:
            for rand_state in arch_gym_configs.rand_state_bo:
                for num_iter in arch_gym_configs.num_iter_bo:
                    taskList.append({"algo": FLAGS.algo,
                    "workload": workload,
                    "rand_state": rand_state,
                    "num_iter": num_iter
                    })
    print(taskList)
    for task_el in taskList:
        run_task(task_el) 

if __name__ == '__main__':
   app.run(main)
