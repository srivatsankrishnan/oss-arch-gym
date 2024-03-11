import subprocess
import numpy as np
from itertools import product
import sys
import os
import yaml
import json
import socket
from datetime import date, datetime
os.sys.path.insert(0, os.path.abspath('../../'))
from configs import arch_gym_configs
from sims.AstraSim.frontend.parse_parameter_specs import parse_csv_to_knobs

from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string('experiment', './experiments_files/experiment.yml', 'yaml with paths to all experiment files')
flags.DEFINE_string('summary_dir', './all_logs', 'Directory to store the summary')
flags.DEFINE_integer('num_iter', 30, 'Number of iterations')
flags.DEFINE_string('knobs', 'astrasim_220_example/knobs.py', "path to knobs spec file")
flags.DEFINE_string('network', 'astrasim_220_example/network_input.yml', "path to network input file")
flags.DEFINE_string('system', 'astrasim_220_example/system_input.json', "path to system input file")
flags.DEFINE_string('workload_file', 'astrasim_220_example/workload_cfg.json', "path to workload input file")
flags.DEFINE_string('reward_formulation', 'cycles', 'Reward formulation to use')
flags.DEFINE_string('algo', 'ga', 'Which Algorithm to run')
flags.DEFINE_string('workload', 'resnet18', 'Which workload to run')
flags.DEFINE_bool('congestion_aware', True, "astra-sim congestion aware or not")
# flags.DEFINE_string('parameter_specs', 'workload_validation_parameters.csv', "Parameter specs file")

# BO
flags.DEFINE_integer('rand_state', 0, 'Random state')

# GA
flags.DEFINE_integer('num_agents', 10, 'Number of agents')
flags.DEFINE_float('prob_mutation', 0.1, 'Probability of mutation.')

# ACO
flags.DEFINE_integer('ant_count', 2, 'Number of ants')
flags.DEFINE_float('evaporation', 0.25, 'Evaporation rate')
flags.DEFINE_float('greediness', 0.25, 'Greedy rate')
flags.DEFINE_float('decay', 0.1, 'Decay rate for pheromone.')
flags.DEFINE_float('start', 0.1, 'Start value for pheromone.')

# RL
flags.DEFINE_string('rl_algo', 'ppo', 'RL algorithm.')
flags.DEFINE_string('rl_form', 'sa1', 'RL form.')
flags.DEFINE_string('reward_form', 'both', 'Reward form.')
flags.DEFINE_string('reward_scale', 'false', 'Scale reward.')
flags.DEFINE_integer('num_steps', 100, 'Number of training steps.')
flags.DEFINE_integer('eval_every', 50, 'Number of evaluation steps.')
flags.DEFINE_integer('eval_episodes', 1, 'Number of evaluation episode.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate.')
flags.DEFINE_float('entropy_cost', 0.1, 'Entropy cost.')
flags.DEFINE_float('ppo_clipping_epsilon', 0.2, 'PPO clipping epsilon.')
flags.DEFINE_bool('clip_value', False, 'Clip value.')
flags.DEFINE_string('summarydir', './all_logs', 'Directory to save summaries.')
flags.DEFINE_string('envlogger_dir', 'trajectory', 'Directory to save envlogger.')
flags.DEFINE_bool('use_envlogger', False, 'Use envlogger.')
flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a '
    'distributed way (the default is a single-threaded agent)')
flags.DEFINE_integer("params_scaling", 1, "Number of training steps")

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
    data['DeepSwarm']['aco']['pheromone']['decay'] = aco_hyperparams['decay']
    data['DeepSwarm']['aco']['pheromone']['start'] = aco_hyperparams['start']
    
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
    

    ### run the specs file ###
    # parameter_specs = os.path.join("frontend", task["parameter_specs"])
    # print("Parameter Specs", parameter_specs)
    # parse_csv_to_knobs(parameter_specs)


    workload = task['workload']
    if(algo == "ga"):
        prob_mut = task["prob_mut"]
        num_agents = task["num_agents"]
        num_iter = task["num_iter"]
        summary_dir = task["summary_dir"]
        reward_formulation = task["reward_formulation"]
        use_envlogger = task["use_envlogger"]
        knobs = task["knobs"]
        congestion_aware = task["congestion_aware"]
        network = task["network"]
        system = task["system"]
        workload_file = task["workload_file"]
        unqiue_ids = [algo, workload, str(prob_mut), str(num_agents)]
    elif (algo == "rw"):
        num_steps = task["num_steps"]
        use_envlogger = task["use_envlogger"]
        summary_dir = task["summary_dir"]
        reward_formulation = task["reward_formulation"]
        unqiue_ids = [algo, workload]
        knobs = task["knobs"]
        congestion_aware = task["congestion_aware"]
        network = task["network"]
        system = task["system"]
        workload_file = task["workload_file"]
    elif (algo == "bo"):
        num_iter = task["num_iter"]
        rand_state = task["rand_state"]
        summary_dir = task["summary_dir"]
        reward_formulation = task["reward_formulation"]
        use_envlogger = task["use_envlogger"]
        knobs = task["knobs"]
        congestion_aware = task["congestion_aware"]
        network = task["network"]
        system = task["system"]
        workload_file = task["workload_file"]
        unqiue_ids = [algo, workload, str(rand_state)]
    elif (algo == "aco"):
        num_iter = task["num_iter"]
        ant_count = task["ant_count"]
        evaporation = task["evaporation"]
        greediness = task["greediness"]
        decay = task["decay"]
        start = task["start"]
        summary_dir = task["summary_dir"]
        reward_formulation = task["reward_formulation"]
        use_envlogger = task["use_envlogger"]
        depth = task["num_iter"]
        knobs = task["knobs"]
        congestion_aware = task["congestion_aware"]
        network = task["network"]
        system = task["system"]
        workload_file = task["workload_file"]
        unqiue_ids = [algo, workload, str(ant_count), str(evaporation), str(greediness), str(decay), str(start)]
    elif (algo == "rl"):
        workload = task["workload"]
        rl_algo = task["rl_algo"]
        rl_form = task["rl_form"]
        reward_form = task["reward_form"]
        reward_scale = task["reward_scale"]
        num_steps = task["num_steps"]
        eval_every = task["eval_every"]
        eval_episodes = task["eval_episodes"]
        seed = task["seed"]
        learning_rate = task["learning_rate"]
        entropy_cost = task["entropy_cost"]
        ppo_clipping_epsilon = task["ppo_clipping_epsilon"]
        clip_value = task["clip_value"]
        summarydir = task["summarydir"]
        envlogger_dir = task["envlogger_dir"]
        use_envlogger = task["use_envlogger"]
        run_distributed = task["run_distributed"]
        params_scaling = task["params_scaling"]
        knobs = task["knobs"]
        congestion_aware = task["congestion_aware"]
        network = task["network"]
        system = task["system"]
        workload_file = task["workload_file"]
        unique_ids = [algo, workload]
    else:
        raise NotImplementedError

    if algo == "ga":
        print("train_ga_astra_sim")
        cmd = "python trainGAAstraSim.py " + \
            "--workload=" + str(workload) + " " \
            "--num_steps=" + str(num_iter) + " " \
            "--prob_mutation=" + str(prob_mut) + " "\
            "--num_agents=" + str(num_agents) + " "\
            "--summary_dir=" + str(summary_dir) + " "\
            "--reward_formulation=" + str(reward_formulation) + " "\
            "--use_envlogger=" + str(use_envlogger) + " "\
            "--knobs=" + str(knobs) + " " \
            "--congestion_aware=" + str(congestion_aware) + " " \
            "--network=" + str(network) + " " \
            "--system=" + str(system) + " " \
            "--workload_file=" + str(workload_file) + " "
        print("Shell Command", cmd)
        
    elif algo == "rw":
        print("train_randomwalker_astra_sim")
        cmd = "python trainRandomWalkerAstraSim.py " + \
            "--workload=" + str(workload) + " " \
            "--num_steps=" + str(num_steps) + " " \
            "--use_envlogger=" + str(use_envlogger) + " " \
            "--summary_dir=" + str(summary_dir) + " " \
            "--reward_formulation=" + str(reward_formulation) + " " \
            "--knobs=" + str(knobs) + " " \
            "--congestion_aware=" + str(congestion_aware) + " " \
            "--network=" + str(network) + " " \
            "--system=" + str(system) + " " \
            "--workload_file=" + str(workload_file) + " "
        print("Shell Command", cmd)
    elif algo == "bo":
        print("train_bo_astra_sim")
        cmd = "python train_bo_AstraSim.py " + \
            "--workload=" + str(workload) + " " \
            "--num_iter=" + str(num_iter) + " " \
            "--random_state=" + str(rand_state) + " " \
            "--summary_dir=" + str(summary_dir) + " " \
            "--reward_formulation=" + str(reward_formulation) + " "\
            "--use_envlogger=" + str(use_envlogger) + " "\
            "--knobs=" + str(knobs) + " " \
            "--congestion_aware=" + str(congestion_aware) + " " \
            "--network=" + str(network) + " " \
            "--system=" + str(system) + " " \
            "--workload_file=" + str(workload_file) + " "
        print("Shell Command", cmd)
    elif algo == "aco":
        aco_agent_config_file = os.path.join(
                                arch_gym_configs.proj_root_path,
                                "settings",
                                arch_gym_configs.aco_config)
        aco_hyperparams = {"evaporation": evaporation,
                            "ant_count": ant_count,
                            "greediness": greediness,
                            "depth": depth, 
                            "decay": decay,
                            "start": start}
        update_aco_agent_configs(aco_agent_config_file, aco_hyperparams)

        print("train_aco_astra_sim")
        cmd = "python trainACOAstraSim.py " + \
            "--workload=" + str(workload) + " " \
            "--depth=" + str(num_iter) + " " \
            "--ant_count=" + str(ant_count) + " " \
            "--evaporation=" + str(evaporation) + " " \
            "--greediness=" + str(greediness) + " " \
            "--decay=" + str(decay) + " " \
            "--start=" + str(start) + " " \
            "--summary_dir=" + str(summary_dir) + " "\
            "--reward_formulation=" + str(reward_formulation) + " "\
            "--use_envlogger=" + str(use_envlogger) + " "\
            "--knobs=" + str(knobs) + " " \
            "--congestion_aware=" + str(congestion_aware) + " " \
            "--network=" + str(network) + " " \
            "--system=" + str(system) + " " \
            "--workload_file=" + str(workload_file) + " "
        print("Shell Command", cmd)
    elif algo == "rl":
        print("train_rl_astra_sim")
        cmd = "python trainSingleAgentAstraSim.py " + \
            "--workload=" + str(workload) + " " \
            "--rl_algo=" + str(rl_algo) + " " \
            "--rl_form=" + str(rl_form) + " " \
            "--reward_form=" + str(reward_form) + " " \
            "--reward_scale=" + str(rl_algo) + " " \
            "--num_steps=" + str(num_steps) + " " \
            "--eval_every=" + str(eval_every) + " " \
            "--eval_episodes=" + str(eval_episodes) + " " \
            "--seed=" + str(seed) + " " \
            "--learning_rate=" + str(learning_rate) + " "\
            "--entropy_cost=" + str(entropy_cost) + " " \
            "--ppo_clipping_epsilon=" + str(ppo_clipping_epsilon) + " " \
            "--clip_value=" + str(clip_value) + " " \
            "--summarydir=" + str(summarydir) + " " \
            "--envlogger_dir=" + str(envlogger_dir) + " " \
            "--use_envlogger=" + str(use_envlogger) + " " \
            "--run_distributed=" + str(run_distributed) + " " \
            "--params_scaling=" + str(params_scaling) + " " \
            "--knobs=" + str(knobs) + " " \
            "--congestion_aware=" + str(congestion_aware) + " " \
            "--network=" + str(network) + " " \
            "--system=" + str(system) + " " \
            "--workload_file=" + str(workload_file) + " "
        print("Shell Command", cmd)
    else:
        raise NotImplementedError
    
    # run the command
    os.system(cmd)
  


def main(_):    
    taskList = []

    experiment_file = FLAGS.experiment
    # parse experiment.yaml
    with open(experiment_file, "r") as stream:
        try:
            experiment_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    FLAGS.algo = experiment_data["ALGORITHM"]
    FLAGS.num_iter = experiment_data["STEPS"]
    FLAGS.knobs = experiment_data["KNOBS"]
    FLAGS.network = experiment_data["NETWORK"]
    FLAGS.system = experiment_data["SYSTEM"]
    FLAGS.workload_file = experiment_data["WORKLOAD"]

    # append hostname to summary dir
    hostname = socket.gethostname()
    FLAGS.summary_dir = FLAGS.summary_dir + "/" + hostname

    if FLAGS.algo == "ga":
        task = {"algo": FLAGS.algo,
                "workload": FLAGS.workload, 
                "num_agents": FLAGS.num_agents, 
                "num_iter": FLAGS.num_iter, 
                "prob_mut": FLAGS.prob_mutation,
                'summary_dir': FLAGS.summary_dir,
                'reward_formulation': FLAGS.reward_formulation,
                'use_envlogger': FLAGS.use_envlogger,
                "knobs": FLAGS.knobs,
                "congestion_aware": FLAGS.congestion_aware,
                # 'parameter_specs': FLAGS.parameter_specs,
                "network": FLAGS.network,
                "system": FLAGS.system,
                "workload_file": FLAGS.workload_file}
        taskList.append(task)
    elif FLAGS.algo == "rw":
        task = {"algo": FLAGS.algo,
                "workload": FLAGS.workload, 
                "num_steps": FLAGS.num_steps, 
                "use_envlogger": FLAGS.use_envlogger,
                'summary_dir': FLAGS.summary_dir,
                'reward_formulation': FLAGS.reward_formulation,
                "knobs": FLAGS.knobs,
                "congestion_aware": FLAGS.congestion_aware,
                "network": FLAGS.network,
                "system": FLAGS.system,
                "workload_file": FLAGS.workload_file}
        taskList.append(task)
    elif FLAGS.algo == "bo":
        task = {"algo": FLAGS.algo,
                "workload": FLAGS.workload, 
                "num_iter": FLAGS.num_iter, 
                "rand_state": FLAGS.rand_state,
                'summary_dir': FLAGS.summary_dir,
                'reward_formulation': FLAGS.reward_formulation,
                'use_envlogger': FLAGS.use_envlogger,
                "knobs": FLAGS.knobs,
                "congestion_aware": FLAGS.congestion_aware,
                "network": FLAGS.network,
                "system": FLAGS.system,
                "workload_file": FLAGS.workload_file}
        taskList.append(task)
    elif FLAGS.algo == "aco":
        task = {"algo": FLAGS.algo,
                "workload": FLAGS.workload, 
                "num_iter": FLAGS.num_iter, 
                "ant_count": FLAGS.ant_count,
                "evaporation": FLAGS.evaporation,
                "greediness": FLAGS.greediness,
                "decay": FLAGS.decay,
                "start": FLAGS.start,
                'summary_dir': FLAGS.summary_dir,
                'reward_formulation': FLAGS.reward_formulation,
                'use_envlogger': FLAGS.use_envlogger,
                "knobs": FLAGS.knobs,
                "congestion_aware": FLAGS.congestion_aware,
                "network": FLAGS.network,
                "system": FLAGS.system,
                "workload_file": FLAGS.workload_file}
        taskList.append(task)
    elif FLAGS.algo == "rl":
        task = {"algo": FLAGS.algo,
                "workload": FLAGS.workload, 
                "rl_algo": FLAGS.rl_algo,
                "rl_form": FLAGS.rl_form,
                "reward_form": FLAGS.reward_form,
                "reward_scale": FLAGS.reward_scale,
                "num_steps": FLAGS.num_steps,
                "eval_every": FLAGS.eval_every,
                "eval_episodes": FLAGS.eval_episodes,
                "seed": FLAGS.seed,
                "learning_rate": FLAGS.learning_rate,
                "entropy_cost": FLAGS.entropy_cost,
                "ppo_clipping_epsilon": FLAGS.ppo_clipping_epsilon,
                "clip_value": FLAGS.clip_value,
                "summarydir": FLAGS.summarydir,
                "envlogger_dir": FLAGS.envlogger_dir,
                "use_envlogger": FLAGS.use_envlogger,
                "run_distributed": FLAGS.run_distributed,
                "params_scaling": FLAGS.params_scaling,
                "knobs": FLAGS.knobs,
                "congestion_aware": FLAGS.congestion_aware,
                "network": FLAGS.network,
                "system": FLAGS.system,
                "workload_file": FLAGS.workload_file}
        taskList.append(task)
    else:
        raise NotImplementedError


    for each_task in taskList:
        # update the workload in simulator
        run_task(each_task)


if __name__ == '__main__':
   app.run(main)
