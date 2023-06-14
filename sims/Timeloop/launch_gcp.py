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

flags.DEFINE_string('algo', 'bo', 'Which Algorithm to run')
flags.DEFINE_string('workload', 'AlexNet', 'Which workload to run')
flags.DEFINE_string('summary_dir', '.', 'Directory to store the summary')
flags.DEFINE_integer('num_iter', 10, 'Number of iterations')
flags.DEFINE_string('reward_formulation', 'area', 'Which reward formulation to use')

# ACO
flags.DEFINE_integer('ant_count', 2, 'Number of ants')
flags.DEFINE_float('evaporation', 0.5, 'Evaporation rate')
flags.DEFINE_float('greediness', 0.5, 'Greedy rate')

# BO
flags.DEFINE_integer('rand_state', 0, 'Random state')

# GA
flags.DEFINE_integer('num_agents', 10, 'Number of agents')
flags.DEFINE_float('prob_mutation', 0.1, 'Probability of mutation.')

# Random Walk
flags.DEFINE_integer('num_episodes', 1, 'Number of episodes')

# RL
flags.DEFINE_string('train_script', 'single_agent', 'Agent type')
# RL algorithm type (ppo, sac)
flags.DEFINE_string('rl_algo', 'ppo', 'RL algorithm')
# RL formulation
flags.DEFINE_string('rl_form', 'sa', 'RL formulation')
# Scale rewards
flags.DEFINE_string('reward_scaling', 'false', 'Scale rewards')
flags.DEFINE_integer('num_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('eval_every', 50, 'Number of evaluation steps.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episode.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate.')
flags.DEFINE_float('entropy_cost', 0.9, 'Entropy cost.')
flags.DEFINE_float('ppo_clipping_epsilon', 0.7, 'PPO clipping epsilon.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_bool('use_envlogger', False, 'Use envlogger.')

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
    
    # Get date and time and construct unique directory name to store outputs
    now = datetime.now()
    now = str(now)
    date = now.split(" ")[0]
    time = now.split(" ")[1]

    # common for timeloop 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(base_dir, "script")
    output_dir = os.path.join(base_dir, "output")
    arch_dir = os.path.join(base_dir, "arch")
    mapper_dir = os.path.join(base_dir, "mapper")
    workload_dir = os.path.join(base_dir, "layer_shapes", FLAGS.workload)

    if (algo == "rw"):
        num_iter = task['num_iter']
        summary_dir = task['summary_dir']
        reward_formulation = task['reward_formulation']
        
        # change the current directory to base_dir
        os.chdir(base_dir)

        # Create the command to run the simulation
        cmd = "python train_rw_timeloop.py --script " + script_dir \
            + " --output " + output_dir \
            + " --arch " + arch_dir \
            + " --mapper " + mapper_dir \
            + " --workload " + workload_dir \
            + " --num_steps " + str(num_iter) \
            + " --summary_dir " + summary_dir \
            + " --reward_formulation " + reward_formulation
        print(cmd)

        os.system(cmd)
    elif (algo == "ga"):
        num_agents = task['num_agents']
        num_iter = task['num_iter']
        prob_mutation = task['prob_mutation']
        summary_dir = task['summary_dir']
        reward_formulation = task['reward_formulation']
        
        # change the current directory to base_dir
        os.chdir(base_dir)

        # Create the command to run the simulation
        cmd = "python train_ga_timeloop.py --script " + script_dir \
            + " --output " + output_dir \
            + " --arch " + arch_dir \
            + " --mapper " + mapper_dir \
            + " --workload " + workload_dir \
            + " --num_agents " + str(num_agents) \
            + " --num_iter " + str(num_iter) \
            + " --prob_mutation " + str(prob_mutation) \
            + " --summary_dir " + summary_dir \
            + " --reward_formulation " + reward_formulation
        print(cmd)

        os.system(cmd)
    elif (algo == "aco"):
        ant_count = task['ant_count']
        num_iter = task['num_iter']
        evaporation = task['evaporation']
        greediness = task['greediness']
        summary_dir = task['summary_dir']
        reward_formulation = task['reward_formulation']

        aco_agent_config_file = os.path.join(
                                arch_gym_configs.proj_root_path,
                                "settings",
                                arch_gym_configs.aco_config)
        aco_hyperparams = {"evaporation": evaporation,
                            "ant_count": ant_count,
                            "greediness": greediness,
                            "depth": num_iter}
        update_aco_agent_configs(aco_agent_config_file, aco_hyperparams)

        # change the current directory to base_dir
        os.chdir(base_dir)
        
        # Create the command to run the simulation
        cmd = "python train_aco_timeloop.py --script " + script_dir \
            + " --output " + output_dir \
            + " --arch " + arch_dir \
            + " --mapper " + mapper_dir \
            + " --workload " + workload_dir \
            + " --ant_count " + str(ant_count) \
            + " --depth " + str(num_iter) \
            + " --evaporation " + str(evaporation) \
            + " --greediness " + str(greediness) \
            + " --summary_dir " + summary_dir \
            + " --reward_formulation " + reward_formulation
        print(cmd)

        os.system(cmd)
    elif (algo == "bo"):
        num_iter = task['num_iter']
        rand_state = task['rand_state']
        summary_dir = task['summary_dir']
        reward_formulation = task['reward_formulation']
        
        # change the current directory to base_dir
        os.chdir(base_dir)

        # Create the command to run the simulation
        cmd = "python train_bo_timeloop.py --script " + script_dir \
            + " --output " + output_dir \
            + " --arch " + arch_dir \
            + " --mapper " + mapper_dir \
            + " --workload " + workload_dir \
            + " --num_iter " + str(num_iter) \
            + " --random_state " + str(rand_state) \
            + " --summary_dir " + summary_dir \
            + " --reward_formulation " + reward_formulation
        print(cmd)

        os.system(cmd)
    # Run RL
    elif algo == "rl":

        rl_algo            = task["rl_algo"]
        rl_form            = task["rl_form"]
        reward_scaling     = task["reward_scaling"]
        num_steps          = task["num_steps"]
        eval_every         = task["eval_every"]
        eval_episodes      = task["eval_episodes"]
        learning_rate      = task["learning_rate"]
        seed               = task["seed"]
        use_envlogger      = task["use_envlogger"]
        summary_dir        = task["summary_dir"]
        reward_formulation = task["reward_formulation"]
        batch_size         = task["batch_size"]
        entropy_cost       = task["entropy_cost"]
        ppo_clipping_epsilon = task["ppo_clipping_epsilon"]
        if FLAGS.train_script == 'single_agent':
            
            reward_formulation_list = FLAGS.reward_formulation.split(" ")

            # join the list of reward formulations with a _
            reward_formulation = "_".join(reward_formulation_list)
            # construct the log dir based on reward formulation
            log_dir = os.path.join(FLAGS.summary_dir, "rl_logs", reward_formulation, FLAGS.workload)
            
            # change the current directory to base_dir
            os.chdir(base_dir)

            # single agent
            cmd = "python train_single_agent.py " + \
            "--script={} ".format(script_dir) + \
            "--output={} ".format(output_dir) + \
            "--arch={} ".format(arch_dir) + \
            "--mapper={} ".format(mapper_dir) + \
            "--workload={} ".format(workload_dir) + \
            "--rl_algo={} ".format(rl_algo) + \
            "--rl_form={} ".format(rl_form) + \
            "--reward_form={} ".format(reward_formulation) + \
            "--reward_scale={} ".format(reward_scaling) + \
            "--summarydir={} ".format(log_dir) + \
            "--num_steps={} ".format(num_steps) + \
            "--eval_every={} ".format(eval_every) + \
            "--eval_episodes={} ".format(eval_episodes) + \
            "--learning_rate={} ".format(learning_rate) + \
            "--batch_size={} ".format(batch_size) + \
            "--entropy_cost={} ".format(entropy_cost) + \
            "--ppo_clipping_epsilon={} ".format(ppo_clipping_epsilon) + \
            "--seed={} ".format(int(seed)) + \
            "--use_envlogger={}".format(use_envlogger)

            print("Shell Command", cmd)
            os.system(cmd)
        else:
            print("Multi agent training not supported yet")
            exit(1)
    else:
        print("Invalid algorithm")
        exit(0)

def main(_):
    taskList = []
    sim_config = arch_gym_configs.sim_config

    if FLAGS.algo == 'rw':
        task = {'algo': FLAGS.algo,
                'workload': FLAGS.workload,
                'num_iter': FLAGS.num_iter,
                'summary_dir': FLAGS.summary_dir,
                'reward_formulation': FLAGS.reward_formulation}
    elif FLAGS.algo == 'ga':
        task = {'algo': FLAGS.algo,
                'workload': FLAGS.workload,
                'num_agents': FLAGS.num_agents,
                'num_iter': FLAGS.num_iter,
                'prob_mutation': FLAGS.prob_mutation,
                'summary_dir': FLAGS.summary_dir,
                'reward_formulation': FLAGS.reward_formulation}
    elif FLAGS.algo == 'aco':
        task = {'algo': FLAGS.algo,
                'workload': FLAGS.workload,
                'ant_count': FLAGS.ant_count,
                'num_iter': FLAGS.num_iter,
                'evaporation': FLAGS.evaporation,
                'greediness': FLAGS.greediness,
                'summary_dir': FLAGS.summary_dir,
                'reward_formulation': FLAGS.reward_formulation}
    elif FLAGS.algo == 'bo':
        task = {'algo': FLAGS.algo,
                'workload': FLAGS.workload,
                'num_iter': FLAGS.num_iter,
                "rand_state": FLAGS.rand_state,
                'summary_dir': FLAGS.summary_dir,
                'reward_formulation': FLAGS.reward_formulation}
    elif FLAGS.algo == "rl":
        task = {"algo"              : FLAGS.algo,
                "workload"          : FLAGS.workload, 
                "rl_algo"           : FLAGS.rl_algo,
                "rl_form"           : FLAGS.rl_form,
                "reward_scaling"    : FLAGS.reward_scaling,
                "num_steps"         : FLAGS.num_steps,
                "eval_every"        : FLAGS.eval_every,
                "eval_episodes"     : FLAGS.eval_episodes,
                "learning_rate"     : FLAGS.learning_rate,
                "seed"              : int(FLAGS.seed),
                "use_envlogger"     : FLAGS.use_envlogger,
                "summary_dir"       : FLAGS.summary_dir,
                "batch_size"        : int(FLAGS.batch_size),
                "entropy_cost"      : FLAGS.entropy_cost,
                "ppo_clipping_epsilon"  : FLAGS.ppo_clipping_epsilon,
                "reward_formulation": FLAGS.reward_formulation}
        taskList.append(task)
    else:
        print("Invalid algorithm")
    taskList.append(task)
    print(taskList)

    for each_task in taskList:
        run_task(each_task)

if __name__ == '__main__':
   app.run(main)

