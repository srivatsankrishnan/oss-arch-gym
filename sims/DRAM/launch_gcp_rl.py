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

from datetime import date, datetime

# DRAMSys workload trace
flags.DEFINE_string('dram_trace', 'stream.stl', 'Workload to run.')
# single agent or multiagent
flags.DEFINE_string('train_script', 'single_agent', 'Agent type')
# RL algorithm type (ppo, sac)
flags.DEFINE_string('rl_algo', 'ppo', 'RL algorithm')
# RL formulation
flags.DEFINE_string('rl_form', 'macme', 'RL formulation')

# reward formulation (latency, power, or both)
flags.DEFINE_string('reward_formulation', 'latency', 'Reward formulation')

# Scale rewards
flags.DEFINE_string('reward_scaling', 'false', 'Scale rewards')

# Summary_dir 
flags.DEFINE_string('summary_dir', 'summaries', 'Summary directory')

flags.DEFINE_integer('num_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('eval_every', 50, 'Number of evaluation steps.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episode.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate.')
flags.DEFINE_float('entropy_cost', 0.1, 'Entropy cost.')
flags.DEFINE_bool('use_enlogger', False, 'Use enlogger.')
FLAGS = flags.FLAGS

def main(_):
    
    if FLAGS.train_script == 'single_agent':

        # construct the log dir based on reward formulation
        log_dir = os.path.join(FLAGS.summary_dir, "rl_logs", FLAGS.reward_formulation, FLAGS.dram_trace)
        # single agent
        cmd = "python train_single_agent.py " + \
            "--rl_algo={} ".format(FLAGS.rl_algo) + \
            "--dram_trace={} ".format(FLAGS.dram_trace) + \
            "--rl_form={} ".format(FLAGS.rl_form) + \
            "--reward_form={} ".format(FLAGS.reward_formulation) + \
            "--reward_scale={} ".format(FLAGS.reward_scaling) + \
            "--summarydir={} ".format(log_dir) + \
            "--num_steps={} ".format(FLAGS.num_steps) + \
            "--eval_every={} ".format(FLAGS.eval_every) + \
            "--eval_episodes={} ".format(FLAGS.eval_episodes) + \
            "--learning_rate={} ".format(FLAGS.learning_rate) + \
            "--seed={} ".format(int(FLAGS.seed)) + \
            "--use_envlogger={}".format(FLAGS.use_enlogger)

        print("Shell Command", cmd)
        os.system(cmd)
    elif FLAGS.train_script == 'multi_agent':
        # construct the log dir based on reward formulation
        log_dir = os.path.join(FLAGS.summary_dir, "rl_logs", FLAGS.reward_formulation, FLAGS.dram_trace)
        # multiagent
        cmd = "python train_multiagent.py " + \
            "--rl_algo={} ".format(FLAGS.rl_algo) + \
            "--dram_trace={} ".format(FLAGS.dram_trace) + \
            "--rl_form={} ".format(FLAGS.rl_form) + \
            "--reward_form={} ".format(FLAGS.reward_formulation) + \
            "--reward_scale={} ".format(FLAGS.reward_scaling) + \
            "--summarydir={} ".format(log_dir) + \
            "--num_steps={} ".format(FLAGS.num_steps) + \
            "--eval_every={} ".format(FLAGS.eval_every) + \
            "--learning_rate={} ".format(FLAGS.learning_rate) + \
            "--seed={} ".format(FLAGS.seed) + \
            "--use_envlogger={}".format(FLAGS.use_enlogger)

        print("Shell Command", cmd)
        os.system(cmd)
    else:
        raise ValueError("Invalid train_script")

if __name__ == '__main__':
   app.run(main)