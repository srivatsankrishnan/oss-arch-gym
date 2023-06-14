import random
import itertools
import os
import sys

print("Import works!!")

prob_mutation = [0.01, 0.05, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
workload = ["resnet18", "vgg16"]
num_iter = [100, 1000, 10000]
num_agents = [16, 32, 64, 128]



args = list(itertools.product(prob_mutation, workload, num_iter, num_agents))

# Find the total number of combinations
total_combinations = len(args)
print("Total number of combinations: {}".format(total_combinations))

def run_once(cmd):

    print("Running command: {}".format(cmd))
    
    # use python subprocess to run the command
    os.system(cmd)

    print("Finished running command: {}".format(cmd))


def generate_command(args):

    prob_mutation, workload, num_iter, num_agents = args
    command = ("python launch_gcp.py "
           f"--prob_mutation {prob_mutation} "
           f"--workload {workload} "
           f"--num_iter {num_iter} "
           f"--num_agents {num_agents}")
    
    return command

for arg in args:
    prob_mutation, workload, num_iter, num_agents = arg
    command = generate_command(arg)
    print(command)
    run_once(command)
    
