import random
import itertools
import os
import sys

print("Import works!!")

use_reorder = [True, False]
use_growing = [True, False]
use_aging = [True, False]

alpha_reorder = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
alpha_growing = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
alpha_aging = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


args = list(itertools.product(use_reorder, use_growing, use_aging, alpha_reorder, alpha_growing, alpha_aging))

# Find the total number of combinations
total_combinations = len(args)
print("Total number of combinations: {}".format(total_combinations))


count = 0

def generate_command(args):

    use_reorder, use_growing, use_aging, alpha_reorder, alpha_growing, alpha_aging = args
    command = "python main.py --use_reorder {} --use_growing {} --use_aging {} --reorder_alpha {} --growing_alpha {} --aging_alpha {}".format(use_reorder,
    use_growing, use_aging, alpha_reorder, alpha_growing, alpha_aging)
    
    return command

def run_once(cmd):

    print("Running command: {}".format(cmd))
    
    # use python subprocess to run the command
    os.system(cmd)

    print("Finished running command: {}".format(cmd))

for arg in args:
    use_reorder, use_growing, use_aging, alpha_reorder, alpha_growing, alpha_aging = arg
    command = generate_command(arg)
    print(command)
    run_once(command)
    


    
    

