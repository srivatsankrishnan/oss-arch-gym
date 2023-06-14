#!/usr/bin/env python3
#import importlib
import os
import sys
sys.path.insert(0, 'sims/Sniper/')
import shutil
import time
import simulate_benchmark


launcher = simulate_benchmark.SniperLauncher(16)

# TO do : Move this to env helpers

def create_agent_configs(agent_ids, cfg):
    for i in range(len(jobs)):
        shutil.copy(config_file, 'arch_gym_x86_agent_{}.cfg'.format(i))

    # return absolute paths to the config files
    return [os.path.abspath('arch_gym_x86_agent_{}.cfg'.format(i)) for i in range(len(jobs))]

def delete_agents_configs(configs):
    status = False
    for config in configs:
        try:
            os.remove(config)
            status = True
        except Exception as e:
            print(e)
            status = False
    return status    

def create_callback(output_dir):
    # Define a function which will be called upon completion of the benchmark.

    # Regarding the function that is created here:
    # 'res' is always passed in by Python when calling the callback
    # function.  Supposedly it's the return value of the processes, but in
    # my testing, it always shows up as 'None', so I just ignore it.

    # Instead, just call 'error_check' on output_dir to check for errors.
    # It is **CRITICAL** that the callback function return immediately and
    # not do anything fancy like raising an exception.  This simple example
    # just prints whether there was an error; it does NOT raise a
    # ConfigurationError exception.
    return lambda res : simulate_benchmark.error_check(output_dir)

jobs = ['600', '600']

# create copies of config files
config_file = 'arch_gym_x86.cfg'


config_files = create_agent_configs(range(len(jobs)), config_file)
output_dirs = []
results = []

for agent_idx in range(len(jobs)):
    output_dir = 'agent_.{}_.{}'.format(agent_idx,jobs[agent_idx])
    output_dirs.append(output_dir)

    benchmark = jobs[agent_idx]
    def upon_completion(res):
        try:
            simulate_benchmark.error_check(output_dir)
        except Exception as e:
            print(e)

    # The callback function is optional.
    result = launcher.batch_benchmark(benchmark, 'CPU2017', output_dir, config_files[agent_idx], callback=create_callback(output_dir))
    results.append(result)
    print('Launched Agent{}_{}'.format(agent_idx,benchmark))

# Wait for the all the jobs to finish.
for result in results:
    result.wait()

for output_dir in output_dirs:
    try:
        # Now combine the stats.
        simulate_benchmark.combine_stats(output_dir)

        # clear the agent specific config files
        delete_agents_configs(config_files)
    except Exception as e:
        # If the output files are not found, Python will throw an exception.  That can be gracefully handled here.
        print(e)
