import subprocess
import numpy as np
from itertools import product
import sys
import os
import yaml
from datetime import date, datetime

LAYERS_DIR = "/n/janapa_reddi_lab/Lab/susobhan/arch-gym/sims/Timeloop/layer_shapes/"
MAPPER_FILE = "mapper.yaml"
ACO_CONFIG_FILE = "default_timeloop.yaml"


def edit_mapper_config(mapperdir, timeout):
    '''Edits the mapper configuration file to set the timeout'''
    fpath = mapperdir + "/" + MAPPER_FILE
    with open(fpath, 'r') as stream:
        try:
            loaded = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Modify the fields from the dict
    loaded['mapper']['timeout'] = timeout

    # Save it again
    with open(fpath, 'w') as stream:
        try:
            yaml.dump(loaded, stream, default_flow_style=False)
        except yaml.YAMLError as exc:
            print(exc)

    return mapperdir


def edit_aco_settings_file(settingsdir, ant_count, greediness, evaporation, max_depth):
    '''Edits the ACO settings file'''
    fpath = settingsdir + "/" + ACO_CONFIG_FILE
    with open(fpath, 'r') as stream:
        try:
            loaded = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Modify the fields from the dict
    loaded['DeepSwarm']['aco']['ant_count'] = ant_count
    loaded['DeepSwarm']['aco']['greediness'] = greediness
    loaded['DeepSwarm']['aco']['pheromone']['evaporation'] = evaporation
    loaded['DeepSwarm']['max_depth'] = max_depth

    # Save it again
    with open(fpath, 'w') as stream:
        try:
            yaml.dump(loaded, stream, default_flow_style=False)
        except yaml.YAMLError as exc:
            print(exc)

    return fpath


def main():
    idx = int(sys.argv[1])
    outputdir = str(sys.argv[2])
    mapperdir = str(sys.argv[3])
    archdir = str(sys.argv[4])
    scriptdir = str(sys.argv[5])
    settingsdir = str(sys.argv[6])

    if idx < 500:
        experiment = 1
        idx = idx-0
    elif idx < 1000:
        experiment = 2
        idx = idx-500
    elif idx < 1500:
        experiment = 3
        idx = idx-1000
    elif idx < 7000:
        experiment = 4
        idx = idx-1500

    # baseline values for each workload
    # 3 tuple of (energy, area, cycles)
    baseline_vals = {"AlexNet": [29206, 2.03, 7885704],
                     "mobilenet_v2": [76395, 2.03, 33515584],
                     "resnet50": [71230, 2.03, 32728064]}

    workloads = ["AlexNet", "mobilenet_v2", "resnet50"]

    # ---------- Experiment 1: Randomwalk ---------------------

    if experiment == 1:

        timeout = [100, 150, 200]
        maxsteps = [10000, 20000]

        # % of improvement in energy
        target_energy_improv = [0.99]

        # % of improvement in area
        target_area_improv = [0.99]

        # % of improvement in cycles
        target_cycle_improv = [0.99]

        combinations = list(product(workloads, timeout, maxsteps,
                            target_energy_improv, target_area_improv, target_cycle_improv))

        # pick the current experiment by index
        wl, t, ms, tei, tai, tci = combinations[idx]

        # calculate target values
        target_energy = baseline_vals[wl][0] * (1-tei)
        target_area = baseline_vals[wl][1] * (1-tai)
        target_cycle = baseline_vals[wl][2] * (1-tci)

        # configure target directories
        mapperdir = edit_mapper_config(mapperdir, t)
        workload_dir = LAYERS_DIR + str(wl)
        time_to_complete_path = os.getcwd()

        # Experiment name prefix to store files
        exp_name = f'rw_wl={wl}_t={t}_ms={ms}_'
        logdir = time_to_complete_path + "/logs/" + exp_name

        # use sva 1 flag to store actions in pkl file, otherwise set it to 0

        print(
            f'python timeloop_randomwalk.py -a {archdir} -o {outputdir} -w {workload_dir} -s {scriptdir} -m {mapperdir} -ms {ms} -log {logdir} -te {target_energy} -ta {target_area} -tc {target_cycle} -sva 0')

        start = datetime.now()
        subprocess.run(
            f'python timeloop_randomwalk.py -a {archdir} -o {outputdir} -w {workload_dir} -s {scriptdir} -m {mapperdir} -ms {ms} -log {logdir} -te {target_energy} -ta {target_area} -tc {target_cycle} -sva 0', shell=True)
        end = datetime.now()
        time_taken = (end - start).total_seconds()
        with open(time_to_complete_path + "/time_to_complete.txt", "a") as f:
            f.write(exp_name + ": " + str(time_taken) + "\n")

    # ---------- Experiment 2: Genetic Algorithm ---------------
    if experiment == 2:
        timeout = [100, 150, 200]
        maxsteps = [16, 32, 64]
        prob_mut = [0.01, 0.05]
        pop_size = [8, 16, 32, 64]

        # % of improvement in energy
        target_energy_improv = [0.99]

        # % of improvement in area
        target_area_improv = [0.99]

        # % of improvement in cycles
        target_cycle_improv = [0.99]

        combinations = list(product(workloads, timeout, maxsteps, prob_mut, pop_size,
                            target_energy_improv, target_area_improv, target_cycle_improv))

        # pick the current experiment by index
        wl, t, ms, pm, ps, tei, tai, tci = combinations[idx]

        # calculate target values
        target_energy = baseline_vals[wl][0] * (1-tei)
        target_area = baseline_vals[wl][1] * (1-tai)
        target_cycle = baseline_vals[wl][2] * (1-tci)

        # configure target directories
        mapperdir = edit_mapper_config(mapperdir, t)
        workload_dir = LAYERS_DIR + str(wl)
        time_to_complete_path = os.getcwd()

        # Experiment name prefix to store files
        exp_name = f'ga_wl={wl}_t={t}_ms={ms}_pm={pm}_ps={ps}_'
        logdir = time_to_complete_path + "/logs/" + exp_name

        print(
            f'python train_ga_timeloop.py -a {archdir} -o {outputdir} -w {workload_dir} -s {scriptdir} -m {mapperdir} -ms {ms} -en {exp_name} -log {logdir} -te {target_energy} -ta {target_area} -tc {target_cycle} -pop {ps} -pm {pm}')

        start = datetime.now()
        subprocess.run(
            f'python train_ga_timeloop.py -a {archdir} -o {outputdir} -w {workload_dir} -s {scriptdir} -m {mapperdir} -ms {ms} -en {exp_name} -log {logdir} -te {target_energy} -ta {target_area} -tc {target_cycle} -pop {ps} -pm {pm}', shell=True)
        end = datetime.now()
        time_taken = (end - start).total_seconds()
        with open(time_to_complete_path + "/ga_time_to_complete.txt", "a") as f:
            f.write(exp_name + ": " + str(time_taken) + "\n")

    # ---------- Experiment 3: BO ---------------
    if experiment == 3:
        timeout = [100, 150, 200]
        maxsteps = [512, 1024, 2048, 4096, 8192]
        random_state = [1, 2, 3, 4, 5]

        # % of improvement in energy
        target_energy_improv = [0.99]

        # % of improvement in area
        target_area_improv = [0.99]

        # % of improvement in cycles
        target_cycle_improv = [0.99]

        combinations = list(product(workloads, timeout, maxsteps, random_state,
                            target_energy_improv, target_area_improv, target_cycle_improv))

        # pick the current experiment by index
        wl, t, ms, seed, tei, tai, tci = combinations[idx]

        # calculate target values
        target_energy = baseline_vals[wl][0] * (1-tei)
        target_area = baseline_vals[wl][1] * (1-tai)
        target_cycle = baseline_vals[wl][2] * (1-tci)

        # configure target directories
        mapperdir = edit_mapper_config(mapperdir, t)
        workload_dir = LAYERS_DIR + str(wl)
        time_to_complete_path = os.getcwd()

        # Experiment name prefix to store files
        exp_name = f'bo_wl={wl}_t={t}_ms={ms}_sd={seed}_'
        logdir = time_to_complete_path + "/logs/" + exp_name

        print(
            f'python train_bo_timeloop.py -a {archdir} -o {outputdir} -w {workload_dir} -s {scriptdir} -m {mapperdir} -ms {ms} -sd {seed} -log {logdir} -en {exp_name} -te {target_energy} -ta {target_area} -tc {target_cycle}')

        start = datetime.now()
        subprocess.run(
            f'python train_bo_timeloop.py -a {archdir} -o {outputdir} -w {workload_dir} -s {scriptdir} -m {mapperdir} -ms {ms} -sd {seed} -log {logdir} -en {exp_name} -te {target_energy} -ta {target_area} -tc {target_cycle}', shell=True)
        end = datetime.now()
        time_taken = (end - start).total_seconds()
        with open(time_to_complete_path + "/bo_time_to_complete.txt", "a") as f:
            f.write(exp_name + ": " + str(time_taken) + "\n")

    # ---------- Experiment 4: ACO ---------------
    if experiment == 4:
        timeout = [100, 150, 200]
        num_ants = [2, 4, 8, 16, 32, 64]
        greediness = [0.0, 0.25, 0.5, 0.75, 1]
        evaporation = [0.1, 0.25, 0.5, 0.75, 1]
        depth = [2, 4, 8, 16]

        # % of improvement in energy
        target_energy_improv = [0.99]

        # % of improvement in area
        target_area_improv = [0.99]

        # % of improvement in cycles
        target_cycle_improv = [0.99]

        combinations = list(product(workloads, timeout, num_ants, greediness, evaporation, depth,
                            target_energy_improv, target_area_improv, target_cycle_improv))

        # pick the current experiment by index
        wl, t, na, g, ev, dpt, tei, tai, tci = combinations[idx]

        # calculate target values
        target_energy = baseline_vals[wl][0] * (1-tei)
        target_area = baseline_vals[wl][1] * (1-tai)
        target_cycle = baseline_vals[wl][2] * (1-tci)

        # configure target directories
        mapperdir = edit_mapper_config(mapperdir, t)
        aco_settings_file = edit_aco_settings_file(settingsdir, na, g, ev, dpt)
        workload_dir = LAYERS_DIR + str(wl)
        time_to_complete_path = os.getcwd()

        # Experiment name prefix to store files
        exp_name = f'aco_wl={wl}_t={t}_na={na}_g={g}_ev={ev}_dpt={dpt}_'
        logdir = time_to_complete_path + "/logs/" + exp_name

        print(
            f'python train_aco_timeloop.py -a {archdir} -o {outputdir} -w {workload_dir} -s {scriptdir} -m {mapperdir} -en {exp_name} -log {logdir} -te {target_energy} -ta {target_area} -tc {target_cycle} -st {aco_settings_file}')

        start = datetime.now()
        subprocess.run(
            f'python train_aco_timeloop.py -a {archdir} -o {outputdir} -w {workload_dir} -s {scriptdir} -m {mapperdir} -en {exp_name} -log {logdir} -te {target_energy} -ta {target_area} -tc {target_cycle} -st {aco_settings_file}', shell=True)
        end = datetime.now()
        time_taken = (end - start).total_seconds()
        with open(time_to_complete_path + "/aco_time_to_complete.txt", "a") as f:
            f.write(exp_name + ": " + str(time_taken) + "\n")


if __name__ == "__main__":
    main()
