import os
import sys
import csv
import numpy as np
import pandas as pd

from absl import flags
from absl import app

# Define flags

flags.DEFINE_string('sim_name', 'AstraSim', 'Name of the simulation')
flags.DEFINE_bool('feasible', False, 'Creates a feasible dataset as well')

FLAGS = flags.FLAGS


def extract_headers(lines):
    cols = []

    for i in range(len(lines)):
        splitted_line = lines[i].split(': ')[:-1]
        for splitted_str in splitted_line:
            sub_str = splitted_str.split(',')[-1][2:-1]
            if i == 0:
                if sub_str != 'topology-name' and sub_str != 'dimensions-count':
                    for j in range(1, 3):
                        cols.append(sub_str + '-' + str(j))
                else:
                    cols.append(sub_str)
            else:
                cols.append(sub_str)
    
    return cols


def col_extractor(lines):
    final = []
    for i in range(2):
        splitted_line = lines[i].split(': ')[1:]
        for splitted_str in splitted_line:
            sub_str = splitted_str.split(', ')
            if splitted_str != splitted_line[-1]:
                sub_str = sub_str[:-1]
            for sub in sub_str:
                sub = sub.replace('[', '').replace(']', '').replace('}', '').replace("'", '')
                final.append(sub)

    return final


def main(_):
    sim_path = os.path.join('../sims', FLAGS.sim_name)
    if not os.path.exists(sim_path):
        print('Simulation path does not exist')
        sys.exit(1)
    
    # Read the log file
    log_path = os.path.join(sim_path, 'random_walker_logs', 'latency')
    if not os.path.exists(log_path):
        print('Logs do not exist')
        sys.exit(1)

    log_files = os.listdir(log_path)
    
    f = open(os.path.join(log_path, log_files[0], 'actions.csv'), 'r')
    for all_lines in csv.reader(f):
        lines = all_lines
        break
    f.close()

    actions_cols = extract_headers(lines)

    # Define the dataframe for actions and observations
    actions_pd = pd.DataFrame(columns=actions_cols)
    observations_pd = pd.DataFrame(columns=['observation-' + str(i) for i in range(1, 6)])

    for log_file in log_files:
        print('Processing log file: ', log_file)
        # Read the actions and observations
        f_obs = open(os.path.join(log_path, log_file, 'observations.csv'), 'r')
        f_act = open(os.path.join(log_path, log_file, 'actions.csv'), 'r')

        # Save the observations
        for lines in csv.reader(f_obs):
            if len(lines) == 5:
                splitted_line = [float(i) for i in lines]
                observations_pd.loc[len(observations_pd.index)] = splitted_line
            else:
                observations_pd.loc[len(observations_pd.index)] = [-10000 for i in range(5)]

        # Save the actions
        for lines in csv.reader(f_act):
            splitted_line = col_extractor(lines)
            actions_pd.loc[len(actions_pd.index)] = splitted_line

    # Saving the dataframes
    actions_pd.to_csv('../proxy-pipeline/data/actions_full.csv', index=False)
    observations_pd.to_csv('../proxy-pipeline/data/observations_full.csv', index=False)
    print('Saved the dataframes at: ', '../proxy-pipeline/data/')

    if FLAGS.feasible:
        # Create a feasible dataset
        actions_feasible = pd.DataFrame(columns=actions_pd.columns)
        observations_feasible = pd.DataFrame(columns=observations_pd.columns)

        # Iterate over the observations and actions to find the feasible ones
        for i in range(observations_pd.shape[0]):
            if observations_pd.iloc[i, 0] == -10000:
                continue
            else:
                observations_feasible.loc[len(observations_feasible)] = observations_pd.iloc[i]
                actions_feasible.loc[len(actions_feasible)] = actions_pd.iloc[i]

        # Saving the feasible dataframes
        actions_feasible.to_csv('../proxy-pipeline/data/actions_feasible.csv', index=False)
        observations_feasible.to_csv('../proxy-pipeline/data/observations_feasible.csv', index=False)
        print('Saved the feasible dataframes at: ', '../proxy-pipeline/data/')


if __name__ == '__main__':
    app.run(main)
