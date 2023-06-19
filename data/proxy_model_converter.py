import pandas as pd

from absl import flags
from absl import app
import numpy as np
import ast
import json
import csv

flags.DEFINE_string('data_file', '', 'Path to the trajectory data file')
flags.DEFINE_string('output_file', './data.csv', 'Path to the trajectory data file')

FLAGS = flags.FLAGS


# Format of each line in the trajectory files should be
# reward, {action_dict}, [[energy, power, latency]]

def main(_):
    data_file = FLAGS.data_file
    schema = ['Reward']
    csv_lines = []
    with open(data_file, 'r') as data:
        for line in data.readlines():
            csv_line = []
            line = line.split(',', 1)
            try:
                reward = float(line[0])
            except ValueError:
                # Skip invalid lines
                continue

            csv_line.append(reward)

            line = line[1].split('"', 2)
            actions = ast.literal_eval(line[1])
            if len(schema) <= 1:
                action_categories = sorted(actions.keys())
                for k in action_categories:
                    schema.append(k)
            for k in action_categories:
                csv_line.append(actions[k])

            line = line[2]
            line = line.split()
            energy = float(line[0][3:])
            power = float(line[1])
            latency = float(line[2][:-2])
            csv_line.extend([energy, power, latency])
            if len(schema) < len(csv_line):
                schema.extend(['Energy', 'Power', 'Latency'])
            csv_lines.append(csv_line)
        with open(FLAGS.output_file, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(schema)
            writer.writerows(csv_lines)


if __name__ == '__main__':
    app.run(main)
