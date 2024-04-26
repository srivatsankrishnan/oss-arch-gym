import yaml
import csv

csv_file_path = "experiment_commands_vit.csv"
command_list = []

# b) ga
steps_range = [64, 128, 512, 1024, 10000, 50000]
num_agents_range = [2, 4, 8, 16, 32, 64]
prob_mut_range = [0.01, 0.05, 0.001]

# Iterate through combinations and create YAML files
for i in range(1, 6, 1):
    for step in steps_range:
        for num_agents in num_agents_range:
            for prob in prob_mut_range:
                
                # Write the updated YAML content to a new file
                # 0.001 = 1, 0.01 = 10, 0.05 = 50
                cmd = "python launch_gcp.py"
                prob_str = ""
                if prob == 0.001:
                    prob_str = '001'
                elif prob == 0.01:
                    prob_str = '01'
                elif prob == 0.05:
                    prob_str = '05'
                file_log = f"experiment{i}b_{step}_{num_agents}_{prob_str}_vit_log"
                file_name = f"experiment{i}b_{step}_{num_agents}_{prob_str}_vit.yml"
                
                cmd += f" --experiment=./experiments_files/vit_large/{file_name}"
                cmd += f" --summary_dir=./{file_log}"
                cmd += f" --timeout=345600"
                # cmd = f"experiment{i}b_{step}_{num_agents}_{prob_str}_vit"

                command_list.append(cmd)


# c) aco
steps_range = [64, 128, 512, 1024, 10000, 50000]
ant_count = [2, 4, 8, 16, 32, 64]
greediness_range = [0.0, 0.25, 0.5, 0.75, 1.0]
evaporation_range = [0.1, 0.25, 0.5, 0.75, 1.0]

# Iterate through combinations and create YAML files
for i in range(1, 6, 1):
    for step in steps_range:
        for ant in ant_count:
            for greed in greediness_range:
                for eva in evaporation_range:

                    cmd = "python launch_gcp.py"
                    # Write the updated YAML content to a new file
                    # 0.001 = 1, 0.01 = 10, 0.05 = 50
                    greed_str = str(int(100 * greed))
                    eva_str = str(int(100 * eva))
                    file_log = f"experiment{i}c_{step}_{ant}_{greed_str}_{eva_str}_vit_log"
                    file_name = f"experiment{i}c_{step}_{ant}_{greed_str}_{eva_str}_vit.yml"

                    cmd += f" --experiment=./experiments_files/vit_large/{file_name}"
                    cmd += f" --summary_dir=./{file_log}"
                    cmd += f" --timeout=345600"

                    # cmd = f"experiment{i}c_{step}_{ant}_{greed_str}_{eva_str}_vit"

                    command_list.append(cmd)


# d) bo
steps_range = [64, 128, 512, 1024, 10000, 50000]
rand_state = [1, 2, 3, 4, 5]

# Iterate through combinations and create YAML files
for i in range(1, 6, 1):
    for step in steps_range:
        for rand in rand_state:
                
                cmd = "python launch_gcp.py"
                file_log = f"experiment{i}d_{step}_{rand}_vit_log"
                file_name = f"experiment{i}d_{step}_{rand}_vit.yml"

                cmd += f" --experiment=./experiments_files/vit_large/{file_name}"
                cmd += f" --summary_dir=./{file_log}"
                cmd += f" --timeout=345600"

                # cmd = f"experiment{i}d_{step}_{rand}_vit"

                command_list.append(cmd)

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    for cmd in command_list:
        writer.writerow([cmd])
