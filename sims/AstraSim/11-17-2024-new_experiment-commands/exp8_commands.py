import yaml
import csv

csv_file_path = "exp8_commands.csv"
# csv_file_path = "exp8_names.csv"
command_list = []

# b) ga
steps_range = [1024, 10000]
num_agents_range = [32, 64]
prob_mut_range = [0.01, 0.05]

# Iterate through combinations and create YAML files
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
            file_log = f"all_logs/experiment8b_{step}_{num_agents}_{prob_str}_gpt3_175b_log"
            file_name = f"experiment8b_{step}_{num_agents}_{prob_str}_gpt3_175b.yml"
            
            cmd += f" --experiment=./11-17-2024-new_experiment-commands/exp8/{file_name}"
            cmd += f" --summary_dir=./{file_log}"
            cmd += f" --timeout=259200"

            # cmd = f"experiment8b_{step}_{num_agents}_{prob_str}_gpt3_175b"

            command_list.append(cmd)


# c) aco
steps_range = [1024, 10000]
ant_count = [8, 16]
greediness_range = [0.25, 0.5, 0.75]
evaporation_range = [0.25, 0.5, 0.75]

# Iterate through combinations and create YAML files
for step in steps_range:
    for ant in ant_count:
        for greed in greediness_range:
            for eva in evaporation_range:
                cmd = "python launch_gcp.py"
                # Write the updated YAML content to a new file
                # 0.001 = 1, 0.01 = 10, 0.05 = 50
                greed_str = str(int(100 * greed))
                eva_str = str(int(100 * eva))
                file_log = f"all_logs/experiment8c_{step}_{ant}_{greed_str}_{eva_str}_gpt3_175b_log"
                file_name = f"experiment8c_{step}_{ant}_{greed_str}_{eva_str}_gpt3_175b.yml"

                cmd += f" --experiment=./11-17-2024-new_experiment-commands/exp8/{file_name}"
                cmd += f" --summary_dir=./{file_log}"
                cmd += f" --timeout=259200"

                # cmd = f"experiment8c_{step}_{ant}_{greed_str}_{eva_str}_gpt3_175b"

                command_list.append(cmd)


# d) bo
steps_range = [1024, 5000]
rand_state = [1, 2, 3, 4]

# Iterate through combinations and create YAML files
for step in steps_range:
    for rand in rand_state:
        cmd = "python launch_gcp.py"
        file_log = f"all_logs/experiment8d_{step}_{rand}_gpt3_175b_log"
        file_name = f"experiment8d_{step}_{rand}_gpt3_175b.yml"

        cmd += f" --experiment=./11-17-2024-new_experiment-commands/exp8/{file_name}"
        cmd += f" --summary_dir=./{file_log}"
        cmd += f" --timeout=259200"

        # cmd = f"experiment8d_{step}_{rand}_gpt3_175b"

        command_list.append(cmd)

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    for cmd in command_list:
        writer.writerow([cmd])

