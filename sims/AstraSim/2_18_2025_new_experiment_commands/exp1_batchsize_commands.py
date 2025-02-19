import yaml
import csv

# csv_file_path = "exp1_batchsize_commands.csv"
csv_file_path = "exp1_batchsize_names.csv"
command_list = []

# b) ga
steps_range = [1024, 10000]
num_agents_range = [32, 64]
prob_mut_range = [0.01, 0.05]

# Iterate through combinations and create YAML files
for workload in ["gpt3_175b", "gpt3_13b", "vit_large", "vit_base"]:
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
                file_log = f"all_logs/experiment1b_batchsize_{step}_{num_agents}_{prob_str}_{workload}_log"
                file_name = f"experiment1b_batchsize_{step}_{num_agents}_{prob_str}_{workload}.yml"
                
                cmd += f" --experiment=./2-18-2025-new_experiment_commands/exp1_batchsize/{file_name}"
                cmd += f" --summary_dir=./{file_log}"
                cmd += f" --timeout=172800"

                cmd = f"experiment1b_batchsize_{step}_{num_agents}_{prob_str}_{workload},exp1b_batchsize"

                command_list.append(cmd)


# c) aco
steps_range = [1024, 10000]
ant_count = [8, 16]
greediness_range = [0.25, 0.5, 0.75]
evaporation_range = [0.25, 0.5, 0.75]

# Iterate through combinations and create YAML files
for workload in ["gpt3_175b", "gpt3_13b", "vit_large", "vit_base"]:
    for step in steps_range:
        for ant in ant_count:
            for greed in greediness_range:
                for eva in evaporation_range:
                    cmd = "python launch_gcp.py"
                    # Write the updated YAML content to a new file
                    # 0.001 = 1, 0.01 = 10, 0.05 = 50
                    greed_str = str(int(100 * greed))
                    eva_str = str(int(100 * eva))
                    file_log = f"all_logs/experiment1c_batchsize_{step}_{ant}_{greed_str}_{eva_str}_{workload}_log"
                    file_name = f"experiment1c_batchsize_{step}_{ant}_{greed_str}_{eva_str}_{workload}.yml"

                    cmd += f" --experiment=./2-18-2025-new_experiment_commands/exp1_batchsize/{file_name}"
                    cmd += f" --summary_dir=./{file_log}"
                    cmd += f" --timeout=172800"

                    cmd = f"experiment1c_batchsize_{step}_{ant}_{greed_str}_{eva_str}_{workload},exp1_batchsize"

                    command_list.append(cmd)


# d) bo
steps_range = [1024, 5000]
rand_state = [1, 2, 3, 4]

# Iterate through combinations and create YAML files
for workload in ["gpt3_175b", "gpt3_13b", "vit_large", "vit_base"]:
    for step in steps_range:
        for rand in rand_state:
            cmd = "python launch_gcp.py"
            file_log = f"all_logs/experiment1d_batchsize_{step}_{rand}_{workload}_log"
            file_name = f"experiment1d_batchsize_{step}_{rand}_{workload}.yml"

            cmd += f" --experiment=./2-18-2025-new_experiment_commands/exp1_batchsize/{file_name}"
            cmd += f" --summary_dir=./{file_log}"
            cmd += f" --timeout=172800"

            cmd = f"experiment1d_batchsize_{step}_{rand}_{workload},exp1_batchsize"

            command_list.append(cmd)

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    for cmd in command_list:
        writer.writerow([cmd])

