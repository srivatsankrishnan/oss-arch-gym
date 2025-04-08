import yaml
import csv

csv_file_path = "exp_commands_scalability.csv"
# csv_file_path = "exp_names_scalability.csv"
command_list = []

reward_type = "runtime*bw"
for workload in ["gpt3_175b", "llama7b", "vit_large"]:
    for exp_num in ["1.2", "4.2"]:
        for batch_size in ["1024", "2048", "4096", "8192", "16384"]:
            # b) ga
            steps_range = [1024, 10000]
            num_agents_range = [32, 64]
            prob_mut_range = [0.01, 0.05]
            # Iterate through combinations and create YAML files
            for step in steps_range:
                for num_agents in num_agents_range:
                    for prob in prob_mut_range:  
                        cmd = "python launch_gcp.py"                      
                        prob_str = ""
                        if prob == 0.01:
                            prob_str = '01'
                        elif prob == 0.05:
                            prob_str = '05'
                        file_log = f"all_logs/experiment{exp_num}b_{step}_{num_agents}_{prob_str}_{workload}_{batch_size}_log"
                        file_name = f"exp{exp_num}_batchsize/experiment{exp_num}b_{step}_{num_agents}_{prob_str}_{workload}_{batch_size}.yml"
                        
                        cmd += f" --experiment=./experiments_files/{file_name}"
                        cmd += f" --summary_dir=./{file_log}"
                        cmd += f" --timeout=172800"

                        # cmd = f"experiment{exp_num}b_{step}_{num_agents}_{prob_str}_{workload}_{batch_size},exp{exp_num}"

                        command_list.append(cmd)

            # d) bo
            steps_range = [1024, 5000]
            rand_state = [1, 2, 3, 4]
            # Iterate through combinations and create YAML files
            for step in steps_range:
                for rand in rand_state:   
                    cmd = "python launch_gcp.py"
                    file_log = f"all_logs/experiment{exp_num}d_{step}_{rand}_{workload}_{batch_size}_log"                 
                    file_name = f"exp{exp_num}_batchsize/experiment{exp_num}d_{step}_{rand}_{workload}_{batch_size}.yml"
                    
                    cmd += f" --experiment=./experiments_files/{file_name}"
                    cmd += f" --summary_dir=./{file_log}"
                    cmd += f" --timeout=172800"

                    # cmd = f"experiment{exp_num}d_{step}_{rand}_{workload}_{batch_size},exp{exp_num}"

                    command_list.append(cmd)

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    for cmd in command_list:
        writer.writerow([cmd])