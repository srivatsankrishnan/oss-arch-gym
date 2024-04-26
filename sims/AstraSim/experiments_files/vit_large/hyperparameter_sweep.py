import yaml

# b) ga
steps_range = [64, 128, 512, 1024, 10000, 50000]
num_agents_range = [2, 4, 8, 16, 32, 64]
prob_mut_range = [0.01, 0.05, 0.001]

# Iterate through combinations and create YAML files
for i in range(1, 6, 1):
    for step in steps_range:
        for num_agents in num_agents_range:
            for prob in prob_mut_range:
                
                cur_yml = {
                    "ALGORITHM": "ga",
                    "STEPS": step,
                    "NUM_AGENTS": num_agents,
                    "PROB_MUT": prob,
                    "REWARD": "both",
                    "KNOBS": f"astrasim_220_example/knobs_{i}.py",
                    "NETWORK": f"astrasim_220_example/network_input_{i}.yml",
                    "SYSTEM": f"astrasim_220_example/system_input_{i}.json",
                    "WORKLOAD": f"astrasim_220_example/workload_cfg_{i}_vit.json"
                }
                
                # Write the updated YAML content to a new file
                # 0.001 = 1, 0.01 = 10, 0.05 = 50
                prob_str = ""
                if prob == 0.001:
                    prob_str = '001'
                elif prob == 0.01:
                    prob_str = '01'
                elif prob == 0.05:
                    prob_str = '05'
                file_name = f"experiment{i}b_{step}_{num_agents}_{prob_str}_vit.yml"
                
                with open(file_name, "w") as file:
                    yaml.dump(cur_yml, file)
                    
                print(file_name)

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
                
                    cur_yml = {
                        "ALGORITHM": "aco",
                        "STEPS": step,
                        "ANT_COUNT": ant,
                        "GREEDINESS": greed,
                        "EVAPORATION": eva,
                        
                        "REWARD": "both",
                        "KNOBS": f"astrasim_220_example/knobs_{i}.py",
                        "NETWORK": f"astrasim_220_example/network_input_{i}.yml",
                        "SYSTEM": f"astrasim_220_example/system_input_{i}.json",
                        "WORKLOAD": f"astrasim_220_example/workload_cfg_{i}_vit.json"
                    }
                    
                    # Write the updated YAML content to a new file
                    # 0.001 = 1, 0.01 = 10, 0.05 = 50
                    greed_str = str(int(100 * greed))
                    eva_str = str(int(100 * eva))
                    file_name = f"experiment{i}c_{step}_{ant}_{greed_str}_{eva_str}_vit.yml"
                    
                    with open(file_name, "w") as file:
                        yaml.dump(cur_yml, file)
                        
                    print(file_name)


# d) bo
steps_range = [64, 128, 512, 1024, 10000, 50000]
rand_state = [1, 2, 3, 4, 5]

# Iterate through combinations and create YAML files
for i in range(1, 6, 1):
    for step in steps_range:
        for rand in rand_state:
            
                cur_yml = {
                    "ALGORITHM": "bo",
                    "STEPS": step,
                    "RAND_STATE_BO": rand,
                    
                    "REWARD": "both",
                    "KNOBS": f"astrasim_220_example/knobs_{i}.py",
                    "NETWORK": f"astrasim_220_example/network_input_{i}.yml",
                    "SYSTEM": f"astrasim_220_example/system_input_{i}.json",
                    "WORKLOAD": f"astrasim_220_example/workload_cfg_{i}_vit.json"
                }
                
                file_name = f"experiment{i}d_{step}_{rand}_vit.yml"
                
                with open(file_name, "w") as file:
                    yaml.dump(cur_yml, file)
                    
                print(file_name)