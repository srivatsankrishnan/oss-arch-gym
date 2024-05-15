import yaml

for workload in {"vit_base", "vit_large", "gpt3_13b", "gpt3_175b"}:
    # b) ga
    steps_range = [512, 1024, 10000, 50000]
    num_agents_range = [16, 32, 64]
    prob_mut_range = [0.01, 0.05]
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
                        "WORKLOAD": f"astrasim_220_example/workload_cfg_{i}_{workload}.json"
                    }
                    
                    prob_str = ""
                    if prob == 0.01:
                        prob_str = '01'
                    elif prob == 0.05:
                        prob_str = '05'
                    file_name = f"{workload}/experiment{i}b_{step}_{num_agents}_{prob_str}_{workload}.yml"
                    
                    with open(file_name, "w") as file:
                        yaml.dump(cur_yml, file)
                        
                    print(file_name)

    # c) aco
    steps_range = [128, 1024, 10000, 50000]
    ant_count = [8, 16, 32]
    greediness_range = [0.25, 0.5, 0.75]
    evaporation_range = [0.25, 0.5, 0.75]

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
                            "WORKLOAD": f"astrasim_220_example/workload_cfg_{i}_{workload}.json"
                        }
                        greed_str = str(int(100 * greed))
                        eva_str = str(int(100 * eva))
                        file_name = f"{workload}/experiment{i}c_{step}_{ant}_{greed_str}_{eva_str}_{workload}.yml"
                        
                        with open(file_name, "w") as file:
                            yaml.dump(cur_yml, file)
                            
                        print(file_name)


    # d) bo
    steps_range = [128, 512, 1024, 10000]
    rand_state = [1, 2, 3, 4]
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
                    "WORKLOAD": f"astrasim_220_example/workload_cfg_{i}_{workload}.json"
                }
                
                file_name = f"{workload}/experiment{i}d_{step}_{rand}_{workload}.yml"
                
                with open(file_name, "w") as file:
                    yaml.dump(cur_yml, file)
                    
                print(file_name)