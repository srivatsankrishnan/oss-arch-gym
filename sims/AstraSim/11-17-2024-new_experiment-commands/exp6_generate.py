import yaml

for workload in ["Chat_decoding", "Chat_prefilling", "Chat", "QA_decoding", "QA_prefilling", "QA"]:
    # b) ga
    steps_range = [1024, 10000]
    num_agents_range = [32, 64]
    prob_mut_range = [0.01, 0.05]
    # Iterate through combinations and create YAML files
    for step in steps_range:
        for num_agents in num_agents_range:
            for prob in prob_mut_range:
                cur_yml = {
                    "ALGORITHM": "ga",
                    "STEPS": step,
                    "NUM_AGENTS": num_agents,
                    "PROB_MUT": prob,
                    "REWARD": "both",
                    "KNOBS": f"astrasim_220_example/knobs_6.py",
                    "NETWORK": f"astrasim_220_example/network_input_6.yml",
                    "SYSTEM": f"astrasim_220_example/system_input_6.json",
                    "WORKLOAD": f"astrasim_220_example/workload_cfg_6_gpt3_175b_{workload}.json"
                }
                
                prob_str = ""
                if prob == 0.01:
                    prob_str = '01'
                elif prob == 0.05:
                    prob_str = '05'
                file_name = f"exp6/experiment6b_{step}_{num_agents}_{prob_str}_{workload}.yml"
                
                with open(file_name, "w") as file:
                    yaml.dump(cur_yml, file)
                    
                print(file_name)

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
                    cur_yml = {
                        "ALGORITHM": "aco",
                        "STEPS": step,
                        "ANT_COUNT": ant,
                        "GREEDINESS": greed,
                        "EVAPORATION": eva,
                        "REWARD": "both",
                        "KNOBS": f"astrasim_220_example/knobs_6.py",
                        "NETWORK": f"astrasim_220_example/network_input_6.yml",
                        "SYSTEM": f"astrasim_220_example/system_input_6.json",
                        "WORKLOAD": f"astrasim_220_example/workload_cfg_6_gpt3_175b_{workload}.json"
                    }
                    greed_str = str(int(100 * greed))
                    eva_str = str(int(100 * eva))
                    file_name = f"exp6/experiment6c_{step}_{ant}_{greed_str}_{eva_str}_{workload}.yml"
                    
                    with open(file_name, "w") as file:
                        yaml.dump(cur_yml, file)
                        
                    print(file_name)


    # d) bo
    steps_range = [1024, 5000]
    rand_state = [1, 2, 3, 4]
    # Iterate through combinations and create YAML files
    for step in steps_range:
        for rand in rand_state:
            cur_yml = {
                "ALGORITHM": "bo",
                "STEPS": step,
                "RAND_STATE_BO": rand,
                "REWARD": "both",
                "KNOBS": f"astrasim_220_example/knobs_6.py",
                "NETWORK": f"astrasim_220_example/network_input_6.yml",
                "SYSTEM": f"astrasim_220_example/system_input_6.json",
                "WORKLOAD": f"astrasim_220_example/workload_cfg_6_gpt3_175b_{workload}.json"
            }
            
            file_name = f"exp6/experiment6d_{step}_{rand}_{workload}.yml"
            
            with open(file_name, "w") as file:
                yaml.dump(cur_yml, file)
                
            print(file_name)