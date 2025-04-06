import yaml

workload = "gpt3_175b"
for exp_num in ["1.1", "1.2", "1.3", "2.1", "2.2", "2.3", "3.1", "3.2", "3.3", "4.1", "4.2", "4.3"]:
    for reward_type in ["runtime*bw", "runtime*cost"]:
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
                        "REWARD": reward_type,
                        "KNOBS": f"astrasim_220_example/NEW_knobs_{exp_num[0]}.py",
                        "NETWORK": f"astrasim_220_example/NEW_network_input_{exp_num}.yml",
                        "SYSTEM": f"astrasim_220_example/NEW_system_input_{exp_num}.json",
                        "WORKLOAD": f"astrasim_220_example/NEW_workload_cfg_{exp_num}_{workload}.json"
                    }
                    
                    prob_str = ""
                    if prob == 0.01:
                        prob_str = '01'
                    elif prob == 0.05:
                        prob_str = '05'
                    file_name = f"exp{exp_num}/experiment{exp_num}b_{step}_{num_agents}_{prob_str}_{workload}_{reward_type}.yml"
                    
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
                    "REWARD": reward_type,
                    "KNOBS": f"astrasim_220_example/NEW_knobs_{exp_num[0]}.py",
                    "NETWORK": f"astrasim_220_example/NEW_network_input_{exp_num}.yml",
                    "SYSTEM": f"astrasim_220_example/NEW_system_input_{exp_num}.json",
                    "WORKLOAD": f"astrasim_220_example/NEW_workload_cfg_{exp_num}_{workload}.json"
                }
                
                file_name = f"exp{exp_num}/experiment{exp_num}d_{step}_{rand}_{workload}_{reward_type}.yml"
                
                with open(file_name, "w") as file:
                    yaml.dump(cur_yml, file)
                    
                print(file_name)