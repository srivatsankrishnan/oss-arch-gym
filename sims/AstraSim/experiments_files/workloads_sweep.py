import yaml 

for workload in {"vit_base", "vit_large", "gpt3_13b", "gpt3_175b"}:
    for alg in {"rw", "ga", "aco", "bo"}:
        for i in range(1, 6, 1):
            if i == 2 and workload == "gpt3_175b":
                for peak_perf in ["2_10", "2_20", "2_40", "2_80"]:
                    steps = 0
                    if alg == "rw":
                        steps = 50000
                    elif alg == "bo":
                        steps = 10000
                    else:
                        steps = 20000

                    cur_yml = {
                        "ALGORITHM": alg,
                        "STEPS": steps,
                        "REWARD": "both",
                        "KNOBS": f"astrasim_220_example/knobs_{i}.py",
                        "NETWORK": f"astrasim_220_example/network_input_{i}.yml",
                        "SYSTEM": f"astrasim_220_example/system_input_{peak_perf}.json",
                        "WORKLOAD": f"astrasim_220_example/workload_cfg_{i}_{workload}.json"
                    }
                    
                    letter = ""
                    if alg == "rw":
                        letter = "a"
                    elif alg == "ga":
                        letter = "b"
                    elif alg == "aco":
                        letter = "c"
                    elif alg == "bo":
                        letter = "d"
                    file_name = f"{workload}/experiment{peak_perf}{letter}_{workload}.yml"
                    with open(file_name, "w") as file:
                        yaml.dump(cur_yml, file)
                        
                    print(file_name)

            elif i == 2:
                pass

            else:
                steps = 0
                if alg == "rw":
                    steps = 50000
                elif alg == "bo":
                    steps = 10000
                else:
                    steps = 20000

                cur_yml = {
                    "ALGORITHM": alg,
                    "STEPS": steps,
                    "REWARD": "both",
                    "KNOBS": f"astrasim_220_example/knobs_{i}.py",
                    "NETWORK": f"astrasim_220_example/network_input_{i}.yml",
                    "SYSTEM": f"astrasim_220_example/system_input_{i}.json",
                    "WORKLOAD": f"astrasim_220_example/workload_cfg_{i}_{workload}.json"
                }
                
                letter = ""
                if alg == "rw":
                    letter = "a"
                elif alg == "ga":
                    letter = "b"
                elif alg == "aco":
                    letter = "c"
                elif alg == "bo":
                    letter = "d"
                file_name = f"{workload}/experiment{i}{letter}_{workload}.yml"
                with open(file_name, "w") as file:
                    yaml.dump(cur_yml, file)
                    
                print(file_name)