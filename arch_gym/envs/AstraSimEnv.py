import gym
from gym.utils import seeding
import numpy as np
import json
import subprocess
import os
import time
import csv
import random

settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
proj_root_path = os.path.join(settings_dir_path, '..', '..')

sim_path = os.path.join(proj_root_path, "sims", "AstraSim")

# astra-sim environment
class AstraSimEnv(gym.Env):
    def __init__(self, rl_form="random_walker", max_steps=5, num_agents=1, reward_formulation="None", reward_scaling=1):
        # action space = set of all possible actions. Space.sample() returns a random action
        self.action_space = gym.spaces.Discrete(2)
        # observation space =  set of all possible observations
        self.observation_space = gym.spaces.Discrete(1)

        # set parameters
        self.max_steps = max_steps
        self.counter = 0
        self.useful_counter = 0

        self.rl_form = rl_form
        self.num_agents = num_agents
        self.reward_formulation = reward_formulation
        self.reward_scaling = reward_scaling

        # goal of the agent is to find the average
        self.goal = 0
        self.init_positions = 0

        # set the reward, state, done, and info
        self.state = 0
        self.done = False
        self.info = {}

        self.exe_path = os.path.join(sim_path, "run_general.sh")
        # self.network_config = os.path.join(sim_path, "general_network.json")
        self.system_config = os.path.join(sim_path, "general_system.txt")

        # V1 networks, systems, and workloads folder
        self.networks_folder = os.path.join(sim_path, "astrasim-archgym/dse/archgen_v1_knobs/templates/network")
        self.workloads_folder = os.path.join(sim_path, "astrasim-archgym/themis/inputs/workload")

        self.network_config = os.path.join(self.networks_folder, "3d_fc_ring_switch.json")
        self.workload_config = os.path.join(sim_path, "realworld_workloads/transformer_1t_fused_only_t.txt")


        print("_____________________*****************************_____________________")

        self.reset()

    # reset function

    def reset(self):
        self.counter = 0
        # get results folder path
        results_folder_path = os.path.join(sim_path, "results", "run_general")

        # # find wildcard csv and m files
        csv_files = [f for f in os.listdir(results_folder_path) if f.endswith('.csv')]

        # # remove the files
        for csv_files in csv_files:
            csv_files = os.path.join(results_folder_path, csv_files)
            if os.path.exists(csv_files):
                os.remove(csv_files)
        return

    # parses a result csv file and stores it in a dictionary
    def parse_result(self, file_name):
        try:
            result_dict = {}
            with open(file_name, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    for header in row:
                        if header not in result_dict:
                            result_dict[header] = []
                        if row[header] != '':
                            result_dict[header].append(row[header])
            return result_dict
        except:
            print("Error parsing file: " + file_name)
            return {}

    # randomize the network config
    def render(self):
        s = "position: {:2d} reward: {:2d} info: {}"
        print(s.format(self.position, self.reward, self.info))

    def seed(self):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass

    # reward only looks at first value of fw, ig, and wg compute
    def calculate_reward(self, observations):
        print("Calculating reward...")
        print(observations)
        sum = 0
        for obs in observations:
            sum += ((float(obs[0]) - 1) ** 2)
            print(sum)
        return 1 / (sum ** 0.5)

    # give it one action: one set of parameters from json file
    def step(self, action_dict):

        # write the three config files
        # with open(self.network_config, "w") as outfile:
        #     outfile.write(json.dumps(action_dict['network'], indent=4))
        if "path" in action_dict["network"]:
            self.network_config = action_dict["network"]["path"]

        if "path" in action_dict["workload"]:
            self.workload_config = action_dict["workload"]["path"]

        # load knobs
        print("system_config")
        print(action_dict["system"])
        with open(self.system_config, 'w') as file:
            for key, value in action_dict["system"].items():
                file.write(f'{key}: {value}\n')

        # the action is actually the parsed parameter files
        print("Step: " + str(self.counter))
        if (self.counter == self.max_steps):
            self.done = True
            print("Maximum steps reached")
            self.reset()
        else:
            self.counter += 1

        # start subrpocess to run the simulation
        # $1: network, $2: system, $3: workload
        print("Running simulation...")
        process = subprocess.Popen([self.exe_path, 
                                    self.network_config, 
                                    self.system_config, 
                                    self.workload_config],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # get the output
        out, err = process.communicate()
        outstream = out.decode()
        print("------------------------------------------------------------------")
        print(outstream)
        print("------------------------------------------------------------------")

        # backend_dim_info.csv
        backend_dim_info = self.parse_result(sim_path + 
            '/results/run_general/backend_dim_info.csv')
        # backend_end_to_end.csv
        backend_end_to_end = self.parse_result(sim_path + 
            '/results/run_general/backend_end_to_end.csv')
        # detailed.csv
        detailed = self.parse_result(sim_path +
            '/results/run_general/detailed.csv')
        # EndToEnd.csv
        end_to_end = self.parse_result(sim_path +
            '/results/run_general/EndToEnd.csv')
        # sample_all_reduce_dimension_utilization.csv
        sample_all_reduce_dimension_utilization = self.parse_result(sim_path +
            '/results/run_general/sample_all_reduce_dimension_utilization.csv')

        # test if the csv files exist (if they don't, the config files are invalid)
        if ((len(backend_dim_info) == 0 or len(backend_end_to_end) == 0 or
             len(detailed) == 0 or len(end_to_end) == 0 or
             len(sample_all_reduce_dimension_utilization) == 0)):
            # set reward to be extremely negative
            reward = -100000
            print("reward: ", reward)
            return [[], reward, self.done, {"useful_counter": self.useful_counter}, self.state]
        else:
            # only recording the first line because apparently they are all the same? TODO
            self.observations = [
                backend_end_to_end["CommsTime"][0],
                end_to_end["fwd compute"][0],
                end_to_end["wg compute"][0],
                end_to_end["ig compute"][0],
                end_to_end["total exposed comm"][0]
            ]
            reward = self.calculate_reward(self.observations)
            print("reward: ", reward)
            print("observations: ", self.observations)

            self.useful_counter += 1

            return [self.observations, reward, self.done, {"useful_counter": self.useful_counter}, self.state]


if __name__ == "__main__":
    print("Testing AstraSimEnv")
    env = AstraSimEnv(rl_form='random_walker', 
                      max_steps=10, 
                      num_agents=1, 
                      reward_formulation='reward_formulation_1', 
                      reward_scaling=1)






    """
    Everytime rest happens: 
    - zero out the observation

    3/24: 
    Communication Time (unit: microseconds)
    Time breakdowns (forward pass, weight gradient, input gradient)
    Exposed communication


    3/31: 
    Catch errors by giving it high negative reward. This way we can test the range. 
    

    """
