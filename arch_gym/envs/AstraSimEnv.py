import gym
from gym.utils import seeding
import numpy as np
import json
import subprocess
import os
import time
import csv
import random

from envHelpers import helpers

settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
proj_root_path = os.path.dirname(os.path.dirname(settings_dir_path))
proj_dir_path = os.path.join(proj_root_path, "sims/AstraSim")

astrasim_archgym = os.path.join(proj_dir_path, "astrasim-archgym")
archgen_v1_knobs = os.path.join(astrasim_archgym, "dse/archgen_v1_knobs")
sim_path = os.path.join(proj_root_path, "sims", "AstraSim")
knobs_spec = os.path.join(archgen_v1_knobs, "archgen_v1_knobs_spec.py")
parameter_knobs= os.path.join(sim_path, "frontend/parameter_knobs.py")

# define AstraSim version
VERSION = 1

# astra-sim environment
class AstraSimEnv(gym.Env):
    def __init__(self, rl_form="sa1", max_steps=5, num_agents=1, reward_formulation="None", reward_scaling=1,):
        self.rl_form = rl_form
        self.helpers = helpers()
        self.system_knobs, self.network_knobs, self.workload_knobs = self.helpers.parse_knobs_astrasim(knobs_spec)

        # set parameters
        self.max_steps = max_steps
        self.counter = 0
        self.useful_counter = 0
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
        self.network_config = os.path.join(sim_path, "general_network.json")
        self.system_config = os.path.join(sim_path, "general_system.txt")

        # V1 networks, systems, and workloads folder
        self.networks_folder = os.path.join(sim_path, "astrasim-archgym/dse/archgen_v1_knobs/templates/network")
        self.workloads_folder = os.path.join(sim_path, "astrasim-archgym/themis/inputs/workload")

        # Config does not matter
        # self.network_config = os.path.join(self.networks_folder, "4d_ring_fc_ring_switch.json")
        self.workload_config = os.path.join(self.workloads_folder, "all_reduce/allreduce_0.65.txt")
        self.astrasim_archgym = os.path.join(sim_path, "astrasim-archgym")
        self.systems_folder = os.path.join(self.astrasim_archgym, "themis/inputs/system")

        self.network_file = os.path.join(self.networks_folder, "4d_ring_fc_ring_switch.json")
        self.system_file = os.path.join(self.systems_folder, "4d_ring_fc_ring_switch_baseline.txt")
        self.workload_file = "all_reduce/allreduce_0.65.txt"


        """TODO: ALCULATE LENGTH OF THE ACTION SPACE"""
        self.param_len = 0
        self.dimension = 0
        if "dimensions-count" in self.network_knobs.keys():
            self.dimension = self.network_knobs["dimensions-count"][0]
        else:
            # else get dimension from the network_file 
            with open(self.network_file, 'r') as file:
                data = json.load(file)
                self.dimension = data["dimensions-count"]

        # add 1 if N/A or TRUE knob, else add dimensions
        for key in self.system_knobs:
            if self.system_knobs[key][1] == "FALSE":
                self.param_len += self.dimension
            else:
                self.param_len += 1

        # param_len = len(self.system_knobs) + len(self.network_knobs) + len(self.workload_knobs)
        print("dimensions: ", self.dimension)
        print("param_len: ", self.param_len)

        if self.rl_form == 'sa1':
            # action space = set of all possible actions. Space.sample() returns a random action
            # observation space =  set of all possible observations
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32) # box is an array of shape len
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.param_len,), dtype=np.float32)

        # reproducing Themis with AstraSim 1.0
        elif self.rl_form == 'rl_themis':
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(16)
        
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.param_len,), dtype=np.float32)

        print("_____________________*****************************_____________________")

        self.reset()

        """TODO: constraints"""
        _, _, _, self.constraints = self.helpers.actual_parse_knobs_astrasim(parameter_knobs)
        print("CONSTRAINTS: ", self.constraints)
        

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

        # TODO: 
        obs = np.zeros(self.observation_space.shape)

        return obs



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
        sum = 1.0
        for obs in observations:
            sum += ((float(obs) - 1) ** 2)
            print(sum)
        return 1 / (sum ** 0.5)


    # give it one action: one set of parameters from json file
    def step(self, action_dict):

        """ RL """
        if not isinstance(action_dict, dict):
            with open(settings_dir_path + "/AstraSimRL_2.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow(action_dict)

            print("STEP: action_dict is a list")
            action_dict_decoded = {}
            action_dict_decoded['network'] = {"path": self.network_file}
            action_dict_decoded['workload'] = {"path": self.workload_file}
            
            # parse system: initial values
            self.helpers.parse_system_astrasim(self.system_file, action_dict_decoded, VERSION)
            self.helpers.parse_network_astrasim(self.network_file, action_dict_decoded, VERSION)

            # returning an 
            print("ACTION DICT")
            print(action_dict)
            print("tunable knobs: ")
            print(self.system_knobs)
            print(self.network_knobs)
            print(self.workload_knobs)

            action_decoded = self.helpers.action_decoder_rl_astraSim(action_dict, 
                                                                     self.system_knobs, 
                                                                     self.network_knobs, 
                                                                     self.workload_knobs, 
                                                                     self.dimension)



            # change all variables decoded into action_dict
            for sect in action_decoded:
                for key in action_decoded[sect]:
                    action_dict_decoded[sect][key] = action_decoded[sect][key]

            action_dict = action_dict_decoded

        if "path" in action_dict["network"]:
            self.network_config = action_dict["network"]["path"]

        if "path" in action_dict["system"]:
            self.system_config = action_dict["system"]["path"]

        if "path" in action_dict["workload"]:
            self.workload_config = action_dict["workload"]["path"]

        print("ACTION DICT")
        print(action_dict)
        # load knobs

        if VERSION == 1:
            with open(self.system_config, 'w') as file:
                for key, value in action_dict["system"].items(): 
                    if isinstance(value, list):
                        file.write(f'{key}: ')
                        for i in range(len(value)-1):
                            file.write(f'{value[i]}_')
                        file.write(f'{value[len(value)-1]}')
                        file.write('\n')
                    else:
                        file.write(f'{key}: {value}\n')
            with open(self.network_config, 'w') as file:
                file.write('{\n')
                for key, value in action_dict["network"].items():
                    if isinstance(value, str):
                        file.write(f'"{key}": "{value}",\n')
                    elif isinstance(value, list) and isinstance(value[0], str):
                        file.write(f'"{key}": [')
                        for i in range(len(value)-1):
                            file.write(f'"{value[i]}", ')
                        file.write(f'"{value[len(value)-1]}"')
                        file.write('],\n')
                    else:
                        file.write(f'"{key}": {value},\n')
                file.seek(file.tell() - 2, os.SEEK_SET)
                file.write('\n')
                file.write('}')
        elif VERSION == 2:
            """
            TODO: ASTRA-sim 2.0 integration
            """
            with open(self.system_config, 'w') as file:
                file.write('{\n')
                for key, value in action_dict["system"].items():
                    if isinstance(value, str):
                        file.write(f'"{key}": "{value}",\n')
                    elif isinstance(value, list) and isinstance(value[0], str):
                        file.write(f'"{key}": [')
                        for i in range(len(value)-1):
                            file.write(f'"{value[i]}", ')
                        file.write(f'"{value[len(value)-1]}"')
                        file.write('],\n')
                    else:
                        file.write(f'"{key}": {value},\n')
                file.seek(file.tell() - 2, os.SEEK_SET)
                file.write('\n')
                file.write('}')
            # WRITE NETWORK FILE TO YAML FILE

        # the action is actually the parsed parameter files
        print("Step: " + str(self.counter))
        self.counter += 1

        # start subrpocess to run the simulation
        # $1: network, $2: system, $3: workload
        print("Running simulation...")
        print(self.exe_path, self.network_config, self.system_config, self.workload_config)
        process = subprocess.Popen([self.exe_path, 
                                    self.network_file, 
                                    self.system_config, 
                                    self.workload_file],
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

        if (self.counter == self.max_steps):
            self.done = True
            print("Maximum steps reached")
            self.reset()

        """
        TODO: add constraints
        """

        operators = {"<=", ">=", "==", "<", ">"}
        command = {"product", "mult", "num"}
        for constraint in self.constraints:
            constraint_args = constraint.split(" ")
            
            left_val, right_val = None, None
            cur_val = None
            operator = ""

            # parse the arguments of a single constraint
            i = 0
            while i < len(constraint_args):
                arg = constraint_args[i]

                if arg in operators:
                    left_val = cur_val
                    cur_val = None
                    operator = arg

                elif arg in command:
                    if arg == "product":
                        # product network npus-count <= num network num-npus
                        knob_dict, knob_name = constraint_args[i+1], constraint_args[i+2]
                        i += 2

                        if knob_dict in action_dict and knob_name in action_dict[knob_dict]:
                            knob_arr = np.array(action_dict[knob_dict][knob_name])
                            cur_val = np.prod(knob_arr)
                        else:
                            print(f"___ERROR: constraint knob name {knob_name} not found____")
                            continue

                    elif arg == "mult":
                        # mult workload data-parallel-degree workload model-parallel-degree == network num-npus
                        left_knob_dict, left_knob_name = constraint_args[i+1], constraint_args[i+2]
                        right_knob_dict, right_knob_name = constraint_args[i+3], constraint_args[i+4]
                        i += 4

                        if (left_knob_dict in action_dict and 
                            left_knob_name in action_dict[left_knob_dict] and
                            right_knob_dict in action_dict and 
                            right_knob_name in action_dict[right_knob_dict]):
                            
                            cur_val = (action_dict[left_knob_dict][left_knob_name] * 
                                       action_dict[right_knob_dict][right_knob_name])
                        else:
                            print(f"___ERROR: constraint knob name {knob_name} not found____")
                            continue

                    elif arg == "num":
                        # num network npus-count <= num network num-npus
                        knob_dict, knob_name = constraint_args[i+1], constraint_args[i+2]
                        i += 2
                        if knob_dict in action_dict and knob_name in action_dict[knob_dict]:
                            cur_val = action_dict[knob_dict][knob_name]
                        else:
                            print(f"___ERROR: constraint knob name {knob_name} not found____")
                            continue
                i += 1
            
            # evaluate the constraint
            right_val = cur_val
            evaluable = str(left_val) + " " + str(operator) + " " + str(right_val)
            if eval(evaluable):
                print("constraint satisfied")
                continue
            else:
                print("constraint not satisfied")
                reward = float("-inf")
                return [], reward, self.done, {"useful_counter": self.useful_counter}, self.state
            

        # HARDCODED EXAMPLE: test if product of npu count <= number of npus
        # if np.prod(action_dict["network"]["npus-count"]) > action_dict["network"]["num-npus"]:
        #     # set reward to be extremely negative
        #     reward = float("-inf")
        #     print("reward: ", reward)
        #     return [], reward, self.done, {"useful_counter": self.useful_counter}, self.state

        # test if the csv files exist (if they don't, the config files are invalid)
        if ((len(backend_dim_info) == 0 or len(backend_end_to_end) == 0 or
             len(detailed) == 0 or len(end_to_end) == 0 or
             len(sample_all_reduce_dimension_utilization) == 0)):
            # set reward to be extremely negative
            reward = float("-inf")
            print("reward: ", reward)
            return [], reward, self.done, {"useful_counter": self.useful_counter}, self.state
        else:
            observations = [
                float(backend_end_to_end["CommsTime"][0])
                # end_to_end["fwd compute"][0],
                # end_to_end["wg compute"][0],
                # end_to_end["ig compute"][0],
                # end_to_end["total exposed comm"][0]
            ]

            
            reward = self.calculate_reward(observations)
    
            print("reward: ", reward)
            
            # reshape observations with shape of observation space
            observations = np.reshape(observations, self.observation_space.shape)
            self.useful_counter += 1

            return observations, reward, self.done, {"useful_counter": self.useful_counter}, self.state


if __name__ == "__main__":
    print("Testing AstraSimEnv")
    env = AstraSimEnv(rl_form='sa1', 
                      max_steps=10, 
                      num_agents=1, 
                      reward_formulation='reward_formulation_1', 
                      reward_scaling=1)
