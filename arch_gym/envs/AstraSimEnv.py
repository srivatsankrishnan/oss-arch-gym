import gym
from gym.utils import seeding
import numpy as np
import json
import subprocess
import os
import time
import csv
import random
import yaml

from envHelpers import helpers

# from sims/AstraSim/astrasim_archgym_public/dse, import MemoryEstimator
os.sys.path.insert(0, os.path.abspath('../../'))
from sims.AstraSim.astrasim_archgym_public.dse.test_memory_constraints import MemoryEstimator


settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
proj_root_path = os.path.dirname(os.path.dirname(settings_dir_path))
proj_dir_path = os.path.join(proj_root_path, "sims/AstraSim")


astrasim_archgym = os.path.join(proj_dir_path, "astrasim-archgym")
archgen_v1_knobs = os.path.join(astrasim_archgym, "dse/archgen_v1_knobs")
sim_path = os.path.join(proj_root_path, "sims", "AstraSim")

workload_cfg_to_et = os.path.join(sim_path, "astrasim_220_example/workload_cfg_to_et.py")

# define AstraSim version
VERSION = 2

# astra-sim environment
class AstraSimEnv(gym.Env):
    def __init__(self, knobs_spec, network, system, workload, rl_form=None, max_steps=5, num_agents=1, 
                reward_formulation="latency", reward_scaling=1, congestion_aware=True, dimension=None, seed=12234):
        self.rl_form = rl_form
        self.congestion_aware = congestion_aware
        self.helpers = helpers()
        self.knobs_spec, self.network, self.system, self.workload = knobs_spec, network, system, workload
        self.system_knobs, self.network_knobs, self.workload_knobs = self.helpers.parse_knobs_astrasim(self.knobs_spec)

        # only generate workload file if workload knobs given
        if self.workload_knobs == {}:
            self.generate_workload = "FALSE"
        else:
            self.generate_workload = "TRUE"

        # set parameters
        self.max_steps = max_steps
        self.counter = 0
        self.useful_counter = 0
        self.num_agents = num_agents
        self.reward_formulation = reward_formulation
        self.reward_scaling = reward_scaling
        self.seed = random.randint(0, 10000) if seed == 12234 else seed

        # goal of the agent is to find the average
        self.goal = 0
        self.init_positions = 0

        # set the reward, state, done, and info
        self.state = 0
        self.done = False
        self.info = {}

        # V1 networks, systems, and workloads folder
        self.networks_folder = os.path.join(sim_path, "astrasim-archgym/dse/archgen_v1_knobs/templates/network")
        self.workloads_folder = os.path.join(sim_path, "astrasim-archgym/themis/inputs/workload")
        self.systems_folder = os.path.join(sim_path, "astrasim-archgym/themis/inputs/system")

        # CONFIG = FILE WITH CHANGED KNOBS
        if VERSION == 1:
            pass
        else:
            self.exe_path = os.path.join(sim_path, "astrasim_220_example/run.sh")
            self.network_config = os.path.join(sim_path, "astrasim_220_example/network.yml")
            self.system_config = os.path.join(sim_path, "astrasim_220_example/system.json")
            if self.congestion_aware:
                self.astrasim_binary = os.path.join(sim_path, 
                "astrasim_archgym_public/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware")
            else:
                self.astrasim_binary = os.path.join(sim_path, 
                "astrasim_archgym_public/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware")

        # FILE = INITIAL INPUTS
        if VERSION == 1:
            self.network_file = os.path.join(self.networks_folder, "4d_ring_fc_ring_switch.json")
            self.system_file = os.path.join(self.systems_folder, "4d_ring_fc_ring_switch_baseline.txt")
            self.workload_files = [os.path.join(self.workloads_folder, "all_reduce/allreduce_0.65.txt")]
        else:
            self.network_file = os.path.join(sim_path, self.network)
            self.system_file = os.path.join(sim_path, self.system)
            self.workload_files = []
            # /home/archgym/workspace/aditi_jared/oss-arch-gym/sims/AstraSim/astrasim_220_example/workload_cfg_7_gpt3_13b.json,
            # astrasim_220_example/workload_cfg_7_gpt3_175b.json,
            # astrasim_220_example/workload_cfg_7_vit_base.json,astrasim_220_example/workload_cfg_7_vit_large.json

            # TODO: if workload_file is a list, then split it into different jsons
            self.workload_files = self.workload.split(',')
            for i, file in enumerate(self.workload_files):
                self.workload_files[i] = os.path.join(sim_path, file)
            print("ENV - self.workload_files: ", self.workload_files)
                    
            
        
        self.param_len = 0
        self.dimension = 0
        # if dimension is given, then it's a tunable knob
        if dimension:
            self.dimension = dimension
        elif VERSION == 1:
            with open(self.network_file, 'r') as file:
                data = json.load(file)
                self.dimension = data["dimensions-count"]
        # if dimension is not a tunable knob, then it's constant
        else:
            data = yaml.load(open(self.network_file), Loader=yaml.Loader)
            self.dimension = len(data["topology"])

        # add 1 if N/A or TRUE knob, else add dimensions
        print("self.param_len: ", self.param_len)
        print("system knobs: ", self.system_knobs)
        for key in self.system_knobs:
            if self.system_knobs[key][1] == "FALSE":
                self.param_len += self.dimension
            else:
                self.param_len += 1
        print("self.param_len: ", self.param_len)
        print("network knobs: ", self.network_knobs)
        for key in self.network_knobs:
            if key == "dimensions-count":
                continue
            if self.network_knobs[key][1] == "FALSE":
                self.param_len += self.dimension
            else:
                self.param_len += 1
        print("self.param_len: ", self.param_len)
        print("workload knobs: ", self.workload_knobs)
        for key in self.workload_knobs:
            if self.workload_knobs[key][1] == "FALSE":
                self.param_len += self.dimension
            else:
                self.param_len += 1

        # param_len = len(self.system_knobs) + len(self.network_knobs) + len(self.workload_knobs)
        print("dimensions: ", self.dimension)
        print("param_len: ", self.param_len)


        self.obs_len = 1
        if self.reward_formulation == "latency" or self.reward_formulation == "memory":
            self.obs_len = 1
        elif self.reward_formulation == "both":
            self.obs_len = 2
        elif self.reward_formulation == "runtime*cost":
            self.obs_len = 1


        # TODO: define observation shape based on reward flag
        if self.rl_form == 'sa1':
            # action space = set of all possible actions. Space.sample() returns a random action
            # observation space =  set of all possible observations
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.obs_len,), dtype=np.float32) # box is an array of shape len
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.param_len,), dtype=np.float32)

        # reproducing Themis with AstraSim 1.0
        elif self.rl_form == 'rl_themis':
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.obs_len,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(16)
        
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.obs_len,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.param_len,), dtype=np.float32)

        print("_____________________*****************************_____________________")

        self.reset()

        self.constraints, self.derived_knobs = self.helpers.parse_constraints_astrasim(self.knobs_spec)
        print("CONSTRAINTS: ", self.constraints)
        print("DERIVED KNOBS: ", self.derived_knobs)
        

    # reset function

    def reset(self):

        self.counter = 0

        # TODO: delete all memory trace files in proj_dir_path
        for root, dirs, files in os.walk(proj_dir_path):
            for file in files:
                # if file begins with local_mem_trace, delete it
                if file.startswith("local_mem_trace"):
                    os.remove(os.path.join(root, file))

        # TODO: delete workload files
        workload_dir = os.path.join(sim_path, "astrasim_220_example/workload-et")
        workload_dir1 = os.path.join(sim_path, "astrasim_220_example/workload-et1")
        # if workload_dir exists, delete all files in it
        if os.path.exists(workload_dir):
            for root, dirs, files in os.walk(workload_dir):
                for file in files:
                    os.remove(os.path.join(root, file))
        if not os.path.exists(workload_dir):
            os.makedirs(workload_dir)

        if os.path.exists(workload_dir1):
            for root, dirs, files in os.walk(workload_dir1):
                for file in files:
                    os.remove(os.path.join(root, file))
        if not os.path.exists(workload_dir1):
            os.makedirs(workload_dir1)


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
        if observations[0] == '0.e+00' or float(observations[0]) == 0:
            if self.rl_form == 'bo':
                return np.format_float_scientific(-1000000000000000.0)
            else:
                return float("-inf")
        sum = 1.0
        for obs in observations:
            sum += ((float(obs) - 1) ** 2)
            print(sum)
        return 1 / (sum ** 0.5)

        # 10^12 for cycles (nanosecond)
        # 10^12 for peak memory (bytes)

    # give it one action: one set of parameters from json file
    def step(self, action_dict):
        # the action is actually the parsed parameter files
        print("Step: " + str(self.counter))
        print("REWARD FORMULATION: ", self.reward_formulation)

        max_cycles_arr = []
        max_peak_mem_arr = []
        network_cost_arr = []

        # stop if maximum steps reached
        if (self.counter == self.max_steps):
            self.done = True
            print("self.counter: ", self.counter)
            print("Maximum steps reached")
        
        # TODO: Now we have the self.workload_files, generate workload for initial step
        for i, cur_workload_file in enumerate(self.workload_files):
            new_workload_path = cur_workload_file.split('/')
            new_workload_file = new_workload_path[-1]
            workload_cfg = new_workload_file
            workload_et = "workload-et/generated.%d.et"
            print('CUR_WORKLOAD_FILE:', cur_workload_file)
            if i == 1 and ("qa" in cur_workload_file.lower() or "chat" in cur_workload_file.lower()):
                workload_et = "workload-et1/generated.%d.et"
            
            workload_command = []

            # TODO: parse the workload file from each workload json file
            # if not step 1, we need to skip over the variables that are in workload knobs
            print(f"ENV - step {self.counter} {i}th workload: {new_workload_file}")
            print("ENV - PARSE ACTION_DICT before: ", action_dict['workload'])
            if self.generate_workload == "TRUE":
                action_dict['workload'] = self.helpers.parse_workload_astrasim(cur_workload_file, action_dict, VERSION, self.workload_knobs)
            else:
                action_dict['workload'] = {"path": cur_workload_file}
            print("ENV - PARSE ACTION_DICT after: ", action_dict['workload'])
            
            if (self.counter == 0) and (self.generate_workload == "FALSE"):
                print("GENERATING WORKLOAD...")
                workload_command = ['python', workload_cfg_to_et, f'--workload_cfg={workload_cfg}', f'--workload_et={workload_et}']
                subprocess.run(workload_command)

            """ RL """
            # [0.5, 0.5, 0.5]
            if not isinstance(action_dict, dict):

                action_dict_decoded = {}

                # only generate workload if knobs exist
                if self.generate_workload == "TRUE":
                    action_dict_decoded['workload'] = self.helpers.parse_workload_astrasim(cur_workload_file, action_dict_decoded, VERSION)
                else:
                    action_dict_decoded['workload'] = {"path": cur_workload_file}
                
                # parse system: initial values
                action_dict_decoded['network'] = self.helpers.parse_network_astrasim(self.network_file, action_dict_decoded, VERSION)
                action_dict_decoded['system'] = self.helpers.parse_system_astrasim(self.system_file, action_dict_decoded, VERSION)

                print('system knobs: ', self.system_knobs)
                print('network knobs: ', self.network_knobs)
                print('workload knobs: ', self.workload_knobs)

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
                if "dimensions-count" in self.network_knobs:
                    action_dict["network"]["dimensions-count"] = self.dimension

                print("action_decoded: ", action_decoded)
                print("action_dict: ", action_dict)

            if "path" in action_dict["network"]:
                self.network_config = action_dict["network"]["path"]

            if "path" in action_dict["system"]:
                self.system_config = action_dict["system"]["path"]

            if "path" in action_dict["workload"]:
                cur_workload_file = action_dict["workload"]["path"]

            # derived knobs configurations:
            for cur_knob in self.derived_knobs:
                # bandwidth
                if cur_knob == "network bandwidth":
                    knob_arr = ["" for i in range(self.dimension)]
                    topology = action_dict["network"]["topology"]
                    for i in range(self.dimension):
                        if topology[i] == "Ring":
                            knob_arr[i] = 50
                        elif topology[i] == "Switch":
                            knob_arr[i] = 100
                        elif topology[i] == "FullyConnected":
                            knob_arr[i] = 100 / (action_dict["network"]["npus-count"][i] - 1)
                    action_dict["network"]["bandwidth"] = knob_arr
                    print("network bandwidth: ", action_dict["network"]["bandwidth"])

                elif cur_knob in {"system all-reduce-implementation", "system all-gather-implementation",
                                "system reduce-scatter-implementation", "system all-to-all-implementation"}:
                    knob_arr = ["" for i in range(self.dimension)]
                    topology = action_dict["network"]["topology"]
                    for i in range(self.dimension):
                        if topology[i] == "Ring":
                            knob_arr[i] = "ring"
                        elif topology[i] == "FullyConnected":
                            knob_arr[i] = "direct"
                        elif topology[i] == "Switch":
                            knob_arr[i] = "halvingDoubling"

                    k = cur_knob.split(" ")[1]
                    action_dict["system"][k] = knob_arr
                    print("system knob: ", action_dict["system"][k])
                
                elif cur_knob == "system implementations":
                    all_reduce = action_dict["system"]["all-reduce-implementation"]
                    action_dict["system"]["all-gather-implementation"] = all_reduce
                    action_dict["system"]["reduce-scatter-implementation"] = all_reduce
                    action_dict["system"]["all-to-all-implementation"] = all_reduce

                elif cur_knob == "network bandwidth-links":
                    nlinks = [2, 7, 2, 1]
                    for i in range(len(nlinks)):
                        action_dict["network"]["bandwidth"][i] /= nlinks[i]
                
                # 4/5/2025: exploring bandwidth
                elif cur_knob == "network bandwidth divided":
                    topology_links = {"Ring": 2, "FullyConnected": 7, "Switch": 1}
                    for i in range(len(action_dict["network"]["bandwidth"])):
                        topo = action_dict["network"]["topology"][i]
                        action_dict["network"]["bandwidth"][i] /= topology_links[topo]

                    
            print("DERIVED action_dict: ", action_dict)

            # write system to json file
            if "path" not in action_dict["system"]:
                with open(self.system_config, 'w') as file:
                    file.write('{\n')
                    for key, value in action_dict["system"].items():
                        if "dimensions-count" in action_dict["network"]:
                            if isinstance(value, list) and key not in self.system_knobs:
                                while len(value) < action_dict["network"]["dimensions-count"]:
                                    value.append(value[0])
                                while len(value) > action_dict["network"]["dimensions-count"]:
                                    value.pop()

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
            # write network to yaml file
            if "path" not in action_dict["network"]:
                data = {}
                for key, value in action_dict["network"].items():
                    if key == "dimensions-count":
                        continue
                    key_split = key.split("-")
                    key_converted = ""
                    # npus_count_
                    for k in key_split:
                        key_converted += k 
                        key_converted += "_"
                    if "dimensions-count" in action_dict["network"]:
                        if isinstance(value, list) and key not in self.network_knobs:
                            while len(value) < action_dict["network"]["dimensions-count"]:
                                value.append(value[0])
                            while len(value) > action_dict["network"]["dimensions-count"]:
                                value.pop()

                    data[key_converted[:-1]] = value

                with open(self.network_config, 'w') as file:
                    yaml.dump(data, file, sort_keys=False)

            # write workload to cfg file
            # TODO: 11/15, check the significance of this block
            # at each steps, we generate the workloads 4 times, and we take the average of the 4 results from those
            if "path" not in action_dict["workload"]:
                with open(cur_workload_file, 'w') as file:
                    file.write('{\n')
                    for key, value in action_dict["workload"].items():
                        file.write(f'"{key}": {value},\n')
                    file.seek(file.tell() - 2, os.SEEK_SET)
                    file.write('\n')
                    file.write('}')


            operators = {"<=", ">=", "==", "<", ">"}
            command = {"product", "mult", "num"}
            # self.constraints = []
            for constraint in self.constraints:
                print("!!!!!! CONSTRAINT: ", constraint)
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
                            elif "path" in action_dict[knob_dict]:
                                print("path in action dict", action_dict[knob_dict]["path"])
                                # load the file
                                if knob_dict == "network":
                                    data = yaml.load(open(action_dict[knob_dict]["path"]), Loader=yaml.Loader)
                                    knob_arr = np.array(data[knob_name])
                                elif knob_dict == "system" or knob_dict == "workload":
                                    with open(action_dict[knob_dict]["path"], 'r') as file:
                                        data = json.load(file)
                                        knob_arr = np.array(data[knob_name])
                                        cur_val = np.prod(knob_arr)
                                else:
                                    print(f"___ERROR: constraint knob name {knob_name} not found____")
                                    continue
                            else:
                                print(f"___ERROR: constraint knob name {knob_name} not found____")
                                continue

                        elif arg == "mult":
                            # mult workload data-parallel-degree workload model-parallel-degree == network num-npus
                            cur_val = 1

                            while constraint_args[i+1] not in operators and constraint_args[i+2] not in operators:
                                cur_knob_dict, cur_knob_name = constraint_args[i+1], constraint_args[i+2]

                                if (cur_knob_dict in action_dict and cur_knob_name in action_dict[cur_knob_dict]):
                                    cur_val *= action_dict[cur_knob_dict][cur_knob_name]
                                elif "path" in action_dict[cur_knob_dict]:
                                    # load the file
                                    if cur_knob_dict == "network":
                                        data = yaml.load(open(action_dict[cur_knob_dict]["path"]), Loader=yaml.Loader)
                                        cur_val *= data[cur_knob_name]
                                    elif cur_knob_dict == "system" or cur_knob_dict == "workload":
                                        with open(action_dict[cur_knob_dict]["path"], 'r') as file:
                                            data = json.load(file)
                                            cur_val *= data[cur_knob_name]
                                    else:
                                        print(f"___ERROR: constraint knob name {cur_knob_name} not found____")
                                        break
                                else:
                                    print(f"___ERROR: constraint knob name {cur_knob_name} not found____")
                                    break
                                i += 2

                        elif arg == "num":
                            # num network npus-count <= num network num-npus
                            knob_dict, knob_name = constraint_args[i+1], constraint_args[i+2]
                            i += 2
                            if knob_dict in action_dict and knob_name in action_dict[knob_dict]:
                                cur_val = action_dict[knob_dict][knob_name]
                            elif "path" in action_dict[knob_dict]:
                                # load the file
                                if knob_dict == "network":
                                    data = yaml.load(open(action_dict[knob_dict]["path"]), Loader=yaml.Loader)
                                    cur_val = data[knob_name]
                                elif knob_dict == "system" or knob_dict == "workload":
                                    with open(action_dict[knob_dict]["path"], 'r') as file:
                                        data = json.load(file)
                                        cur_val = data[knob_name]
                                else:
                                    print(f"___ERROR: constraint knob name {knob_name} not found____")
                                    continue
                            else:
                                print(f"___ERROR: constraint knob name {knob_name} not found____")
                                continue
                    i += 1
                
                # evaluate the constraint
                right_val = cur_val
                evaluable = str(left_val) + " " + str(operator) + " " + str(right_val)
                print("EVALUABLE: ", evaluable)
                if eval(evaluable):
                    print("constraint satisfied")
                    continue
                else:
                    print("constraint not satisfied")
                    if self.rl_form == 'bo':
                        reward = np.format_float_scientific(-1000000000000000.0)
                    else:
                        reward = float("-inf")
                    observations = [float("inf")] * self.obs_len
                    observations = np.reshape(observations, self.observation_space.shape)
                    return observations, reward, self.done, {"useful_counter": self.useful_counter}, self.state
                    # return [], reward, self.done, {"useful_counter": self.useful_counter}, self.state
            
            # TODO: Will's cost model new constraint, first topology cannot be switch else segfault
            if action_dict["network"]["topology"][0].lower() == "switch":
                print("ENV - SWITCH topology[0] constraint violated")
                if self.rl_form == 'bo':
                    reward = np.format_float_scientific(-1000000000000000.0)
                else:
                    reward = float("-inf")
                observations = [float("inf")] * self.obs_len
                observations = np.reshape(observations, self.observation_space.shape)
                return observations, reward, self.done, {"useful_counter": self.useful_counter}, self.state

            # TODO: this is where the workload is generated as et files
            if self.generate_workload == "TRUE":
                print("GENERATING WORKLOAD...")
                workload_command = ['python', workload_cfg_to_et, f'--workload_cfg={workload_cfg}', f'--workload_et={workload_et}']
                subprocess.run(workload_command)

            # start subrpocess to run the simulation
            # $1: network, $2: system, $3: workload
            print("Running simulation...")

            # load the system_config, network_config, and workload_file
            sys = json.load(open(self.system_config))
            net = yaml.load(open(self.network_config), Loader=yaml.Loader)
            work = json.load(open(cur_workload_file))
            print("all configs: ", sys, net, work)

            print("ENV - self.exe_path, network, system, workload_file: ", 
                self.exe_path, self.network_config, self.system_config, cur_workload_file)
            process = subprocess.Popen([self.exe_path, 
                                        self.astrasim_binary, 
                                        self.system_config, 
                                        self.network_config, 
                                        self.generate_workload],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # get the output
            out, err = process.communicate()
            print("OUT: ", out)
            outstream = out.decode()
            print("------------------------------------------------------------------")
            print("standard output: ")
            print(outstream)

            max_cycles = 0
            max_peak_mem = 0
            network_cost = 0
            if VERSION == 2:
                # parse to get the number of cycles
                for line in outstream.splitlines():
                    if ("sys[" in line) and ("] finished," in line) and ("cycles" in line):
                        lb = line.find("finished,") + len("finished,")
                        rb = line.rfind("cycles")
                        cycles = line[lb:rb].strip().replace(",", "")
                        cycles = int(cycles)
                        max_cycles = max(cycles, max_cycles)
                    elif ("Network Cost:" in line):
                        network_cost = line.split("Network Cost: ")[-1].strip()
                        network_cost = float(network_cost)
            
            # call memory estimator . get total memory 
            max_peak_mem = MemoryEstimator.get_total_memory(work)
            print("MAX_PEAK_MEM BEFORE MEMORY CAPACITY: ", max_peak_mem)

            # check with memory-capacity knob in system
            if "memory-capacity" in sys:
                if max_peak_mem > sys["memory-capacity"]:
                    print("Memory constraint violated")
                    if self.rl_form == 'bo':
                        reward = np.format_float_scientific(-1000000000000000.0)
                    else:
                        reward = float("-inf")
                    observations = [float("inf")] * self.obs_len
                    observations = np.reshape(observations, self.observation_space.shape)
                    return observations, reward, self.done, {"useful_counter": self.useful_counter}, self.state

            print("MAX_CYCLES: ", max_cycles)
            print("MAX_PEAK_MEM: ", max_peak_mem)
            print("NETWORK_COST: ", network_cost)
            
            # normalize QA AND CHAT WORKLOADS
            # normalization: QA_latency = QA_prefilling+QA_decoding*200, 
            # Chat_Latency = Chat_Prefilling+1000*Chat_Decoding
            if "qa_decoding" in cur_workload_file.lower():
                max_cycles = max_cycles*200
            elif "chat_decoding" in cur_workload_file.lower():
                max_cycles = max_cycles*1000

            max_cycles_arr.append(max_cycles)
            max_peak_mem_arr.append(max_peak_mem)
            network_cost_arr.append(network_cost)
            print("AVERAGE MAX_CYCLES ARRAY: ", max_cycles_arr)
            print("AVERAGE MAX_PEAK_MEM ARRAY: ", max_peak_mem_arr)
            print("AVERAGE NETWORK_COST ARRAY: ", network_cost_arr)
            print("------------------------------------------------------------------")
        
        max_cycles = 0
        max_peak_mem = 0
        # if decoding and prefilling
        if "decoding" or "prefilling" in self.workload_files[0]:
            max_cycles = sum(max_cycles_arr)
            max_peak_mem = max(max_peak_mem_arr)
        # These are average of the max_cycles across all workloads in the self.workload_files
        else:
            max_cycles = round(sum(max_cycles_arr) / len(max_cycles_arr), 2)
            max_peak_mem = round(sum(max_peak_mem_arr) / len(max_peak_mem_arr), 2)
        print("AVERAGE MAX_CYCLES: ", max_cycles)
        print("AVERAGE MAX_PEAK_MEM: ", max_peak_mem)
        print("AVERAGE MAX_CYCLES ARRAY: ", max_cycles_arr)
        print("AVERAGE MAX_PEAK_MEM ARRAY: ", max_peak_mem_arr)
        print("------------------------------------------------------------------")

        # TODO: THESE steps are for all models, calculate reward once
        observations = []

        if self.reward_formulation == "latency":
            observations = [np.format_float_scientific(max_cycles)]
            if self.rl_form == "sa1":
                observations = [float(max_cycles)]
        elif self.reward_formulation == "memory":
            observations = [np.format_float_scientific(max_peak_mem)]
            if self.rl_form == "sa1":
                observations = [float(max_peak_mem)]
        elif self.reward_formulation == "both":
            observations = [np.format_float_scientific(max_cycles), np.format_float_scientific(max_peak_mem)]
            if self.rl_form == "sa1":
                observations = [float(max_cycles), float(max_peak_mem)]
        elif self.reward_formulation == "runtime*cost":
            observations = [np.format_float_scientific(max_cycles*network_cost)]
            if self.rl_form == "sa1":
                observations = [float(max_cycles*network_cost)]
        elif self.reward_formulation == "runtime*bw":
            # astra-sim only takes PER-LINK bandwidth, so we have to multiply by 
            # topology link count in any context
            total_bandwidth = 0
            topology_links = {"Ring": 2, "FullyConnected": 7, "Switch": 1}
            for i in range(len(action_dict["network"]["bandwidth"])):
                topo = action_dict["network"]["topology"][i]
                total_bandwidth += topology_links[topo] * action_dict["network"]["bandwidth"]
            bandwidth_normalized = total_bandwidth / 2000

            print("RUNTIME*BW] NETWORK:": action_dict["network"])
            print("RUNTIME*BW] TOTAL_BANDWIDTH:", total_bandwidth)
            print("BANDWIDTH NORMALIZED:", bandwidth_normalized)
            
            observations = [np.format_float_scientific(max_cycles*bandwidth_normalized)]

        observations = np.reshape(observations, self.observation_space.shape)
        reward = self.calculate_reward(observations)

        print("observations: ", observations)
        print("reward: ", reward)

        ### LOG for RL ###
        if self.rl_form == "sa1":
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
            log_path = f"{sim_path}/all_logs/rl_logs/rl_form_{self.rl_form}_num_steps_{self.max_steps}_seed_{self.seed}.csv"
            with open(log_path, 'a') as f:
                writer = csv.writer(f)
                # write the timestamp and the action_dict in one row
                if self.obs_len == 2:
                    writer.writerow([timestamp, action_dict, observations[0], observations[1], reward])
                else:
                    writer.writerow([timestamp, action_dict, observations[0], reward])
        
        # reshape observations with shape of observation space
        observations = np.reshape(observations, self.observation_space.shape)
        self.counter += 1
        self.useful_counter += 1

        return observations, reward, self.done, {"useful_counter": self.useful_counter}, self.state
            

if __name__ == "__main__":
    print("Testing AstraSimEnv")
    env = AstraSimEnv(rl_form='sa1', 
                      max_steps=10, 
                      num_agents=1, 
                      reward_formulation='latency', 
                      reward_scaling=1)
