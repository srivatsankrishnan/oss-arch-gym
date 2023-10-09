import os

settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)

os.sys.path.insert(0, settings_dir_path)
os.sys.path.insert(0, settings_dir_path + '/../../configs')
os.sys.path.insert(0, settings_dir_path + '/../../configs/sniper')

from configs.sims import Sniper_config
from sims.Sniper  import simulate_benchmark

import gym
from gym.utils import seeding
from envHelpers import helpers
from loggers import write_csv
import numpy as np

import sys
import math

import subprocess
import time
import re
import numpy
import random
import json
import collections

class SniperEnv(gym.Env):
    def __init__(self):
        
        self.action_space = gym.spaces.Discrete(128)
        # Todo: Change the values if we normalize the observation space
        self.observation_space = gym.spaces.Box(low=0, high=1e10, shape=(1,11))
        
        self.steps = 0
        self.max_steps = 1000
        self.steps = 0
        self.max_episode_len = 10
        self.episode = 0

        self.binary_name = Sniper_config.sniper_binary_name
        self.binary_path = Sniper_config.sniper_binary_path
        self.sniper_config = Sniper_config.sniper_config
        self.sniper_workload = Sniper_config.spec_workload

        # For batch mode, we will pass unique logdir for each agent
        if(Sniper_config.sniper_mode == 'batch'):
            self.output_dirs = []
            self.agent_configs =[]
            self.cummulative_reward = []
            self.logdir = Sniper_config.sniper_logdir
        else:
            self.logdir = Sniper_config.sniper_logdir
            self.cummulative_reward = 0
        
        self.cores = Sniper_config.sniper_numcores
        
        self.helpers = helpers()
        #self.reset()

        self.cummulative_reward = 0
    
    def step_multiagent(self, actions):
        
        '''
        1) Take actions from all agents. 
        2) Launch a batch jobs for each agent
        3) Wait for all jobs to finish
        4) return the observation, reward, done, info for each agent
        '''
        self.steps += 1
        done = False
        agent_write_ok = []
        # create copies of the config files
        for agent_ids in range(len(actions)):
            self.agent_configs.append(self.helpers.create_agent_configs(
                                agent_ids,Sniper_config.sniper_config,
                                ))
                                
        # check if all the config files are created
        for each_config in self.agent_configs:
            if not os.path.exists(each_config):
                print("Error: Config file for agent {} does not exist.".format(each_config))
                sys.exit()
            else:
                print("Config file for agent {} exists.".format(each_config))
        
        # write the actions to the config files for each agent
        for each_action in range(len(self.agent_configs)):
            id = self.agent_configs[each_action].split("/")[-1].split(".")[0].split("_")[-1]
            agent_id = "agent_" + id        
            agent_action = actions[agent_id]
            
            agent_write_ok.append(self.actionToConfigs(agent_action,
                                  self.agent_configs[each_action]))
        
        # if configs are updated, then launch the sniper batch jobs
        if (False in agent_write_ok):
            print("Error: Not all the config files were updated")
            sys.exit()
        else:
            print("Running Sniper Batch Jobs")
            obs_dict = self.runSniperBatch(len(actions))

        # convert obs dict to list of list
        obs = []
        for each_obs in obs_dict:
            print(obs_dict[each_obs])
            obs.append(self.dict_to_obs(obs_dict[each_obs]))

        # Calculate the reward for each agent
        rewards = []
        for each_obs in obs:
            rewards.append(self.calculate_reward(each_obs))
        print(obs)
        return obs, rewards, done, {}
    
    def reset_multiagent(self):
      
        for each_config in self.agent_configs:
            os.remove(each_config)
        
        for each_output in self.output_dirs:
            print("Deleting Old logs!:", each_output)
            os.system("rm -rf {}".format(each_output))
        # sleep for a while to make sure the files are deleted
        time.sleep(60)


    def step(self, action):
        self.steps += 1

        done = False
        
        status = self.actionToConfigs(action,self.sniper_config)

        if(status):
            obs = self.runSniper()
            done = True
            reward = self.calculate_reward(obs)
        else:
            print(f"{bcolors.FAIL}Warning: No active frommets remain. Continue?{bcolors.ENDC}")
       
        
        # To do : Add some stopping conditions when dealing with real workload
        
        return obs, reward, done, {}
    
    def reset(self):
        print("Reseting Environment!")
        self.steps = 0
        self.cummulative_reward = 0
        self.obs = self.observation_space.sample()
        
        # Delete the logs to make sure every step is a new run
        if os.path.exists(self.logdir):
            print("Deleting Logs")
            subprocess.call(["rm", "-rf", self.logdir])

        
        return self.obs
    
    def actionToConfigs(self,action, cfg):

        '''
        Converts actions output from the agent to update the configuration files

        '''
        write_ok = self.helpers.read_modify_write_sniper_config(action,cfg)

        # Todo: Maybe there is a cleaner way to write the config file

        # workaround: add include to the config file each time we take a new action
        # Sniper seems to need this: https://groups.google.com/g/snipersim/c/bXvBb6SXZ0k

        dir_path = os.path.dirname(cfg)
        
        if os.path.exists(cfg):
            print("Adding include to config file")
                
            process = subprocess.Popen(["sed", "-i",'1i #include nehalem',cfg], stdout=subprocess.PIPE
                            ,stderr=subprocess.PIPE)
            out, err = process.communicate()
            print("Adding Include Nehalem")
            process = subprocess.Popen(["sed", "-i",'1i #include rob',cfg], stdout=subprocess.PIPE
                            ,stderr=subprocess.PIPE)                
            out, err = process.communicate()
        
        return write_ok
    
    def runSniperBatch(self, num_agents):
        '''
        Runs the sniper in batch mode
        '''

        done = False
        launcher = simulate_benchmark.SniperLauncher(int(self.cores))

        jobs = [Sniper_config.spec_workload for _ in range(num_agents)]
        results = []

        for agent_idx in range(len(jobs)):
            output_dir = 'agent_.{}_.{}'.format(agent_idx,jobs[agent_idx])
            self.output_dirs.append(output_dir)

            benchmark = jobs[agent_idx]


            # The callback function is optional.
            print("configs:", self.agent_configs[agent_idx])
            result = launcher.batch_benchmark(benchmark, 'CPU2017', output_dir, self.agent_configs[agent_idx], callback=self.create_callback(output_dir))
            results.append(result)
            print('Launched Agent{}_{}'.format(agent_idx,benchmark))

        for result in results:
            result.wait()
        
        for output_dir in self.output_dirs:
            try:
                # Now combine the stats.
                simulate_benchmark.combine_stats(output_dir)
            except Exception as e:
                # If the output files are not found, Python will throw an exception.  That can be gracefully handled here.
                print(e)
        
        obs = collections.defaultdict(dict)
        for idx in range(len(self.output_dirs)):
            basedir = os.path.join(Sniper_config.sniper_binary_path, self.output_dirs[idx])
            
            # rety this operation till stats.json file is created
            retry_count = 0
            while(retry_count < 10):
                if(os.path.exists(os.path.join(basedir, 'stats.json'))):
                    with open(os.path.join(basedir, 'stats.json')) as json_file:
                        data = json.load(json_file)
                        agent_name = "agent_" + str(idx)
                        obs[agent_name]['runtime'] = data['Time']
                        obs[agent_name]['branch_predictor_mpki'] = data["Branch Prediction"]["MPKI"]
                        obs[agent_name]['branch_mispredict_rate'] = data["Branch Prediction"]["misprediction rate"]
                        obs[agent_name]['l1_dcache_mpki'] = data["Cache"]["Cache L1-D"]["MPKI"]
                        obs[agent_name]['l1_dcache_missrate'] = data["Cache"]["Cache L1-D"]["miss rate"]
                        obs[agent_name]['l1_icache_mpki'] = data["Cache"]["Cache L1-I"]["MPKI"]
                        obs[agent_name]['l1_icache_missrate'] = data["Cache"]["Cache L1-I"]["miss rate"]
                        obs[agent_name]['l2_mpki'] = data["Cache"]["Cache L2"]["MPKI"]
                        obs[agent_name]['l2_missrate'] = data["Cache"]["Cache L2"]["miss rate"]
                        obs[agent_name]['l3_mpki'] = data["Cache"]["Cache L3"]["MPKI"]
                        obs[agent_name]['l3_missrate'] = data["Cache"]["Cache L3"]["miss rate"]
                        obs[agent_name]['power_dynamic'] = data["Power"]["Processor"]["Runtime Dynamic"]
                        obs[agent_name]['power_peak'] = data["Power"]["Processor"]["Peak Power"]
                        obs[agent_name]['area'] = data["Power"]["Processor"]["Area"]
                    break
                else:
                    retry_count += 1
                    time.sleep(1)
                    print("stats.json file is not present, retry count: ", retry_count)

            if(retry_count == 10):
                # write a message to a file
                with open(os.path.join(basedir, 'error.log'), 'w') as f:
                    f.write("stats.json file is not present! Shutting down!")

                    
        return obs       

    def runSniper(self):
        done = False
        exe_path = self.binary_path
        exe_name = self.binary_name
        exe_final = os.path.join(exe_path,exe_name)
        output_file = os.path.join(self.logdir,"stats.json")
        
        cmd = exe_final
        args = " -c" + " " + self.sniper_config + " -d " + self.logdir + " -n " + self.cores

        process = subprocess.check_output(["python", cmd, 
                 self.sniper_workload,
                 "-c", self.sniper_config,
                 "-d", self.logdir,
                 "-n", self.cores])

        #process = subprocess.Popen(["python", cmd, args],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #process = subprocess.check_output(["python", cmd, args])#,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #process.wait()
        
        #wait for the output file to be created and update done flag
        if os.path.exists(output_file):
            done = True
        else:
            done = False

        data = {}
        # read from a json file if done
        if done:
            with open(output_file) as json_file:
                data = json.load(json_file)
        else:
            # To do : Gracefully hanfl ethis case
            sys.exit("Error: Sniper did not finish")
        
        # read relevant data from the json file

        runtime = data["Time"]
        branch_predictor_mpki = data["Branch Prediction"]["MPKI"]
        branch_mispredict_rate = data["Branch Prediction"]["misprediction rate"]

        l1_dcache_mpki  = data["Cache"]["Cache L1-D"]["MPKI"]
        l1_dcache_missrate = data["Cache"]["Cache L1-D"]["miss rate"]
        l1_icache_mpki = data["Cache"]["Cache L1-I"]["MPKI"]
        l1_icache_missrate = data["Cache"]["Cache L1-I"]["miss rate"]
        l2_mpki = data["Cache"]["Cache L2"]["MPKI"]
        l2_missrate = data["Cache"]["Cache L2"]["miss rate"]
        l3_mpki = data["Cache"]["Cache L3"]["MPKI"]
        l3_missrate = data["Cache"]["Cache L3"]["miss rate"]

        obs = [runtime, branch_predictor_mpki, branch_mispredict_rate, l1_dcache_mpki, 
                l1_dcache_missrate, l1_icache_mpki, l1_icache_missrate, l2_mpki, l2_missrate, 
                l3_mpki, l3_missrate]

        return obs

    def calculate_reward(self,obs):
        '''
        Calculates the reward based on the current observation
        '''
        reward = 0

        latency = math.pow((obs[0] - Sniper_config.target_latency)/Sniper_config.target_latency,2)
        power = math.pow((obs[1]- Sniper_config.target_power)/Sniper_config.target_power,2)
        area = math.pow((obs[2]- Sniper_config.target_area)/Sniper_config.target_area,2)
        
        reward = math.sqrt(latency + power + area)
        return reward
    
    def create_callback(self, output_dir):
        return lambda res : simulate_benchmark.error_check(output_dir)

    def dict_to_obs(self, obs_dict):
        runtime = obs_dict["runtime"]
        branch_predictor_mpki = obs_dict['branch_predictor_mpki']
        branch_mispredict_rate = obs_dict['branch_mispredict_rate']

        l1_dcache_mpki  = obs_dict['l1_dcache_mpki']
        l1_dcache_missrate = obs_dict['l1_dcache_missrate']
        l1_icache_mpki = obs_dict['l1_icache_mpki']
        l1_icache_missrate = obs_dict['l1_icache_missrate']
        l2_mpki = obs_dict['l2_mpki']
        l2_missrate = obs_dict['l2_missrate']
        l3_mpki = obs_dict['l3_mpki']
        l3_missrate = obs_dict['l3_missrate']
        area = obs_dict['area']
        power = obs_dict['power_dynamic']

        return [runtime, power, area, branch_predictor_mpki, branch_mispredict_rate, l1_dcache_mpki,
                l1_dcache_missrate, l1_icache_mpki, l1_icache_missrate, l2_mpki, l2_missrate,
                l3_mpki, l3_missrate]


        
