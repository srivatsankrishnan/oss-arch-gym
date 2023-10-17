# !! This file now assumes that the cwd is sims/CFU_Playground.

import sys
import numpy as np

from gym import Env, spaces
import subprocess
import os

class CFUPlaygroundEnv(Env):
    def __init__(self, target_vals, max_steps, workload, reward_type = 'both', log_type = 'string', target = 'digilent_arty'):
        super(CFUPlaygroundEnv, self).__init__()

        self.rewardType = reward_type
        self.target_val = target_vals
        self.no_steps = 0
        self.max_steps = max_steps
        self.workload = workload
        self.log_type = log_type
        self.target = target
        self.Branch_predict_types = ['none', 'static', 'dynamic', 'dynamic_target']
        self.action = ""

        # clearing the file of its previous content and writing the column names.
        file = open('Env_logfile','w')
        file.write("Iteration Number, Action, Cycles, Cells, Reward\n")
        file.close()

        # define the observation space: cycles, cells
        self.observation_shape = (2,)
        self.observation_space = spaces.Box(low=0, high=1e12, shape=self.observation_shape)

        # define the action space
        self.action_space = spaces.Dict({
            "Bypass": spaces.Discrete(2),
            "CFU_enable": spaces.Discrete(2),
            "Data_cache_size": spaces.Discrete(11),
            "Hardware_Divider": spaces.Discrete(2),
            "Instruction_cache_size": spaces.Discrete(11),
            "Hardware_Multiplier": spaces.Discrete(2),
            "Branch_predictor_type": spaces.Discrete(4),
            "Safe_mode_enable": spaces.Discrete(2),
            "Single_Cycle_Shifter": spaces.Discrete(2),     
            "Single_Cycle_Multiplier": spaces.Discrete(2)
        })

    def reset(self):
        self.no_steps=0
        return

    def step(self, action):

        #ensure that relative paths work
        #Caller_wd = os.getcwd()
        #os.chdir(os.path.dirname(__file__))

        #Update observations
        self.observation = self.runCFUPlaygroundEnv(action)

        #Calculate reward based on observations
        self.reward = self.calculate_reward()

        #Go bath to the original directory
        #os.chdir(Caller_wd)

        self.no_steps += 1
        complete = False
        if (self.no_steps == self.max_steps):
            complete = True
            print("Max steps reached")

        self.CFUEnvLog()

        #currently assuming all iterations returned properly
        return self.observation, self.reward, complete, {"useful_counter": self.no_steps}
    
    def calculate_reward(self):

        reward = 1e-3
        if(self.rewardType == 'cells'):
            print('Reward Type: cells')
            reward = max(((self.observation[0] - self.target_val[0])/self.target_val[0]), 0)

        elif (self.rewardType == 'cycles'):
            print('Reward type: cycles')
            reward = max(((self.observation[1] - self.target_val[1])/self.target_val[1]), 0)

        elif (self.rewardType == 'both'):
            print('Reward type: both')
            reward_cells = max(((self.observation[0] - self.target_val[0])/self.target_val[0]), 0)
            reward_cycles = max(((self.observation[1] - self.target_val[1])/self.target_val[1]), 0)
            reward = reward_cells*reward_cycles

        # assuming the algo gives error when reward is 0.
        if (reward == 0):
            reward = 1e-5

        return reward
        
    def runCFUPlaygroundEnv(self, action):

        # update action string to pass to subprocess        
        self.action = str(action["Bypass"])
        #self.action += ',' + str(action["CFU_enable"])
        self.action += ',0'      # CFU_enable (currently set to false)
        self.action += ',' + ('0' if action["Data_cache_size"] == 0 else str(1<<(4+action["Data_cache_size"])))
        self.action += ',' + str(action["Hardware_Divider"])
        self.action += ',' + ('0' if action["Instruction_cache_size"] == 0 else str(1<<(4+action["Instruction_cache_size"])))
        self.action += ',' + str(action["Hardware_Multiplier"])
        self.action += ',' + self.Branch_predict_types[action["Branch_predictor_type"]]
        self.action += ',' + str(action["Safe_mode_enable"])
        self.action += ',' + str(action["Single_Cycle_Shifter"])
        self.action += ',' + str(action["Single_Cycle_Multiplier"])
        self.action += ',' + self.target
        self.action += ',' + self.workload

        #Update communication file
        file = open('CFU_log', 'w')
        file.write(self.action)
        file.close()

        subprocess.run(['. ./CFU-Playground/env/conda/bin/activate cfu-symbiflow && python run_CFU.py'], shell = True, executable='/bin/bash')

        #update action to be stored in the format required by logger
        if (self.log_type == 'number'):
            action = [str(a) for a in action]
            self.action = ','.join(action) + ',' + self.target + ',' + self.workload
        
        #Read output from the script
        file = open('CFU_log', 'r')
        output = file.read()
        file.close()

        output = output.split()
        try:
            return [np.float64(output[0]), np.int32(output[1])]
        except:
            return [1000.0, 1000]

    def CFUEnvLog(self):

        envlog_file = open('Env_logfile','a')

        #writing the iteration number into the log file
        envlog_file.write(str(self.no_steps)+',')

        #writing the action
        envlog_file.write(self.action+',')

        # writing cycles and cells.
        envlog_file.write(str(self.observation[0])+','+str(self.observation[1])+',')
        # writing reward
        envlog_file.write(str(self.reward)+"\n")        
        envlog_file.close()