import sys
import numpy as np

from gym import Env, spaces
import subprocess
import os

class SimpleArch(Env):
    def __init__(self, target_vals, reward_type = 'both', max_steps = 5, log_type = 'number', workload = 'mcycle', target = 'digilent_arty'):
        super(SimpleArch, self).__init__()

        self.rewardType = reward_type
        self.target_val = target_vals
        self.no_steps=0
        self.max_steps = max_steps
        self.log_type = log_type
        self.workload = workload
        self.target = target
        self.Branch_predict_types = ['none', 'static', 'dynamic', 'dynamic_target']

        # clearing the file of its previous content and writing the column names.
        file = open('Env_logfile','w')
        file.write("Iteration Number, Action, Cycles, Cells, Reward\n")
        file.close()

        # define the observation space: cycles, cells
        self.observation_shape = (2,)
        self.observation_space = spaces.Box(low=0, high=sys.maxsize, shape=self.observation_shape, dtype=np.int32)
        
        # define the action space
        self.action_space = spaces.Tuple((
            spaces.Discrete(2),     
            spaces.Discrete(2),     
            spaces.Discrete(11),
            spaces.Discrete(2),    
            spaces.Discrete(11),
            spaces.Discrete(2),
            spaces.Discrete(4),
            spaces.Discrete(2),
            spaces.Discrete(2),     
            spaces.Discrete(2)
        ))

    def reset(self):
        self.no_steps=0

    # How many options for csr Plugin config?
    def step(self, action):

        #ensure that relative paths work
        Caller_wd = os.getcwd()
        os.chdir(os.path.dirname(__file__))

        #Update observations
        self.observation = self.runCFUPlaygroundEnv(action)

        #Calculate reward based on observations
        self.reward = self.calculate_reward()

        #Go bath to the original directory
        os.chdir(Caller_wd)

        self.no_steps += 1
        complete = False
        if (self.no_steps == self.max_steps):
            complete = True
            print("Max steps reached")

        self.CFUEnvLog()

        return self.observation, self.reward, complete, {}
    
    def calculate_reward(self):

        reward = 1e-3
        if(self.rewardType == 'cells'):
            reward = max(((self.observation[0] - self.target_val[0])/self.target_val[0]), 0)

        elif (self.rewardType == 'cycles'):
            reward = max(((self.observation[1] - self.target_val[1])/self.target_val[1]), 0)

        elif (self.rewardType == "both"):
            reward_cells = max(((self.observation[0] - self.target_val[0])/self.target_val[0]), 0)
            reward_cycles = max(((self.observation[1] - self.target_val[1])/self.target_val[1]), 0)
            reward = reward_cells*reward_cycles

        # assuming the algo gives error when reward is 0.
        if (reward == 0):
            reward = 1e-5

        return reward
        
    def runCFUPlaygroundEnv(self, action):

        # update action string to pass to subprocess
        self.action = self.workload
        
        self.action += ',' + str(action[0])         # Bypass
        #self.action += ',' + str(action[1])        # CFU_enable 
        self.action += ',0'                         # CFU_enable (currently set to false)
        self.action += ',' + ('0' if action[2] == 0 else str(1<<(4+action[2])))
                                                    # Data cache size
        self.action += ',' + str(action[3])         # Hardware Divider
        self.action += ',' + ('0' if action[4] == 0 else str(1<<(4+action[4])))
                                                    # Instruction cache size
        self.action += ',' + str(action[5])         # Hardware Multiplier
        self.action += ',' + self.Branch_predict_types[action[6]]
                                                    # Branch predictor
        self.action += ',' + str(action[7])         # Safe mode
        self.action += ',' + str(action[8])         # Single Cycle Shifter
        self.action += ',' + str(action[9])         # Single Cycle Multiplier
        self.action += ',' + self.target

        #Update communication file
        file = open('CFU_log', 'w')
        file.write(self.action)
        file.close()

        subprocess.run(['. ../../sims/CFU-Playground/env/conda/bin/activate cfu-symbiflow && python CFUPlaygroundWrapper.py'], shell = True, executable='/bin/bash')

        #update action to be stored in the format required by logger
        if (self.log_type == 'number'):
            action = [str[a] for a in action]
            self.action = self.workload + ',' + ','.join(action) + ',' + self.target
        
        #Read output from the script
        file = open('CFU_log', 'r')
        output = file.read()
        file.close()

        output = output.split()
        return [float(output[0]), int(output[1])]

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