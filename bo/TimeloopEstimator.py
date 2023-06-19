import copy
import time
from sims.Timeloop.timeloop_wrapper import TimeloopWrapper
from sims.Timeloop.process_params import TimeloopConfigParams
import pandas as pd
import numpy as np
from arch_gym.envs.TimeloopEnv import TimeloopEnv
from arch_gym.envs.timeloop_acme_wrapper import make_timeloop_env
from arch_gym.envs.envHelpers import helpers
from sklearn.base import BaseEstimator
from absl import logging
import configs.arch_gym_configs as arch_gym_configs
import envlogger
import configparser
import os


class TimeloopEstimator(BaseEstimator):

    def __init__(self, script=None, traj_dir=None, exp_name=None, output_dir=None, log=None, reward_formulation="energy", use_envlogger=False, arch=None, workload=None,
                 mapper_dir = None, target_area=0.0, target_cycles=0, target_energy=0.0,  **params):
        '''
        Initialize default values for all the parameters in Timeloop
        '''

        self.timeloop = TimeloopWrapper()
        

        self.timeloop_config = TimeloopConfigParams(arch_gym_configs.timeloop_parameters)

        # self.arch_params = self._convert_flat_arch_params(
        #     self.timeloop.get_arch_param_template(), params)
        
        self.action_dict = {}

        self.action_dict['NUM_PEs'] = params['NUM_PEs']
        self.action_dict['MAC_MESH_X'] = params['MAC_MESH_X']
        self.action_dict['IFMAP_SPAD_CLASS'] = params['IFMAP_SPAD_CLASS']
        self.action_dict['IFMAP_SPAD_ATRIBUTES.memory_depth'] = params['IFMAP_SPAD_ATRIBUTES.memory_depth']
        self.action_dict['IFMAP_SPAD_ATRIBUTES.block-size'] = params['IFMAP_SPAD_ATRIBUTES.block-size']
        self.action_dict['IFMAP_SPAD_ATRIBUTES.read_bandwidth'] = params['IFMAP_SPAD_ATRIBUTES.read_bandwidth']
        self.action_dict['IFMAP_SPAD_ATRIBUTES.write_bandwidth'] = params['IFMAP_SPAD_ATRIBUTES.write_bandwidth']
        self.action_dict['PSUM_SPAD_CLASS'] = params['PSUM_SPAD_CLASS']
        self.action_dict['PSUM_SPAD_ATRIBUTES.memory_depth'] = params['PSUM_SPAD_ATRIBUTES.memory_depth']
        self.action_dict['PSUM_SPAD_ATRIBUTES.block-size'] = params['PSUM_SPAD_ATRIBUTES.block-size']
        self.action_dict['PSUM_SPAD_ATRIBUTES.read_bandwidth'] = params['PSUM_SPAD_ATRIBUTES.read_bandwidth']
        self.action_dict['PSUM_SPAD_ATRIBUTES.write_bandwidth'] = params['PSUM_SPAD_ATRIBUTES.write_bandwidth']
        self.action_dict['WEIGHTS_SPAD_CLASS'] = params['WEIGHTS_SPAD_CLASS']
        self.action_dict['WEIGHTS_SPAD_ATRIBUTES.memory_depth'] = params['WEIGHTS_SPAD_ATRIBUTES.memory_depth']
        self.action_dict['WEIGHTS_SPAD_ATRIBUTES.block-size'] = params['WEIGHTS_SPAD_ATRIBUTES.block-size']
        self.action_dict['WEIGHTS_SPAD_ATRIBUTES.read_bandwidth'] = params['WEIGHTS_SPAD_ATRIBUTES.read_bandwidth']
        self.action_dict['WEIGHTS_SPAD_ATRIBUTES.write_bandwidth'] = params['WEIGHTS_SPAD_ATRIBUTES.write_bandwidth']
        self.action_dict['DUMMY_BUFFER_CLASS'] = params['DUMMY_BUFFER_CLASS']
        self.action_dict['DUMMY_BUFFER_ATTRIBUTES.depth'] = params['DUMMY_BUFFER_ATTRIBUTES.depth']
        self.action_dict['DUMMY_BUFFER_ATTRIBUTES.block-size'] = params['DUMMY_BUFFER_ATTRIBUTES.block-size']
        self.action_dict['SHARED_GLB_CLASS'] = params['SHARED_GLB_CLASS']
        self.action_dict['SHARED_GLB_ATTRIBUTES.memory_depth'] = params['SHARED_GLB_ATTRIBUTES.memory_depth']
        self.action_dict['SHARED_GLB_ATTRIBUTES.n_banks'] = params['SHARED_GLB_ATTRIBUTES.n_banks']
        self.action_dict['SHARED_GLB_ATTRIBUTES.block-size'] = params['SHARED_GLB_ATTRIBUTES.block-size']
        self.action_dict['SHARED_GLB_ATTRIBUTES.read_bandwidth'] = params['SHARED_GLB_ATTRIBUTES.read_bandwidth']
        self.action_dict['SHARED_GLB_ATTRIBUTES.write_bandwidth'] = params['SHARED_GLB_ATTRIBUTES.write_bandwidth']

        self.helper = helpers()
        
        # read from the config file
        config = configparser.ConfigParser()
        config.read("exp_config.ini")
        
        traj_dir = config.get("experiment_configuration", "trajectory_dir")
        exp_name = config.get("experiment_configuration", "exp_name")
        log_dir = config.get("experiment_configuration", "log_dir")
        target_area = config.get("experiment_configuration", "target_area")
        target_cycles = config.get("experiment_configuration", "target_cycles")
        target_energy = config.get("experiment_configuration", "target_energy")

        # read the all the parameters from exp_config.ini
        self.traj_dir = traj_dir
        self.exp_name = exp_name
        self.log = log
        self.reward_formulation = reward_formulation
        self.use_envlogger = use_envlogger
        self.output = output_dir
        self.arch = arch
        self.workload = workload
        self.target_area = float(target_area)
        self.target_cycles = float(target_cycles)
        self.target_energy = float(target_energy)
        self.script = script
        self.mapper = mapper_dir

        print("self.traj_dir", self.traj_dir)
        print("self.exp_name", self.exp_name)
        print("self.log", self.log)
        print("self.reward_formulation", self.reward_formulation)
        print("self.use_envlogger", self.use_envlogger)
        print("self.output", self.output)
        print("self.arch", self.arch)
        print("self.workload", self.workload)

        
        target_val = [self.target_energy, self.target_cycles, self.target_area]
        self.timeloop_env = TimeloopEnv(script_dir=self.script, output_dir=self.output, arch_dir=self.arch,
                                        mapper_dir=self.mapper, workload_dir=self.workload, target_val = target_val, reward_formulation=self.reward_formulation)

        self.bo_steps = 0

        # self.fitness_hist = []
        self.fitness_hist = {}
    

    def log_fitness_to_csv(self, exp_name, log_dir):
        df = pd.DataFrame([self.fitness_hist['fitness']])
        print(log_dir)
        print(exp_name)
        csvfile = log_dir + "/" + exp_name + "bo_fitness.csv"
        df.to_csv(csvfile, index=False, header=False, mode='a')

        df = pd.DataFrame([self.fitness_hist])
        csvfile = log_dir + "/" + exp_name + "bo_traj.csv"
        df.to_csv(csvfile, index=False, header=False, mode='a')


    def wrap_in_envlogger(self, env, envlogger_dir):
        metadata = {
            'agent_type': 'RandomWalker',
            'env_type': type(env).__name__,
        }
        if self.use_envlogger == 'True':
            logging.info('Wrapping environment with EnvironmentLogger...')
            env = envlogger.EnvLogger(env,
                                    data_directory=envlogger_dir,
                                    max_episodes_per_file=1000,
                                    metadata=metadata)
            logging.info('Done wrapping environment with EnvironmentLogger.')
            return env
        else:
            print("Not using envlogger")
            return env
    
    def _convert_flat_array_to_indexes(self, action):
        '''
        Converts the flat array of params to index values accepted by the env wrapper
        '''
        action_array = []
        allparams = self.timeloop_config.get_all_params_flattened()
        
        for key, value in action.items():
            # Add 1 since helper decode method uses 1-based indexing
            if(type(value) == str or type(value) == int):
                index = allparams[key].index(value)
                action_array.append(index)
            else:
                index = allparams[key].index(value[0])
                action_array.append(index)
            
        # get the index of the value in the list of all possible values for the parameter
            
        return action_array

    def fit(self, X, y=None):
        '''
        1) Call the Timeloop simulator and return energy, area, and cycles
        2) The parameter must be updated before the calling the Timeloop simulator
        3)  X is the trace files (.e., Workload)
        '''
        self.bo_steps += 1

        # read from the config file
        config = configparser.ConfigParser()
        config.read("exp_config.ini")

        # read the all the parameters from exp_config.ini
        traj_dir = config.get("experiment_configuration", "trajectory_dir")
        exp_name = config.get("experiment_configuration", "exp_name")
        log_dir = config.get("experiment_configuration", "log_dir")
        target_area = config.get("experiment_configuration", "target_area")
        target_cycles = config.get("experiment_configuration", "target_cycles")
        target_energy = config.get("experiment_configuration", "target_energy")

        env_wrapper = make_timeloop_env(env=self.timeloop_env)

        def step_fn(unused_timestep, unused_action, unused_env):
            return {'timestamp': time.time()}

        reward = 0
        self.fitness_hist = {}

        env = self.wrap_in_envlogger(env_wrapper, self.log)
        action_indexes = self._convert_flat_array_to_indexes(self.action_dict)
        env.reset()
        _, reward, _, info = env.step(action_indexes)

        
        self.fitness_hist['fitness'] = reward
        self.fitness_hist['action'] = self.action_dict
        self.fitness_hist['observation'] = info

        # check if trajectory directory exists
        if self.use_envlogger == 'True':
            if not os.path.exists(traj_dir):
                os.makedirs(traj_dir)
        # check if log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print("Logging fitness")
        
        if(self.bo_steps == 1):
            self.log_fitness_to_csv(exp_name, log_dir)
        
        return reward

    def predict(self, X, y):
        return NotImplementedError

    def score(self, X, y=None):
        return NotImplementedError

    def get_params(self, deep=False):
        return {'NUM_PEs': self.action_dict['NUM_PEs'],
                'MAC_MESH_X': self.action_dict['MAC_MESH_X'],
                'IFMAP_SPAD_CLASS': self.action_dict['IFMAP_SPAD_CLASS'],
                'IFMAP_SPAD_ATRIBUTES.memory_depth': self.action_dict['IFMAP_SPAD_ATRIBUTES.memory_depth'],
                'IFMAP_SPAD_ATRIBUTES.block-size': self.action_dict['IFMAP_SPAD_ATRIBUTES.block-size'],
                'IFMAP_SPAD_ATRIBUTES.read_bandwidth': self.action_dict['IFMAP_SPAD_ATRIBUTES.read_bandwidth'],
                'IFMAP_SPAD_ATRIBUTES.write_bandwidth': self.action_dict['IFMAP_SPAD_ATRIBUTES.write_bandwidth'],
                'PSUM_SPAD_CLASS': self.action_dict['PSUM_SPAD_CLASS'],
                'PSUM_SPAD_ATRIBUTES.memory_depth': self.action_dict['PSUM_SPAD_ATRIBUTES.memory_depth'],
                'PSUM_SPAD_ATRIBUTES.block-size': self.action_dict['PSUM_SPAD_ATRIBUTES.block-size'],
                'PSUM_SPAD_ATRIBUTES.read_bandwidth': self.action_dict['PSUM_SPAD_ATRIBUTES.read_bandwidth'],
                'PSUM_SPAD_ATRIBUTES.write_bandwidth': self.action_dict['PSUM_SPAD_ATRIBUTES.write_bandwidth'],
                'WEIGHTS_SPAD_CLASS': self.action_dict['WEIGHTS_SPAD_CLASS'],
                'WEIGHTS_SPAD_ATRIBUTES.memory_depth': self.action_dict['WEIGHTS_SPAD_ATRIBUTES.memory_depth'],
                'WEIGHTS_SPAD_ATRIBUTES.block-size': self.action_dict['WEIGHTS_SPAD_ATRIBUTES.block-size'],
                'WEIGHTS_SPAD_ATRIBUTES.read_bandwidth': self.action_dict['WEIGHTS_SPAD_ATRIBUTES.read_bandwidth'],
                'WEIGHTS_SPAD_ATRIBUTES.write_bandwidth': self.action_dict['WEIGHTS_SPAD_ATRIBUTES.write_bandwidth'],
                'DUMMY_BUFFER_CLASS': self.action_dict['DUMMY_BUFFER_CLASS'],
                'DUMMY_BUFFER_ATTRIBUTES.depth': self.action_dict['DUMMY_BUFFER_ATTRIBUTES.depth'],
                'DUMMY_BUFFER_ATTRIBUTES.block-size': self.action_dict['DUMMY_BUFFER_ATTRIBUTES.block-size'],
                'SHARED_GLB_CLASS': self.action_dict['SHARED_GLB_CLASS'],
                'SHARED_GLB_ATTRIBUTES.memory_depth': self.action_dict['SHARED_GLB_ATTRIBUTES.memory_depth'],
                'SHARED_GLB_ATTRIBUTES.n_banks': self.action_dict['SHARED_GLB_ATTRIBUTES.n_banks'],
                'SHARED_GLB_ATTRIBUTES.block-size': self.action_dict['SHARED_GLB_ATTRIBUTES.block-size'],
                'SHARED_GLB_ATTRIBUTES.read_bandwidth': self.action_dict['SHARED_GLB_ATTRIBUTES.read_bandwidth'],
                'SHARED_GLB_ATTRIBUTES.write_bandwidth': self.action_dict['SHARED_GLB_ATTRIBUTES.write_bandwidth'],
               }

    def set_params(self, **params):
        _params = params
        self.action_dict['NUM_PEs'] = _params['NUM_PEs']
        self.action_dict['MAC_MESH_X'] = _params['MAC_MESH_X']
        self.action_dict['IFMAP_SPAD_CLASS'] = _params['IFMAP_SPAD_CLASS'],
        self.action_dict['IFMAP_SPAD_ATRIBUTES.memory_depth'] = _params['IFMAP_SPAD_ATRIBUTES.memory_depth'],
        self.action_dict['IFMAP_SPAD_ATRIBUTES.block-size'] = _params['IFMAP_SPAD_ATRIBUTES.block-size'],
        self.action_dict['IFMAP_SPAD_ATRIBUTES.read_bandwidth'] = _params['IFMAP_SPAD_ATRIBUTES.read_bandwidth'],
        self.action_dict['IFMAP_SPAD_ATRIBUTES.write_bandwidth'] = _params['IFMAP_SPAD_ATRIBUTES.write_bandwidth'],
        self.action_dict['PSUM_SPAD_CLASS'] = _params['PSUM_SPAD_CLASS'],
        self.action_dict['PSUM_SPAD_ATRIBUTES.memory_depth'] = _params['PSUM_SPAD_ATRIBUTES.memory_depth'],
        self.action_dict['PSUM_SPAD_ATRIBUTES.block-size'] = _params['PSUM_SPAD_ATRIBUTES.block-size'],
        self.action_dict['PSUM_SPAD_ATRIBUTES.read_bandwidth'] = _params['PSUM_SPAD_ATRIBUTES.read_bandwidth'],
        self.action_dict['PSUM_SPAD_ATRIBUTES.write_bandwidth'] = _params['PSUM_SPAD_ATRIBUTES.write_bandwidth'],
        self.action_dict['WEIGHTS_SPAD_CLASS'] = _params['WEIGHTS_SPAD_CLASS'],
        self.action_dict['WEIGHTS_SPAD_ATRIBUTES.memory_depth'] = _params['WEIGHTS_SPAD_ATRIBUTES.memory_depth'],
        self.action_dict['WEIGHTS_SPAD_ATRIBUTES.block-size'] = _params['WEIGHTS_SPAD_ATRIBUTES.block-size'],
        self.action_dict['WEIGHTS_SPAD_ATRIBUTES.read_bandwidth'] = _params['WEIGHTS_SPAD_ATRIBUTES.read_bandwidth'],
        self.action_dict['WEIGHTS_SPAD_ATRIBUTES.write_bandwidth'] = _params['WEIGHTS_SPAD_ATRIBUTES.write_bandwidth'],
        self.action_dict['DUMMY_BUFFER_CLASS'] = _params['DUMMY_BUFFER_CLASS'],
        self.action_dict['DUMMY_BUFFER_ATTRIBUTES.depth'] = _params['DUMMY_BUFFER_ATTRIBUTES.depth'],
        self.action_dict['DUMMY_BUFFER_ATTRIBUTES.block-size'] = _params['DUMMY_BUFFER_ATTRIBUTES.block-size'],
        self.action_dict['SHARED_GLB_CLASS'] = _params['SHARED_GLB_CLASS'],
        self.action_dict['SHARED_GLB_ATTRIBUTES.memory_depth'] = _params['SHARED_GLB_ATTRIBUTES.memory_depth'],
        self.action_dict['SHARED_GLB_ATTRIBUTES.n_banks'] = _params['SHARED_GLB_ATTRIBUTES.n_banks'],
        self.action_dict['SHARED_GLB_ATTRIBUTES.block-size'] = _params['SHARED_GLB_ATTRIBUTES.block-size'],
        self.action_dict['SHARED_GLB_ATTRIBUTES.read_bandwidth'] = _params['SHARED_GLB_ATTRIBUTES.read_bandwidth'],
        self.action_dict['SHARED_GLB_ATTRIBUTES.write_bandwidth'] = _params['SHARED_GLB_ATTRIBUTES.write_bandwidth']

        return self

