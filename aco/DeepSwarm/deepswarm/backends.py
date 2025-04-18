# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

from arch_gym.envs import FARSI_sim_wrapper
from configs.sims import DRAMSys_config
from configs.sims import Timeloop_config
from configs.algos import rl_config
import copy
from . import cfg, nodes
import collections
import pandas as pd
import numpy as np
import random
from sims.Timeloop.process_params import TimeloopConfigParams
from arch_gym.envs.TimeloopEnv import TimeloopEnv
from arch_gym.envs.timeloop_acme_wrapper import make_timeloop_env
from arch_gym.envs.dramsys_wrapper import make_dramsys_env
from arch_gym.envs.maestero_wrapper import make_maestro_env
from arch_gym.envs.AstraSimWrapper import make_astraSim_env
from arch_gym.envs.SniperEnv import SniperEnv
from arch_gym.envs.DRAMEnv import DRAMEnv
from arch_gym.envs.AstraSimEnv import AstraSimEnv
from arch_gym.envs.envHelpers import helpers
import json
import envlogger
import os
import time
import yaml

from abc import ABC, abstractmethod

import sys
import os
from . import settings
from absl import logging
from absl import flags

os.sys.path.insert(0, os.path.abspath('/../../configs'))
print("NODES PREVIOUS: ", nodes)

def step_fn(unused_timestep, unused_action, unused_env):
    return {'timestamp': time.time()}


class Dataset:
    """Class responsible for encapsulating all the required data."""

    def __init__(self, training_examples, training_labels, testing_examples, testing_labels,
                 validation_data=None, validation_split=0.1):
        self.x_train = training_examples
        self.y_train = training_labels
        self.x_test = testing_examples
        self.y_test = testing_labels
        self.validation_data = validation_data
        self.validation_split = validation_split


class BaseBackend(ABC):
    """Abstract class used to define Backend API."""

    def __init__(self, dataset, optimizer=None):
        self.dataset = dataset
        self.optimizer = optimizer

    @abstractmethod
    def generate_model(self, path):
        """Create and return a backend model representation.

        Args:
            path [Node]: list of nodes where each node represents a single
            network layer, the path starts with InputNode and ends with EndNode.
        Returns:
            model which represents neural network structure in the implemented
            backend, this model can be evaluated using evaluate_model method.
        """

    @abstractmethod
    def reuse_model(self, old_model, new_model_path, distance):
        """Create a new model by reusing layers (and their weights) from the old model.

        Args:
            old_model: old model which represents neural network structure.
            new_model_path [Node]: path representing new model.
            distance (int): distance which shows how many layers from old model need
            to be removed in order to create a base for new model i.e. if old model is
            NodeA->NodeB->NodeC->NodeD and new model is NodeA->NodeB->NodeC->NodeE,
            distance = 1.
        Returns:
            model which represents neural network structure.
        """

    @abstractmethod
    def train_model(self, model):
        """Train model which was created using generate_model method.

        Args:
            model: model which represents neural network structure.
        Returns:
            model which represents neural network structure.
        """

    @abstractmethod
    def fully_train_model(self, model, epochs, augment):
        """Fully trains the model without early stopping. At the end of the
        training, the model with the best performing weights on the validation
        set is returned.

        Args:
            model: model which represents neural network structure.
            epochs (int): for how many epoch train the model.
            augment (kwargs): augmentation arguments.
        Returns:
            model which represents neural network structure.
        """

    @abstractmethod
    def evaluate_model(self, model):
        """Evaluate model which was created using generate_model method.

        Args:
            model: model which represents neural network structure.
        Returns:
            loss & accuracy tuple.
        """

    @abstractmethod
    def save_model(self, model, path):
        """Saves model on disk.

        Args:
            model: model which represents neural network structure.
            path: string which represents model location.
        """

    @abstractmethod
    def load_model(self, path):
        """Load model from disk, in case of fail should return None.

        Args:
            path: string which represents model location.
        Returns:
            model: model which represents neural network structure, or in case
            fail None.
        """

    @abstractmethod
    def free_gpu(self):
        """Frees GPU memory."""


class AstraSimBackend(BaseBackend):

    def __init__(self, dataset=None, optimizer=None, exp_name=None, traject_dir=None, log_dir=None, reward_formulation=None, 
                use_envlogger=False, VERSION=2, knobs_spec=None, network=None, system=None, workload=None, congestion_aware=True, 
                dimension=None, astrasim_ant_count=None, astrasim_greediness=None, astrasim_decay=None, 
                astrasim_evaporation=None, astrasim_start=None, astrasim_max_depth=None):
        super().__init__(dataset, optimizer)
        self.exp_name = exp_name
        self.traject_dir = traject_dir
        self.log_dir = log_dir
        self.reward_formulation = reward_formulation
        self.use_envlogger = use_envlogger
        self.VERSION = VERSION
        self.knobs_spec, self.network, self.system, self.workload = knobs_spec, network, system, workload
        self.congestion_aware = congestion_aware
        self.dimension = dimension
        self.astrasim_ant_count = astrasim_ant_count
        self.astrasim_greediness = astrasim_greediness
        self.astrasim_decay = astrasim_decay
        self.astrasim_evaporation = astrasim_evaporation
        self.astrasim_start = astrasim_start
        self.astrasim_max_depth = astrasim_max_depth

    def generate_model(self, path):
        return DummyAstraSim(path, self.exp_name, self.traject_dir, self.log_dir, self.reward_formulation, self.use_envlogger, 
                            self.VERSION, self.knobs_spec, self.network, self.system, self.workload, self.congestion_aware, 
                            self.dimension, self.astrasim_ant_count, self.astrasim_greediness, self.astrasim_decay, 
                            self.astrasim_evaporation, self.astrasim_start, self.astrasim_max_depth)

    def reuse_model(self, old_model, new_model_path, distance):
        return DummyAstraSim(new_model_path, self.exp_name, self.traject_dir, self.log_dir, self.reward_formulation, self.use_envlogger, 
                            self.VERSION, self.knobs_spec, self.network, self.system, self.workload, self.congestion_aware, 
                            self.dimension, self.astrasim_ant_count, self.astrasim_greediness, self.astrasim_decay, 
                            self.astrasim_evaporation, self.astrasim_start, self.astrasim_max_depth)

    def train_model(self, model):
        return model

    def fully_train_model(self, model, epochs, augment):
        return model

    def evaluate_model(self, model):
        value = model.fit(self.dataset.x_train)
        return (value, value)

    def save_model(self, model, path):
        return

    def load_model(self, path):
        return

    def free_gpu(self):
        return


class DRAMSysBackend(BaseBackend):
    """Backend based on DRAMSys API"""

    def __init__(self, dataset=None, optimizer=None, exp_name=None, traject_dir=None,
                 log_dir=None, reward_formulation=None, use_envlogger=False):
        super().__init__(dataset, optimizer)
        self.exp_name = exp_name
        self.traject_dir = traject_dir
        self.log_dir = log_dir
        self.reward_formulation = reward_formulation
        self.use_envlogger = use_envlogger

    def generate_model(self, path):
        return DummyDRAMSys(path, self.exp_name, self.traject_dir, self.log_dir, self.reward_formulation, self.use_envlogger)

    def reuse_model(self, old_model, new_model_path, distance):
        return DummyDRAMSys(new_model_path, self.exp_name, self.traject_dir, self.log_dir, self.reward_formulation, self.use_envlogger)

    def train_model(self, model):
        return model

    def fully_train_model(self, model, epochs, augment):
        return model

    def evaluate_model(self, model):
        value = model.fit(self.dataset.x_train)
        return (value, value)

    def save_model(self, model, path):
        return

    def load_model(self, path):
        return

    def free_gpu(self):
        return


class SniperBackend(BaseBackend):
    """Backend based on DRAMSys API"""

    def generate_model(self, path):
        return DummySniper(path)

    def generate_model_parallel(self, path):
        models = []
        for each_path in path:
            models.append(DummySniper(each_path))
        return models

    def reuse_model(self, old_model, new_model_path, distance):
        return DummySniper(new_model_path)

    def train_model(self, model):
        return model

    def fully_train_model(self, model, epochs, augment):
        return model

    def evaluate_model(self, model):
        value = model.fit(self.dataset.x_train)
        return (value, value)

    def evaluate_model_parallel(self, model):
        sph = Sniper_Parallel_Helper(model)
        value = sph.fit_parallel(model)

        return (value, value)

    def save_model(self, model, path):
        return

    def load_model(self, path):
        return

    def free_gpu(self):
        return


class Sniper_Parallel_Helper():
    def __init__(self, model):
        self.env = SniperEnv()
        self.helper = helpers()

    def fit_parallel(self, model):

        action_dicts = collections.defaultdict(dict)
        for id in range(len(model)):
            agent_id = "agent_" + str(id)
            action_dicts[agent_id]['core_dispatch_width'] = model[id].action_dict["core_dispatch_width"]
            action_dicts[agent_id]['core_window_size'] = model[id].action_dict["core_window_size"]
            action_dicts[agent_id]['core_outstanding_loads'] = model[id].action_dict["core_outstanding_loads"]
            action_dicts[agent_id]['core_outstanding_stores'] = model[id].action_dict["core_outstanding_stores"]
            action_dicts[agent_id]['core_commit_width'] = model[id].action_dict["core_commit_width"]
            action_dicts[agent_id]['core_rs_entries'] = model[id].action_dict["core_rs_entries"]
            action_dicts[agent_id]['l1_icache_size'] = model[id].action_dict["l1_icache_size"]
            action_dicts[agent_id]['l1_dcache_size'] = model[id].action_dict["l1_dcache_size"]
            action_dicts[agent_id]['l2_cache_size'] = model[id].action_dict["l2_cache_size"]
            action_dicts[agent_id]['l3_cache_size'] = model[id].action_dict["l3_cache_size"]

        # Always reset before step
        self.env.reset_multiagent()
        obs, reward, _, _ = self.env.step_multiagent(action_dicts)

        # lowers the runtime
        #runtimes = [-1*x[0] for x in obs]
        fitness = [-1 for each_reward in reward]
        self.env.reset_multiagent()
        return fitness


class DummySniper():
    def __init__(self, path):
        self.env = SniperEnv()
        self.helper = helpers()
        self.fitness_hist = []
        self.action_dict = {}
        self.action_dict["core_dispatch_width"] = 2
        self.action_dict["core_window_size"] = 2
        self.action_dict["core_outstanding_loads"] = 2
        self.action_dict["core_outstanding_stores"] = 2
        self.action_dict["core_commit_width"] = 2
        self.action_dict["core_rs_entries"] = 2
        self.action_dict["l1_icache_size"] = 4
        self.action_dict["l1_dcache_size"] = 4
        self.action_dict["l2_cache_size"] = 128
        self.action_dict["l3_cache_size"] = 512

        for node in path:
            if hasattr(node, "CoreDispatchWidth"):
                self.action_dict["core_dispatch_width"] = node.CoreDispatchWidth
                self.action_dict["core_window_size"] = node.CoreWindowSize
                self.action_dict["core_outstanding_loads"] = node.CoreOutstandingLoads
                self.action_dict["core_outstanding_stores"] = node.CoreOutstandingStores
                self.action_dict["core_commit_width"] = node.CoreCommitWidth
                self.action_dict["core_rs_entries"] = node.CoreRSEntries
                self.action_dict["l1_icache_size"] = node.L1_icache_size
                self.action_dict["l1_dcache_size"] = node.L1_dcache_size
                self.action_dict["l2_cache_size"] = node.L2_cache_size
                self.action_dict["l3_cache_size"] = node.L3_cache_size

    def fit(self, x):
        # Always reset before step
        self.env.reset()
        #obs,reward,_,_ = self.env.step(self.action_dict)

        fitness = -1  # *self.calculate_reward(obs)

        # Todo: log fitness history for plotting

        return fitness

    def calculate_reward(self, obs):
        # Todo: Change this depending upon the optimization goal
        # for now, just return the runtime

        return obs[0]


class DummyAstraSim():

    def __init__(self, path, exp_name, traject_dir, log_dir, reward_formulation, use_envlogger, VERSION, 
                    knobs_spec, network, system, workload, congestion_aware, dimension, astrasim_ant_count, 
                    astrasim_greediness, astrasim_decay, astrasim_evaporation, astrasim_start, astrasim_max_depth):
        self.helper = helpers()
        self.fitness_hist = {}

        self.traject_dir = traject_dir
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.reward_formulation = reward_formulation
        self.use_envlogger = use_envlogger
        self.VERSION = VERSION
        self.knobs_spec, self.network, self.system, self.workload = knobs_spec, network, system, workload
        self.congestion_aware = congestion_aware
        self.dimension = dimension
        self.astrasim_ant_count = astrasim_ant_count
        self.astrasim_greediness = astrasim_greediness
        self.astrasim_decay = astrasim_decay
        self.astrasim_evaporation = astrasim_evaporation
        self.astrasim_start = astrasim_start
        self.astrasim_max_depth = astrasim_max_depth

        self.settings_file_path = os.path.realpath(__file__)
        self.settings_dir_path = os.path.dirname(self.settings_file_path)
        self.proj_root_path = os.path.dirname(os.path.dirname(os.path.dirname(self.settings_dir_path)))
        self.proj_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(self.settings_dir_path)))
        self.astrasim = os.path.join(self.proj_dir_path, "sims/AstraSim")

        self.astrasim_archgym = os.path.join(self.astrasim, "astrasim-archgym")
        self.knobs = os.path.join(self.astrasim, self.knobs_spec)

        self.env = AstraSimEnv(knobs_spec=self.knobs, network=network, system=system, workload=workload)

        self.systems_folder = os.path.join(
            self.astrasim_archgym, "themis/inputs/system")
        self.network_folder = os.path.join(
            self.astrasim_archgym, "dse/archgen_v1_knobs/templates/network")
        self.workload_folder = os.path.join(
            self.astrasim_archgym, "themis/inputs/workload")

        # parse knobs
        system_knob, network_knob, workload_knob = self.helper.parse_knobs_astrasim(self.knobs)
        if workload_knob == {}:
            GENERATE_WORKLOAD = "FALSE"
        else:
            GENERATE_WORKLOAD = "TRUE"

        if self.VERSION == 1:
            self.system_file = os.path.join(
                self.systems_folder, "4d_ring_fc_ring_switch_baseline.txt")
            self.network_file = os.path.join(
                self.network_folder, "4d_ring_fc_ring_switch.json")
            self.workload_file = os.path.join(
                self.workload_folder, "all_reduce/allreduce_0.65.txt"
            )
        else:
            self.network_file = os.path.join(self.astrasim, self.network)
            self.system_file = os.path.join(self.astrasim, self.system)
            self.workload_file = os.path.join(self.astrasim, self.workload)

        # SET UP ACTION DICT
        self.action_dict = {}
        # only generate workload if knobs exist
        # if GENERATE_WORKLOAD == "TRUE":
        #     self.action_dict['workload'] = self.helper.parse_workload_astrasim(self.workload_file, self.action_dict, self.VERSION)
        # else:
        #     self.action_dict['workload'] = {"path": self.workload_file}
        self.action_dict['workload'] = {}

        # parse system and network files
        self.helper.parse_system_astrasim(self.system_file, self.action_dict, self.VERSION)
        self.helper.parse_network_astrasim(self.network_file, self.action_dict, self.VERSION)

        dicts = [(system_knob, 'system'), (network_knob, 'network'), (workload_knob, 'workload')]
        yaml_path = os.path.join(self.proj_root_path, 'settings/default_astrasim.yaml')
        print('BACKEND DICTS: ', dicts)
        print("YAML PATH: ", yaml_path)
        # set to everything in yaml file's default (last iteration)
        data = yaml.load(open(yaml_path), Loader=yaml.Loader)

        print("DIMENSION1: ", self.dimension)
        dimension = self.dimension

        # write knobs to yaml file
        # Rewrite the flags
        data['DeepSwarm']['aco']["ant_count"] = self.astrasim_ant_count
        data['DeepSwarm']['aco']["greediness"] = self.astrasim_greediness
        data['DeepSwarm']['aco']["pheromone"]["decay"] = self.astrasim_decay
        data['DeepSwarm']['aco']["pheromone"]["evaporation"] = self.astrasim_evaporation
        data['DeepSwarm']['aco']["pheromone"]["start"] = self.astrasim_start
        data['DeepSwarm']["max_depth"] = self.astrasim_max_depth

        # Rewrite the attributes
        data['Nodes']['ArchParamsNode']['attributes'] = {}
        for dict_type, dict_name in dicts:
            for knob in dict_type:
                if knob == "dimensions-count":
                    continue
                # from hyphen to CamelCase
                knob_converted = self.helper.convert_knob_ga_astrasim(knob)
                if isinstance(dict_type[knob][0], set):
                    if dict_type[knob][1] == "FALSE":
                        for i in range(1, dimension + 1):
                            knob_dimension = knob_converted + str(i)
                            list_sorted = sorted(list(dict_type[knob][0]))
                            data['Nodes']['ArchParamsNode']['attributes'][knob_dimension] = [i for i in list_sorted]
                    else:
                        list_sorted = sorted(list(dict_type[knob][0]))
                        data['Nodes']['ArchParamsNode']['attributes'][knob_converted] = [i for i in list_sorted]
                else:
                    if dict_type[knob][1] == "FALSE":
                        for i in range(1, dimension + 1):
                            knob_dimension = knob_converted + str(i)
                            data['Nodes']['ArchParamsNode']['attributes'][knob_dimension] = [i for i in range(dict_type[knob][0][0], dict_type[knob][0][1] + 1)]
                    else:
                        data['Nodes']['ArchParamsNode']['attributes'][knob_converted] = [i for i in range(dict_type[knob][0][0], dict_type[knob][0][1] + 1)]

        print("YAML DATA: ", data)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
            f.flush()
        

        if len(system_knob.keys()) != 0:
            rand_attr = sorted(list(system_knob.keys()))[0]
        elif len(network_knob.keys()) != 0:
            rand_attr = sorted(list(network_knob.keys()))[0]
            if rand_attr == "dimensions-count":
                rand_attr = sorted(list(network_knob.keys()))[1]
        else:
            rand_attr = sorted(list(workload_knob.keys()))[0]
        rand_attr = self.helper.convert_knob_ga_astrasim(rand_attr)
        
        # writes action dictionary of tunable knobs
        # unexposed knobs are filled in inside AstraSimEnv
        # TODO: put knobs back to the action_dict
        print("reached node in path: ", path)
        # three knotes of Nodes: ArchParamsNode, InputNode, OutputNode
        for node in path:
            if hasattr(node, rand_attr) or hasattr(node, rand_attr + "1"):
                print("node has attr: ", rand_attr)
                for attr, value in node.__dict__.items():
                    print("node dict: ", node.__dict__)
                    print("node :", node)
                    # converts from camelCase to hyphen
                    knob_converted = self.helper.revert_knob_ga_astrasim(attr)
                    if attr.lower() in workload_knob:
                        knob_converted = attr.lower()
                    for dict_type, dict_name in dicts:
                        # print("DICT TYPE: ", dict_type)
                        # print("DICT NAME: ", dict_name)
                        # print("KNOB NOT CONVERTED: ", attr)
                        # print("KNOB CONVERTED: ", knob_converted)
                        # if tunable knob found in current dict_type
                        if knob_converted in dict_type:
                            # False means it changes with each dimension
                            if dict_type[knob_converted][1] == "FALSE":
                                if attr[-1] == "1":
                                    self.action_dict[dict_name][knob_converted] = [value]
                                else:
                                    if isinstance(self.action_dict[dict_name][knob_converted], str):
                                        self.action_dict[dict_name][knob_converted] += value
                                    else:
                                        self.action_dict[dict_name][knob_converted].append(value)
                            elif dict_type[knob_converted][1] == "TRUE":
                                self.action_dict[dict_name][knob_converted] = [value for _ in range(dimension)]
                            else:
                                self.action_dict[dict_name][knob_converted] = value
        
        print("BACKEND FINAL ACTION DICT: ", self.action_dict)
        print("BACKEND FINAL ACTION DICT WORKLOADS: ", self.action_dict['workload'])

        if "dimensions-count" in network_knob:
            print("Dimension count in network knob")
            self.action_dict["network"]["dimensions-count"] = dimension
    


    # Fit function = step function
    # Environment already calculates reward so don't need calc_reward

    def wrap_in_envlogger(self, env, envlogger_dir, use_envlogger):
        metadata = {
            'agent_type': 'ACO',
            'env_type': type(env).__name__,
        }
        if use_envlogger == True:
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

    def fit(self, X, y=None):
        '''
        This is the function that is called by the optimizer. ACO by defaul tries to minimize the fitness function.
        If you have a fitness function that you want to maximize, you can simply return the negative of the fitness function.
        '''

        env_wrapper = make_astraSim_env(
            knobs_spec=self.knobs, network=self.network_file, system=self.system_file, workload=self.workload_file, 
            reward_formulation=self.reward_formulation, rl_form="aco", congestion_aware=self.congestion_aware, max_steps=self.astrasim_max_depth)

        if self.use_envlogger:
            # check if trajectory directory exists
            if not os.path.exists(self.traject_dir):
                os.makedirs(self.traject_dir)

        # check if log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        env = self.wrap_in_envlogger(env_wrapper, self.traject_dir, self.use_envlogger)
        env.reset()

        # HERE is where ACO takes a step for Astra-Sim
        print("BACKENDS STEP ACTION_DICT: ", self.action_dict)
        step_type, reward, discount, info = env.step(self.action_dict)

        self.fitness_hist['reward'] = reward
        self.fitness_hist['action_dict'] = self.action_dict
        self.fitness_hist['obs'] = info

        print("Reward: ", reward)
        print("Action Dict: ", self.action_dict)
        print("Info: ", info)

        self.log_fitness_to_csv()
        return -1 * reward

    def log_fitness_to_csv(self):
        df_traj = pd.DataFrame([self.fitness_hist])
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
        df_traj.insert(0, 'timestamp', timestamp)

        filename = os.path.join(self.log_dir, self.exp_name + "_traj.csv")
        df_traj.to_csv(filename, index=False, header=False, mode='a')

        df_rewards = pd.DataFrame([self.fitness_hist['reward']])
        filename = os.path.join(self.log_dir, self.exp_name + "_rewards.csv")
        df_rewards.to_csv(filename,
                          index=False, header=False, mode='a')


class DummyDRAMSys():
    """Dummy placeholder for DRAMSys to do POC"""

    def __init__(self, path, exp_name, traject_dir, log_dir, reward_formulation, use_envlogger):
        ''' 
        All the default values of the "model" should be initialized here. 
        '''
        # NOTE: Whatever default is, influences pheremone trail before full depth
        # is reached if choose to have each node be arch param bc it iteratively
        # builds up network
        self.env = DRAMEnv()
        self.fitness_hist = {}
        self.action_dict = {}

        self.PagePolicy = 'Open'
        self.Scheduler = "Fifo"
        self.SchedulerBuffer = "Bankwise"
        self.RequestBufferSize = 1
        self.RespQueue = "Fifo"
        self.RefreshPolicy = "NoRefresh"
        self.RefreshMaxPostponed = 1
        self.RefreshMaxPulledin = 1
        self.Arbiter = "Simple"
        self.MaxActiveTransactions = 16

        self.traject_dir = traject_dir
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.reward_formulation = reward_formulation
        self.use_envlogger = use_envlogger

        # Single layer network implementation
        for node in path:
            if hasattr(node, "PagePolicy"):
                self.action_dict["PagePolicy"] = node.PagePolicy
                self.action_dict["Scheduler"] = node.Scheduler
                self.action_dict["SchedulerBuffer"] = node.SchedulerBuffer
                self.action_dict["RequestBufferSize"] = node.RequestBufferSize
                self.action_dict["RespQueue"] = node.RespQueue
                self.action_dict["RefreshPolicy"] = node.RefreshPolicy
                self.action_dict["RefreshMaxPostponed"] = node.RefreshMaxPostponed
                self.action_dict["RefreshMaxPulledin"] = node.RefreshMaxPulledin
                self.action_dict["Arbiter"] = node.Arbiter
                self.action_dict["MaxActiveTransactions"] = node.MaxActiveTransactions

    def wrap_in_envlogger(self, env, envlogger_dir, use_envlogger):
        metadata = {
            'agent_type': 'RandomWalker',
            'env_type': type(env).__name__,
        }
        if use_envlogger == True:
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

    def fit(self, X, y=None):
        '''
        This is the function that is called by the optimizer. ACO by defaul tries to minimize the fitness function.
        If you have a fitness function that you want to maximize, you can simply return the negative of the fitness function.
        '''

        env_wrapper = make_dramsys_env(
            reward_formulation=self.reward_formulation)

        if self.use_envlogger:
            # check if trajectory directory exists
            if not os.path.exists(self.traject_dir):
                os.makedirs(self.traject_dir)

        # check if log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        env = self.wrap_in_envlogger(
            env_wrapper, self.traject_dir, self.use_envlogger)
        env.reset()
        _, reward, _, info = env.step(self.action_dict)

        self.fitness_hist['reward'] = reward
        self.fitness_hist['action_dict'] = self.action_dict
        self.fitness_hist['obs'] = info

        print("Reward: ", reward)
        print("Action Dict: ", self.action_dict)
        print("Info: ", info)

        self.log_fitness_to_csv()
        return -1 * reward

    def calculate_reward(self, energy=None, power=None, latency=None):
        if power is not None and latency is not None:
            print(
                "*********************OPTIMIZE FOR POWER & LATENCY*********************")
            reward = np.sum([np.square(DRAMSys_config.target_power - power),
                            np.square(DRAMSys_config.target_latency - latency)])
            reward = np.sqrt(reward)
        elif power is not None:
            print("*********************OPTIMIZE FOR POWER*********************")
            reward = np.sum([np.square(DRAMSys_config.target_power - power)])
            reward = np.sqrt(reward)
        elif latency is not None:
            print("*********************OPTIMIZE FOR LATENCY*********************")
            reward = np.sum(
                [np.square(DRAMSys_config.target_latency - latency)])
            reward = np.sqrt(reward)
        return reward

    def read_modify_write_simconfigs(self):
        mem_ctrl_filename = "policy.json"
        op_success = False
        full_path = os.path.join(
            DRAMSys_config.dram_mem_controller_config, mem_ctrl_filename)

        try:
            with open(full_path, "r") as JsonFile:
                data = json.load(JsonFile)
                data['mcconfig']['PagePolicy'] = self.PagePolicy
                data['mcconfig']['Scheduler'] = self.Scheduler
                data['mcconfig']['SchedulerBuffer'] = self.SchedulerBuffer
                data['mcconfig']['RequestBufferSize'] = self.RequestBufferSize
                data['mcconfig']['RespQueue'] = self.RespQueue
                data['mcconfig']['RefreshPolicy'] = self.RefreshPolicy
                data['mcconfig']['RefreshMaxPostponed'] = self.RefreshMaxPostponed
                data['mcconfig']['RefreshMaxPulledin'] = self.RefreshMaxPulledin
                data['mcconfig']['Arbiter'] = self.Arbiter
                data['mcconfig']['MaxActiveTransactions'] = self.MaxActiveTransactions

                with open(full_path, "w") as JsonFile:
                    json.dump(data, JsonFile)
                op_success = True
        except Exception as e:
            print(str(e))
            op_success = False
        return op_success

    def log_fitness_to_csv(self):
        df_traj = pd.DataFrame([self.fitness_hist])
        filename = os.path.join(self.log_dir, self.exp_name + "_traj.csv")
        df_traj.to_csv(filename,
                       index=False, header=False, mode='a')

        df_rewards = pd.DataFrame([self.fitness_hist['reward']])
        filename = os.path.join(self.log_dir, self.exp_name + "_rewards.csv")
        df_rewards.to_csv(filename,
                          index=False, header=False, mode='a')


class MaestroBackend(BaseBackend):
    """Backend based on Maestro API"""

    def __init__(self, dataset=None, optimizer=None, exp_name=None, traject_dir=None,
                 log_dir=None, reward_formulation=None, use_envlogger=False):
        super().__init__(dataset, optimizer)
        self.exp_name = exp_name
        self.traject_dir = traject_dir
        self.log_dir = log_dir
        self.reward_formulation = reward_formulation
        self.use_envlogger = use_envlogger

    def generate_model(self, path):
        return DummyMaestro(path, self.exp_name, self.traject_dir, self.log_dir, self.reward_formulation, self.use_envlogger)

    def reuse_model(self, old_model, new_model_path, distance):
        return DummyMaestro(new_model_path, self.exp_name, self.traject_dir, self.log_dir, self.reward_formulation, self.use_envlogger)

    def train_model(self, model):
        return model

    def fully_train_model(self, model, epochs, augment):
        return model

    def evaluate_model(self, model):
        value = model.fit(self.dataset.x_train)
        return (value, value)

    def save_model(self, model, path):
        return

    def load_model(self, path):
        return

    def free_gpu(self):
        return


class DummyMaestro():
    """Dummy placeholder for Maestro to do POC"""

    def __init__(self, path, exp_name, traject_dir, log_dir, reward_formulation, use_envlogger):
        ''' 
        All the default values of the "model" should be initialized here. 
        '''
        # NOTE: Whatever default is, influences pheremone trail before full depth
        # is reached if choose to have each node be arch param bc it iteratively
        # builds up network
        self.fitness_hist = {}
        self.action_dict = {}

        self.action_dict['seed_l2'] = 1234
        self.action_dict['ckxy_l2'] = 1
        self.action_dict['s_l2'] = 2
        self.action_dict['r_l2'] = 2
        self.action_dict['k_l2'] = 1
        self.action_dict['c_l2'] = 1
        self.action_dict['x_l2'] = 1
        self.action_dict['y_l2'] = 1
        self.action_dict['ckxy_l1'] = 2
        self.action_dict['s_l1'] = 2
        self.action_dict['r_l1'] = 2
        self.action_dict['k_l1'] = 1
        self.action_dict['c_l1'] = 2
        self.action_dict['x_l1'] = 1
        self.action_dict['y_l1'] = 1
        self.action_dict['seed_l1'] = 3211
        self.action_dict['num_pe'] = 10

        self.traject_dir = traject_dir
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.reward_formulation = reward_formulation
        self.use_envlogger = use_envlogger

        # Single layer network implementation
        for node in path:
            if hasattr(node, "seed_l2"):
                self.action_dict['seed_l2'] = node.seed_l2
                self.action_dict['ckxy_l2'] = node.ckxy_l2
                self.action_dict['s_l2'] = node.s_l2
                self.action_dict['r_l2'] = node.r_l2
                self.action_dict['k_l2'] = node.k_l2
                self.action_dict['c_l2'] = node.c_l2
                self.action_dict['x_l2'] = node.x_l2
                self.action_dict['y_l2'] = node.y_l2
                self.action_dict['seed_l1'] = node.seed_l1
                self.action_dict['ckxy_l1'] = node.ckxy_l1
                self.action_dict['s_l1'] = node.s_l1
                self.action_dict['r_l1'] = node.r_l1
                self.action_dict['k_l1'] = node.k_l1
                self.action_dict['c_l1'] = node.c_l1
                self.action_dict['x_l1'] = node.x_l1
                self.action_dict['y_l1'] = node.y_l1
                self.action_dict['num_pe'] = node.num_pe

    def wrap_in_envlogger(self, env, envlogger_dir, use_envlogger):
        metadata = {
            'agent_type': 'ACO',
            'env_type': type(env).__name__,
        }
        if use_envlogger == True:
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

    def convert_action_dict_to_action(self, action_dict):
        order = [
            'seed_l2', 'ckxy_l2', 's_l2', 'r_l2', 'k_l2',
            'c_l2', 'x_l2', 'y_l2', 'ckxy_l1', 's_l1',
            'r_l1', 'k_l1', 'c_l1', 'x_l1', 'y_l1',
            'seed_l1', 'num_pe'
        ]

        action_list = [action_dict[key] for key in order]
        print(action_list)

        return action_list

    def fit(self, X, y=None):
        '''
        This is the function that is called by the optimizer. ACO by defaul tries to minimize the fitness function.
        If you have a fitness function that you want to maximize, you can simply return the negative of the fitness function.
        '''

        env_wrapper = make_maestro_env(
            reward_formulation=self.reward_formulation, rl_form="aco")

        if self.use_envlogger:
            # check if trajectory directory exists
            if not os.path.exists(self.traject_dir):
                os.makedirs(self.traject_dir)

        # check if log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        env = self.wrap_in_envlogger(
            env_wrapper, self.traject_dir, self.use_envlogger)
        env.reset()

        # convert action dict to action list
        action_list = self.convert_action_dict_to_action(self.action_dict)

        _, reward, _, info = env.step(action_list)

        self.fitness_hist['reward'] = reward
        self.fitness_hist['action_dict'] = self.action_dict
        self.fitness_hist['obs'] = info

        print("Reward: ", reward)
        print("Action Dict: ", self.action_dict)
        print("Info: ", info)

        self.log_fitness_to_csv()
        return -1 * reward

    def log_fitness_to_csv(self):
        df_traj = pd.DataFrame([self.fitness_hist])
        filename = os.path.join(self.log_dir, self.exp_name + "_traj.csv")
        df_traj.to_csv(filename,
                       index=False, header=False, mode='a')

        df_rewards = pd.DataFrame([self.fitness_hist['reward']])
        filename = os.path.join(self.log_dir, self.exp_name + "_rewards.csv")
        df_rewards.to_csv(filename,
                          index=False, header=False, mode='a')


class TimeloopBackend(BaseBackend):
    """Timeloop Backend for ACO"""

    def __init__(self, dataset, optimizer=None, script_dir=None, output_dir=None,
                 arch_dir=None, mapper_dir=None, workload_dir=None, target_val=None,
                 log_dir=None, use_envlogger=None, reward_formulation=None, exp_name=None):
        super().__init__(dataset, optimizer)
        self.script_dir = script_dir
        self.output_dir = output_dir
        self.arch_dir = arch_dir
        self.mapper_dir = mapper_dir
        self.workload_dir = workload_dir
        self.target_val = target_val
        self.log_dir = log_dir
        self.use_envlogger = use_envlogger
        self.reward_formulation = reward_formulation
        self.exp_name = exp_name

    def generate_model(self, path):
        return DummyTimeloop(path, self.script_dir, self.output_dir, self.arch_dir,
                             self.mapper_dir, self.workload_dir, self.target_val, self.log_dir,
                             self.use_envlogger, self.reward_formulation, self.exp_name)

    def generate_model_parallel(self, path):
        models = []
        for each_path in path:
            models.append(DummyTimeloop(each_path, self.script_dir, self.output_dir,
                          self.arch_dir, self.mapper_dir, self.workload_dir, self.target_val, self.log_dir,
                          self.use_envlogger, self.reward_formulation, self.exp_name))
        return models

    def reuse_model(self, old_model, new_model_path, distance):
        return DummyTimeloop(new_model_path, self.script_dir, self.output_dir, self.arch_dir,
                             self.mapper_dir, self.workload_dir, self.target_val, self.log_dir,
                             self.use_envlogger, self.reward_formulation, self.exp_name)

    def train_model(self, model):
        return model

    def fully_train_model(self, model, epochs, augment):
        return model

    def evaluate_model(self, model):
        value = model.fit(self.dataset.x_train)
        return (value, value)

    def evaluate_model_parallel(self, model):
        sph = Timeloop_Parallel_Helper(self.script_dir, self.output_dir, self.arch_dir,
                                       self.mapper_dir, self.workload_dir, self.target_val, self.log_dir)
        value = sph.fit_parallel(model)

        return (value, value)

    def save_model(self, model, path):
        return

    def load_model(self, path):
        return

    def free_gpu(self):
        return


class DummyTimeloop():
    def __init__(self, path, script_dir, output_dir, arch_dir, mapper_dir, workload_dir, target_val, log_dir,
                 use_envlogger, reward_formulation, exp_name):

        self.log_dir = log_dir
        self.timeloop_config = TimeloopConfigParams(
            Timeloop_config.timeloop_parameters)
        self.fitness_hist = {}
        self.action_dict = []
        self.flat_params = self.timeloop_config.get_all_params_flattened()

        self.script_dir = script_dir
        self.output_dir = output_dir
        self.arch_dir = arch_dir
        self.mapper_dir = mapper_dir
        self.workload_dir = workload_dir
        self.target_val = target_val
        self.reward_formulation = reward_formulation
        self.use_envlogger = use_envlogger
        self.helper = helpers()
        self.exp_name = exp_name

        self.timeloop_env = TimeloopEnv(script_dir=self.script_dir,
                                        output_dir=self.output_dir,
                                        arch_dir=self.arch_dir,
                                        mapper_dir=self.mapper_dir,
                                        workload_dir=self.workload_dir,
                                        target_val=self.target_val,
                                        reward_formulation=self.reward_formulation)

        for node in path:
            if hasattr(node, "NUM_PEs"):
                self._set_action_dict(node)

    def _set_action_dict(self, node):
        for field, values in self.flat_params.items():
            if '.' in field:
                self.action_dict.append(values.index(
                    getattr(node, '_'.join(field.split('.')))) + 1)
            else:
                self.action_dict.append(values.index(getattr(node, field)) + 1)

        self.action_dict = np.array(self.action_dict)

    def wrap_in_envlogger(self, env, envlogger_dir, use_envlogger):
        metadata = {
            'agent_type': 'ACO',
            'env_type': type(env).__name__,
        }
        if use_envlogger == True:
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

    def fit(self, x):
        # Always reset before step

        env_wrapper = make_timeloop_env(env=TimeloopEnv(
            script_dir=self.script_dir,
            output_dir=self.output_dir,
            arch_dir=self.arch_dir,
            mapper_dir=self.mapper_dir,
            workload_dir=self.workload_dir,
            target_val=self.target_val,
            reward_formulation=self.reward_formulation)
        )

        env = self.wrap_in_envlogger(
            env_wrapper, self.log_dir, self.use_envlogger)
        env.reset()

        action_dict = self.helper.decode_timeloop_action(self.action_dict)
        _, reward, _, info = env.step(self.action_dict)

        self.fitness_hist['reward'] = reward
        self.fitness_hist['action_dict'] = action_dict
        self.fitness_hist['obs'] = info

        self.log_fitness_to_csv()

        return reward

    def calculate_reward(self, obs):
        # Todo: Change this depending upon the optimization goal
        # for now, just return the runtime

        return obs[0]

    def log_fitness_to_csv(self):
        df_traj = pd.DataFrame([self.fitness_hist])
        filename = os.path.join(self.log_dir, self.exp_name + "_traj.csv")
        df_traj.to_csv(filename,
                       index=False, header=False, mode='a')

        df_rewards = pd.DataFrame([self.fitness_hist['reward']])
        filename = os.path.join(self.log_dir, self.exp_name + "_rewards.csv")
        df_rewards.to_csv(filename,
                          index=False, header=False, mode='a')


class Timeloop_Parallel_Helper():
    def __init__(self, script_dir, output_dir, arch_dir, mapper_dir, workload_dir, target_val, log_dir):
        self.timeloop_env = TimeloopEnv(script_dir, output_dir,
                                        arch_dir, mapper_dir, workload_dir, target_val)

        self.env_wrapper = make_timeloop_env(
            env=self.timeloop_env, multi_agent=True)
        self.log_dir = log_dir

    def fit_parallel(self, model):

        action_dicts = []
        for id in range(len(model)):
            # agent_id = "agent_" + str(id)
            action_dicts.append(copy.deepcopy(model[id].action_dict))

        action_dicts = np.array(action_dicts)

        env_wrapper = make_timeloop_env(
            env=self.timeloop_env, multi_agent=True)

        with envlogger.EnvLogger(env_wrapper,
                                 data_directory=self.log_dir,
                                 max_episodes_per_file=1000,
                                 metadata={
                                     'agent_type': 'aco',
                                     'env_type': type(env_wrapper).__name__
                                 }, step_fn=step_fn) as env:

            # Always reset before step
            env.reset()
            _, reward, _, _ = env.step(action_dicts)
            env.reset()

            return reward


class FARSIBackend(BaseBackend):
    """Backend based on FARSI API"""

    def __init__(self, dataset=None, optimizer=None, exp_name=None, traject_dir=None,
                 log_dir=None, reward_formulation=None, workload=None, use_envlogger=False):
        super().__init__(dataset, optimizer)
        self.exp_name = exp_name
        self.traject_dir = traject_dir
        self.log_dir = log_dir
        self.reward_formulation = reward_formulation
        self.workload = workload
        self.use_envlogger = use_envlogger

    def generate_model(self, path):
        return DummyFARSI(path, self.exp_name, self.traject_dir, self.log_dir, self.reward_formulation, self.workload, self.use_envlogger)

    def reuse_model(self, old_model, new_model_path, distance):
        return DummyFARSI(new_model_path, self.exp_name, self.traject_dir, self.log_dir, self.reward_formulation, self.workload, self.use_envlogger)

    def train_model(self, model):
        return model

    def fully_train_model(self, model, epochs, augment):
        return model

    def evaluate_model(self, model):
        value = model.fit(self.dataset.x_train)
        return (value, value)

    def save_model(self, model, path):
        return

    def load_model(self, path):
        return

    def free_gpu(self):
        return


class DummyFARSI():
    """Dummy placeholder for FARSI to do POC"""

    def __init__(self, path, exp_name, traject_dir, log_dir, reward_formulation, workload, use_envlogger):
        ''' 
        All the default values of the "model" should be initialized here. 
        '''
        # NOTE: Whatever default is, influences pheremone trail before full depth
        # is reached if choose to have each node be arch param bc it iteratively
        # builds up network
        self.fitness_hist = {}
        self.action_dict = {}

        self.pe_allocation_0 = 0
        self.pe_allocation_1 = 0
        self.pe_allocation_2 = 0

        self.mem_allocation_0 = 0
        self.mem_allocation_1 = 0
        self.mem_allocation_2 = 0

        self.bus_allocation_0 = 0
        self.bus_allocation_1 = 0
        self.bus_allocation_2 = 0

        self.pe_to_bus_connection_0 = 0
        self.pe_to_bus_connection_1 = 0
        self.pe_to_bus_connection_2 = 0

        self.bus_to_bus_connection_0 = -1
        self.bus_to_bus_connection_1 = -1
        self.bus_to_bus_connection_2 = -1

        self.bus_to_mem_connection_0 = -1
        self.bus_to_mem_connection_1 = -1
        self.bus_to_mem_connection_2 = -1

        self.task_to_pe_mapping_0 = 0
        self.task_to_pe_mapping_1 = 0
        self.task_to_pe_mapping_2 = 0
        self.task_to_pe_mapping_3 = 0
        self.task_to_pe_mapping_4 = 0
        self.task_to_pe_mapping_5 = 0
        self.task_to_pe_mapping_6 = 0
        self.task_to_pe_mapping_7 = 0

        self.task_to_mem_mapping_0 = 0
        self.task_to_mem_mapping_1 = 0
        self.task_to_mem_mapping_2 = 0
        self.task_to_mem_mapping_3 = 0
        self.task_to_mem_mapping_4 = 0
        self.task_to_mem_mapping_5 = 0
        self.task_to_mem_mapping_6 = 0
        self.task_to_mem_mapping_7 = 0

        self.traject_dir = traject_dir
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.reward_formulation = reward_formulation
        self.workload = workload
        self.use_envlogger = use_envlogger
        self.env = FARSI_sim_wrapper.make_FARSI_sim_env(
            reward_formulation=self.reward_formulation, workload=self.workload)

        # Single layer network implementation
        for node in path:
            if hasattr(node, "pe_allocation_0"):
                self.action_dict["pe_allocation"] = [
                    node.pe_allocation_0, node.pe_allocation_1, node.pe_allocation_2]
                self.action_dict["mem_allocation"] = [
                    node.mem_allocation_0, node.mem_allocation_1, node.mem_allocation_2]
                self.action_dict["bus_allocation"] = [
                    node.bus_allocation_0, node.bus_allocation_1, node.bus_allocation_2]
                self.action_dict["pe_to_bus_connection"] = [
                    node.pe_to_bus_connection_0, node.pe_to_bus_connection_1, node.pe_to_bus_connection_2, ]
                self.action_dict["bus_to_bus_connection"] = [
                    node.bus_to_bus_connection_0, node.bus_to_bus_connection_1, node.bus_to_bus_connection_2, ]
                self.action_dict["bus_to_mem_connection"] = [
                    node.bus_to_mem_connection_0, node.bus_to_mem_connection_1, node.bus_to_mem_connection_2, ]
                self.action_dict["task_to_pe_mapping"] = [node.task_to_pe_mapping_0,
                                                          node.task_to_pe_mapping_1,
                                                          node.task_to_pe_mapping_2,
                                                          node.task_to_pe_mapping_3,
                                                          node.task_to_pe_mapping_4,
                                                          node.task_to_pe_mapping_5,
                                                          node.task_to_pe_mapping_6,
                                                          node.task_to_pe_mapping_7,
                                                          ]
                self.action_dict["task_to_mem_mapping"] = [node.task_to_mem_mapping_0,
                                                           node.task_to_mem_mapping_1,
                                                           node.task_to_mem_mapping_2,
                                                           node.task_to_mem_mapping_3,
                                                           node.task_to_mem_mapping_4,
                                                           node.task_to_mem_mapping_5,
                                                           node.task_to_mem_mapping_6,
                                                           node.task_to_mem_mapping_7,
                                                           ]

    def wrap_in_envlogger(self, env, envlogger_dir, use_envlogger):
        metadata = {
            'agent_type': 'ACO',
            'env_type': type(env).__name__,
        }
        if use_envlogger == True:
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

    def fit(self, X, y=None):
        '''
        This is the function that is called by the optimizer. ACO by defaul tries to minimize the fitness function.
        If you have a fitness function that you want to maximize, you can simply return the negative of the fitness function.
        '''

        FARSI_sim_helper = helpers()
        design_space_mode = "limited"  # ["limited", "comprehensive"]
        SOC_design_space = FARSI_sim_helper.gen_SOC_design_space(
            self.env, design_space_mode)
        encoding_dictionary = FARSI_sim_helper.gen_SOC_encoding(
            self.env, SOC_design_space)

        if self.use_envlogger:
            # check if trajectory directory exists
            if not os.path.exists(self.traject_dir):
                os.makedirs(self.traject_dir)

        # check if log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        env = self.wrap_in_envlogger(
            self.env, self.traject_dir, self.use_envlogger)
        env.reset()
        '''
        with envlogger.EnvLogger(env_wrapper,
                                 data_directory=self.traject_dir,
                                 max_episodes_per_file=1000,
                                 metadata={
                                     'agent_type': 'aco',
                                     'env_type': type(env_wrapper).__name__
                                 }, step_fn=step_fn) as env:
            env.reset()
        '''
        # flatten action_dict into one list
        flattened_action = []
        flattened_action.extend(self.action_dict["pe_allocation"])
        flattened_action.extend(self.action_dict["mem_allocation"])
        flattened_action.extend(self.action_dict["bus_allocation"])
        flattened_action.extend(self.action_dict["pe_to_bus_connection"])
        flattened_action.extend(self.action_dict["bus_to_bus_connection"])
        flattened_action.extend(self.action_dict["bus_to_mem_connection"])
        flattened_action.extend(self.action_dict["task_to_pe_mapping"])
        flattened_action.extend(self.action_dict["task_to_mem_mapping"])

        action = FARSI_sim_helper.action_decoder_FARSI(
            flattened_action, encoding_dictionary)
        _, reward, _, info = env.step(action)

        action_dict_for_logging = {}
        for key in action.keys():
            if "encoding" not in key:
                action_dict_for_logging[key] = action[key]

        self.fitness_hist['action_dict'] = action_dict_for_logging
        self.fitness_hist["reward"] = reward.item()
        self.fitness_hist["obs"] = [metric.item() for metric in info]

        print("Reward: ", reward)
        print("Action Dict: ", self.action_dict)
        print("Info: ", info)

        self.log_fitness_to_csv()
        return -1 * reward

    def log_fitness_to_csv(self):
        df_traj = pd.DataFrame([self.fitness_hist])
        filename = os.path.join(self.log_dir, self.exp_name + "_traj.csv")
        df_traj.to_csv(filename,
                       index=False, header=False, mode='a')

        df_rewards = pd.DataFrame([self.fitness_hist['reward']])
        filename = os.path.join(self.log_dir, self.exp_name + "_rewards.csv")
        df_rewards.to_csv(filename,
                          index=False, header=False, mode='a')
