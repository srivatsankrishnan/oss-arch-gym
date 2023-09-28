# AstraSim Simulator Documentation

## Running Training Scripts

Inside sims/AstraSim:

* **Ant Colony Optimization**: run trainACOAstraSim.py

* **Bayesian Optimization**: run trainBOAstraSim.py

* **Genetic Algorithm**: run trainGAAstraSim.py

* **Random Walker**: run trainRandomWalkerAstraSim.py

* **Reinforcement Learning**: run trainSingleAgentAstraSim.py

To update the input network, system, and workload files for the training scripts, follow these steps:

For GA and Random Walker, define the input file paths in the network_file, system_file, and workload_file variables in the training scripts.

For ACO, in aco/deepswarm/backends.py, define the network and workload file paths in self.action_dict["network"]['path'] and self.action_dict["workload"]['path']. Define the system file path self.system_file. 

For BO, define the input file paths in the network_file, system_file, and workload_file variables in bo/AstraSimEstimator.py.

For RL, define the input file paths in the network_file, system_file, and workload_file variables in envs/AstraSimEnv.py.

## Updating Hyperparameters

## Parameter Space

## Paramter Mapping
