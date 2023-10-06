# AstraSim Simulator Documentation

## Installing AstraSim

Clone AstraSim from this repo: 
```
git clone --recursive https://github.com/astra-sim/astra-sim.git
```

Install conda environment.

Run the compilation script with analytical backend: 
```
./build/astra_analytical/build.sh -c
```


## Running Training Scripts

Inside sims/AstraSim:

* **Ant Colony Optimization**: ```python trainACOAstraSim.py```

* **Bayesian Optimization**: ```python trainBOAstraSim.py```

* **Genetic Algorithm**: ```python trainGAAstraSim.py```

* **Random Walker**: ```python trainRandomWalkerAstraSim.py```

* **Reinforcement Learning**: ```python trainSingleAgentAstraSim.py```

To update the input network, system, and workload files for the training scripts, follow these steps:

For GA and Random Walker, define the input file paths in the network_file, system_file, and workload_file variables in the training scripts.

For ACO, in ```aco/deepswarm/backends.py```, define the network and workload file paths in ```self.action_dict["network"]['path']``` and ```self.action_dict["workload"]['path']```. Define the system file path self.system_file. 

For BO, define the input file paths in the network_file, system_file, and workload_file variables in ```bo/AstraSimEstimator.py```.

For RL, define the input file paths in the network_file, system_file, and workload_file variables in ```envs/AstraSimEnv.py```.


## Updating Hyperparameters

## Parameter Space
| System Parameter      | Values        |
| ----------------      | ------------- |
| scheduling-policy      | "FIFO", "LIFO"  |
| endpoint-delay      | 1-1000, Default: 1  |
| active-chunks-per-dimension      | 1-32, Default: 1  |
| preferred-dataset-splits      | 16-1024, Default: 64 |
| boost-mode      | 1  |
| all-reduce-implementation      | "ring", "direct", "oneRing", "oneDirect", "hierarchicalRing", "doubleBinaryTree", "halvingDoubling", "oneHalvingDoubling" |
| all-gather-implementation      | "ring", "direct", "oneRing", "oneDirect", "hierarchicalRing", "doubleBinaryTree", "halvingDoubling", "oneHalvingDoubling" |
| reduce-scatter-implementation      | "ring", "direct", "oneRing", "oneDirect", "hierarchicalRing", "doubleBinaryTree", "halvingDoubling", "oneHalvingDoubling" |
| all-to-all-implementation     | "ring", "direct", "oneRing", "oneDirect", "hierarchicalRing", "doubleBinaryTree", "halvingDoubling", "oneHalvingDoubling" |
| collective-optimization      | "localBWAware", "baseline" |
| intra-dimension-scheduling      | "FIFO", "SCF" |
| inter-dimension-scheduling    | "baseline", "themis" |


| Network Parameter  | Values        |
| ----------------  | ------------- |
| topology-name    | "Hierarchical"  |
| topologies-per-dim     | "Ring", "Switch", "FullyConnected"  |
| dimension-type    | "N"  |
| dimensions-count     | 1-5, Default: 2  |
| units-count    | 2-1024, Default: 32 |
| links-count    | 1-10, Default: 1 |
| link-latency    | 1-1000, Default: 1  |
| link-bandwidth    | 1e-5 - 1e5, Default: 100 |
| nic-latency    | 0-1000, Default: 0  |
| router-latency    | 0-1000, Default: 0 |
| hbm-latency    | 1 |
| hbm-bandwidth    | 1 |
| hbm-scale    | 1 |


## Parameter Mapping
