# AstraSim Simulator Documentation

## Installing ArchGym + Astra-sim (Docker)
```
git clone https://github.com/srivatsankrishnan/oss-arch-gym.git
cd oss-arch-gym/sims/AstraSim && docker build -t archgym-astrasim .
```

## Running Training Scripts (Docker)

First, in Dockerfile line 38, replace x in --algo=<x> with the desired algorithm to run, such that x \in {aco, bo, ga, rw, rl}. (These flags correspond to Ant Colony Optimization, Bayesian Optimization, Genetic Algorithm, Random Walker, and Reinforcement Learning, respectively.)

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "arch-gym", "python", "launch_gcp.py", "--algo=bo"]

Rebuild the docker for each of the five algorithms and tag them with different names (such as "archgym-astrasim-aco" and "archgym-astrasim-bo"). Then, run each of the five algorithms using their separate docker images using the following mount command:

```
docker run -v <output path on local machine>:/workdir/oss-arch-gym/sims/AstraSim/all_logs -it <name_of_docker_image>
```

Alternatively, if you've built a docker image without an endpoint, run the algorithm inside the docker environment as follows: 

First, run the docker environment: 
```
docker run -it <hash> /bin/bash
```
Then, in /workdir/oss-arch-gym/sims/AstraSim/ of the docker environment (make sure arch-gym conda environment is activated), run: 

* **Ant Colony Optimization**: ```python trainACOAstraSim.py```

* **Bayesian Optimization**: ```python trainBOAstraSim.py```

* **Genetic Algorithm**: ```python trainGAAstraSim.py```

* **Random Walker**: ```python trainRandomWalkerAstraSim.py```

* **Reinforcement Learning**: ```python trainSingleAgentAstraSim.py```

The logs will be outputted in the /workdir/oss-arch-gym/sims/AstraSim/all_logs/ folder. 


## Updating Knobs: 

To update the input network, system, and workload files for the training scripts, follow these steps:

For GA and Random Walker, define the input file paths in the network_file, system_file, and workload_file variables in the training scripts.

For ACO, in ```aco/deepswarm/backends.py```, define the network and workload file paths in ```self.action_dict["network"]['path']``` and ```self.action_dict["workload"]['path']```. Define the system file path self.system_file. 

For BO, define the input file paths in the network_file, system_file, and workload_file variables in ```bo/AstraSimEstimator.py```.

For RL, define the input file paths in the network_file, system_file, and workload_file variables in ```envs/AstraSimEnv.py```.


## Updating Hyperparameters

For ACO, update hyperparameters such as evaporation, ant count, greediness, and depth in ```sims/AstraSim/trainACOAstraSim.py```.

For BO, update hyperparameters such as number of training steps in ```sims/AstraSim/trainBOAstraSim.py```.

For GA, update hyperparameters such as number of steps, number of agents, and probability of mutation in ```sims/AstraSim/trainGAAstraSim.py```.

For RW, update hyperparameters such as number of steps and number of episodes in ```sims/AstraSim/trainRandomWalkerAstraSim.py```.

For RL, update hyperparameters such as max_steps in ```sims/AstraSim/AstraSimWrapper.py```.


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

