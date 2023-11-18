# FARSI Simulator Documentation

##### 1. Clone ArchGym Repository and Create Conda Environment
Do these steps from the location where you intend to clone the ArchGym repository:
```
git clone https://github.com/srivatsankrishnan/oss-arch-gym.git
conda env create -f oss-arch-gym/environment.yml
conda activate arch-gym
```

##### 2. Install Vizier for Collection of Agents
From the repository root `oss-arch-gym` run:
```
./install_sim.sh viz
```

##### 3. Installing FARSI simulator
* The below commands are to replace the existing Project_FARSI folder with its latest version as a submodule. The shell script also updates the conda environment dependencies required for FARSI, and installs ACME framework for reinforcement learning.

(Note: the script takes a while to run): 
```
cd oss-arch-gym/
rm -r Project_FARSI
git rm -r --cached Project_FARSI
./install_sim.sh farsi
```

* Replace the content of `Project_FARSI/settings/config_cacti.py` file with this:

```
import os

# get the base path of arch-gym
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


cact_bin_addr = os.path.join(base_path, "Project_FARSI/cacti_for_FARSI/cacti")

print(cact_bin_addr, os.path.exists(cact_bin_addr))

cacti_param_addr = os.path.join(base_path, "Project_FARSI/cacti_for_FARSI/farsi_gen.cfg")

print(cacti_param_addr, os.path.exists(cacti_param_addr))

cacti_data_log_file = os.path.join(base_path, "Project_FARSI/cacti_for_FARSI/data_log.csv")

print(cacti_data_log_file, os.path.exists(cacti_data_log_file))

```

* In `Project_FARSI/settings/config.py`, replace the following line (line 276):
```
database_data_dir = os.path.join(home_dir, "specs", "database_data")
```
with this:
```
database_data_dir = os.path.join(home_dir, "oss-arch-gym", "Project_FARSI", "specs", "database_data")
```


## Running Training Scripts

Inside ```sims/FARSI_sim```:

* **Ant Colony Optimization**: ```python train_aco_FARSIEnv.py```

* **Bayesian Optimization**: ```python train_bo_FARSIEnv.py.py```

* **Genetic Algorithm**: ```python train_ga_FARSIEnv.py```

* **Random Walker**: ```python train_randomwalker_FARSIEnv.py```

* **Reinforcement Learning**: ```python train_single_agent.py```
  
* **Emukit Algorithm**: ```python train_emukit_vizier.py```
  
* **Grid-Search Algorithm**: ```python train_gridsearch_vizier.py```

* **Quasi-Random Algorithm**: ```python train_quasirandom_vizier.py```

* **Random-Search Algorithm**: ```python train_randomsearch_vizier.py```


## Updating Hyperparameters
You can update hyperparameters of the different algorithms in their respective .py files shown above.
