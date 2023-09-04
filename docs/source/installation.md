# Setting up ArchGym Environment

## Clone ArchGym Repository 
```shell
git clone https://github.com/srivatsankrishnan/oss-arch-gym.git
```
## Install Anaconda/ Miniconda if not already present
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
```

## Create conda environment for arch-gym which has all the required dependencies
```
conda env create -f oss-arch-gym/environment.yml
```

## Activate arch-gym environment 
```
conda activate arch-gym
```

## Install Vizier 

### Clone Vizier Repository
```
git clone https://github.com/ShvetankPrakash/vizier
```

### Change directory into vizier 
```
cd vizier/
```

### Run the following commands on terminal 
```
sudo apt-get install -y libprotobuf-dev # Needed for proto libraries.
pip install -r requirements.txt --use-deprecated=legacy-resolver # Installs dependencies
pip install -e . # Installs Vizier
./build_protos.sh
```

## Install Algo and Benchmark Dependencies
```
pip install -r requirements-algorithms.txt
pip install -r requirements-benchmarks.txt
```

## Verify Vizier Installation 
- Make a copy of this script [vizier_dse.py](https://github.com/google/CFU-Playground/blob/main/proj/dse_template/vizier_dse.py)
- Remove line 10
- Replace line 40 to return random int ; for example: cycles, cells = 1, 1
- Run this file, sample output should look something like this:
EXITING DSE...

Optimal Trial Suggestion and Objective: ParameterDict(_items={'bypass': True, 'cfu': False, 'hardwareDiv': True, 'mulDiv': False, 'singleCycleShift': False, 'singleCycleMulDiv': True, 'safe': False, 'prediction': dynamic_target, 'iCacheSize': 4096.0, 'dCacheSize': 512.0}) Measurement(metrics={'cycles': Metric(value=16638.0, std=None), 'cells': Metric(value=1152797428.0, std=None)}, elapsed_secs=0.0, steps=0)
Optimal Trial Suggestion and Objective: ParameterDict(_items={'bypass': False, 'cfu': True, 'hardwareDiv': False, 'mulDiv': False, 'singleCycleShift': True, 'singleCycleMulDiv': True, 'safe': True, 'prediction': dynamic, 'iCacheSize': 4096.0, 'dCacheSize': 8192.0}) Measurement(metrics={'cycles': Metric(value=17932.0, std=None), 'cells': Metric(value=911708646.0, std=None)}, elapsed_secs=0.0, steps=0)

## Fixes for some common errors while Vizier installation 
1. Errors related to "cvxopt" can be fixed using the following commands:
```
pip install cvxopt
export CVXOPT_BUILD_FFTW=1
pip install cvxopt --no-binary cvxopt
conda install -c conda-forge cvxopt
```

2. “No module named emukit”
```
pip install emukit
```

If pip install emukit throws an error related to gcc compiler, then try installing it using this -
```
sudo apt update && sudo apt install -y build-essential
```

You may further encounter errors related to : E: Could not open lock file /var/lib/ dpkg/lock-frontend - open (13:Permission denied) E: Unable to acquire the dpkg frontend lock /var/lib/dpkg/lock-frontend, are you root?

To solve this follow this :
```
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock*
sudo apt update
```
Run this again : 
```
sudo apt update && sudo apt install -y build-essential 
```

Gcc installation is done, now you can proceed with pip install emukit 
Now try running vizier_dse.py again. It should run without errors. 
