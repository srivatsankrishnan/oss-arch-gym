# Install Vizier

Activate the arch-gym conda environment, then in the repository root run the following script to install vizier :

```
./install_sim.sh viz
```

## Verify Vizier Installation 
Run the following:
```
python ./vizier_verify.py
```

The output should look like this:

Optimal Trial Suggestion and Objective: ParameterDict(_items={'bypass': False, 'cfu': True, 'hardwareDiv': True, 'mulDiv': True, 'singleCycleShift': True, 'singleCycleMulDiv': True, 'safe': False, 'prediction': none, 'iCacheSize': 8192.0, 'dCacheSize': 8192.0}) Measurement(metrics={'cycles': Metric(value=0.0, std=None), 'cells': Metric(value=0.0, std=None)}, elapsed_secs=0.0, steps=0)
Optimal Trial Suggestion and Objective: ParameterDict(_items={'bypass': False, 'cfu': True, 'hardwareDiv': True, 'mulDiv': False, 'singleCycleShift': True, 'singleCycleMulDiv': False, 'safe': False, 'prediction': none, 'iCacheSize': 4096.0, 'dCacheSize': 16384.0}) Measurement(metrics={'cycles': Metric(value=0.0, std=None), 'cells': Metric(value=0.0, std=None)}, elapsed_secs=0.0, steps=0)

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
