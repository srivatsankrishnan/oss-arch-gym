# Setting Up a New Virtual Machine and Arch Gym Env Installation !

Follow the below instruction to setup a remote access of virtual machine and peform the necessary intallation for creating arch-gym enviroment


## Initializing and Starting VM
1. Generating the ssh public and private key using : `ssh-keygen -t rsa -b 2048 -C [USERNAME]`
2. Get the ssh access from adminstrator
3. Open terminal and run :`ssh -i <PATH_TO_PRIVATE_KEY> <USERNAME@IP_ADDRESS>`
4. Open VS code and download Remote-SSH extension by microsoft
5. Press F1 select Remote-SSH: Connect to Host...use the same `USERNAME@IP_ADDRESS` as in step 2
6. New VS Code window will be opened and If VS Code cannot automatically detect the type of server you are connecting to, you will be asked to select the type manually. 

## Installing Conda
In terminal run the following commands to install conda for your remote virtual machine
1. `curl -O https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh`
2. `sha256sum Anaconda3-2023.07-1-Linux-x86_64.sh`
3. `bash Anaconda3-2023.07-1-Linux-x86_64.sh ( press yes and enter for all steps )`
4. `source ~/.bashrc`


## Creating Arch-Gym Enviroment
Follow the below steps for setting up vizier and arch-gym. In terminal 
1. `git clone https://github.com/srivatsankrishnan/oss-arch-gym.git`
2. `cd oss-arch-gym/`
3. `conda env create -f environment.yml`
4. `conda activate arch-gym`
5. `cd ..`
6. `git clone https://github.com/ShvetankPrakash/vizier.git`
7. cd into vizier directory
8. `sudo apt-get install -y libprotobuf-dev`
9. `pip install -r requirements.txt --use-deprecated=legacy-resolver` ( you may see some package compatibility issues, ignore them )
10. `pip install -e .` ( you may see some package compatibility issues, ignore them )
11. `./build_protos.sh`
12. `pip install -r requirements-algorithms.txt` (you may probably end up with gcc compiler issue, ignore as of now)
13. `pip install -r requirements-benchmarks.txt` ( you may see some package compatibility issues, ignore them )
14. Open VS code and make a copy of this script:  [https://github.com/google/CFU-Playground/blob/main/proj/dse_template/vizier_dse.py](https://github.com/google/CFU-Playground/blob/main/proj/dse_template/vizier_dse.py)
15. Remove line 10 of your local copy
16. Go to line 40, comment it out and add this line `cycles, cells = 1, 1`
17.  In terminal Run `python vizier_dse.py` to test working. Note : all this should be done with arch-gym virtual env activated only
18. If you get ModuleNotFoundError: No module named 'emukit'. Run `pip install emukit`
19. If pip install emukit throws error related to gcc compiler, then try to install it using this -  `sudo apt update && sudo apt install -y build-essential`
20. Run `pip install emukit` again 
21.  Run `python vizier_dse.py` to test its working
The output should look like 
`Suggested Parameters (bypass, cfu, dCacheSize, hardwareDiv, iCacheSize, mulDiv, prediction, safe, singleCycleShift, singleCycleMulDiv): True False 8192.0 True 4096.0 True static False True False.............`


## Testing Overall Installation

Come out of vizier directory in terminal using `cd ..`

1. `cd oss-arch-gym/acme`
2. `pip install .[jax,tf,testing,envs]`
3. `which python`
	Output eg : `/home/<USERNAME>/anaconda3/envs/arch-gym/bin/python`
	Replace `bin/python` with `lib` and copy it : `/home/<USERNAME>/anaconda3/envs/arch-gym/lib`
####  In VS Code
1. Go to .bashrc file inside your username folder
2. Paste this in last : 	`export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/<USERNAME>/anaconda3/envs/arch-gym/lib/"`

#### In terminal
Remember that your arch-gym env should be activated all the times
1. Run  `sudo apt-get install libgmp-dev`
2. `cd oss-arch-gym/sims/customenv`
3. Rull all the Python files ( all should run without error )