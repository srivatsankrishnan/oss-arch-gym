# Setting up ArchGym
Instructions for installing ArchGym using Docker and natively are both provided.
## 1a) Docker Install 
###### Clone latest ArchGym Image
```
git clone https://github.com/ShvetankPrakash/arch-gym-container.git
```

###### Build Image
```
docker build -t arch-gym-image .
```

###### Run Container
```
docker run -it --name arch-gym-container arch-gym-image
```

## 1b) Native Install (Tested on Ubuntu 20.04 LTS)

###### Install Basic Packages & Anaconda / Miniconda if not already present
```
sudo apt-get update -y
sudo apt-get install -y sudo git wget build-essential vim libgmp-dev 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
sudo bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
conda init
exit
```

###### Clone ArchGym Repository and Create Conda Environment 
Start a new terminal and run:
```
git clone https://github.com/srivatsankrishnan/oss-arch-gym.git
conda env create -f oss-arch-gym/environment.yml
conda activate arch-gym
```

###### Install ACME Framework for Reinforcement Learning
```
cd oss-arch-gym/acme
pip install .[tf,testing,envs,jax]
```

Now run `conda env list`, which will show the path to your arch-gym environment.

Add the following line to your `bashrc`:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<PATH_TO_ARCH-GYM_CONDA_ENV>/lib/
```
and then source your updated `bashrc` and re-activate the environment:
```
source ~/.bashrc
conda activate arch-gym
```

###### Install Vizier for Collection of Agents
From the repository root `oss-arch-gym` run:
```
./install_sim.sh viz
```

## 2) Verify Setup
###### Verify ArchGym Installation 
Run:
```
cd sims/DRAM
python train_ga_DRAMSys.py
```
You should see the following output (with perhaps different values):
```
iter: 0
Agents Action [  3.   2.   2.   8.   1.   0.   3.   6.   2. 103.]
[  3.   2.   2.   8.   1.   0.   3.   6.   2. 103.]
Action Dict {'PagePolicy': 'ClosedAdaptive', 'Scheduler': 'FrFcfs', 'SchedulerBuffer': 'Shared', 'RequestBufferSize': 8, 'RespQueue': 'Reorder', 'RefreshPolicy': 'NoRefresh', 'RefreshMaxPostponed': 3, 'RefreshMaxPulledin': 6, 'Arbiter': 'Reorder', 'MaxActiveTransactions': 103}
[envHelpers][Action] {'PagePolicy': 'ClosedAdaptive', 'Scheduler': 'FrFcfs', 'SchedulerBuffer': 'Shared', 'RequestBufferSize': 8, 'RespQueue': 'Reorder', 'RefreshPolicy': 'NoRefresh', 'RefreshMaxPostponed': 3, 'RefreshMaxPulledin': 6, 'Arbiter': 'Reorder', 'MaxActiveTransactions': 103}
[Environment] Observation: [2.46681075e-01 2.82248000e+00 8.73850000e-05]
Power: 2.82248 Latency: 8.7385e-05 Target Power: 1 Target Latency: 0.1
Episode: 0  Rewards: 0.5487028664237742
...
```
You can `ctrl-c` to kill the training run when you see the similar output above. You have successfully installed ArchGym!

###### Verify Vizier Installation 
Run:
```
 python train_randomsearch_vizier.py
```
You should see the following output (with perhaps different values):

```
...
['0' '2' '0' '32' '0' '1' '4' '2' '0' '32']
Action Dict {'PagePolicy': 'Open', 'Scheduler': 'FrFcfs', 'SchedulerBuffer': 'Bankwise', 'RequestBufferSize': 32, 'RespQueue': 'Fifo', 'RefreshPolicy': 'AllBank', 'RefreshMaxPostponed': 4, 'RefreshMaxPulledin': 2, 'Arbiter': 'Simple', 'MaxActiveTransactions': 32}
[envHelpers][Action] {'PagePolicy': 'Open', 'Scheduler': 'FrFcfs', 'SchedulerBuffer': 'Bankwise', 'RequestBufferSize': 32, 'RespQueue': 'Fifo', 'RefreshPolicy': 'AllBank', 'RefreshMaxPostponed': 4, 'RefreshMaxPulledin': 2, 'Arbiter': 'Simple', 'MaxActiveTransactions': 32}
E1018 11:11:07.816930157   15103 fork_posix.cc:76]           Other threads are currently calling into gRPC, skipping fork() handlers
[Environment] Observation: [2.41416e-01 3.17027e+00 7.61500e-05]
Power: 3.17027 Latency: 7.615e-05 Target Power: 1 Target Latency: 0.1
Episode: 0  Rewards: 0.4607721619890613
```

You can `ctrl-c` to kill the training run when you see the similar output above. You have successfully installed Vizier's Agents!
