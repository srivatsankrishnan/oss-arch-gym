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

## Install acme

Activate the arch-gym conda environment, then in ```oss-arch-gym/acme``` run 
```
pip install .[tf,testing,envs,jax]
```

Now, run ```conda env list```, which will show the path to your arch-gym environment.

Add the following line to your bashrc:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<PATH_TO_ARCH-GYM_CONDA_ENV>/lib/
```