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

## 1b) Native Install
###### Clone ArchGym Repository 
```shell
git clone https://github.com/srivatsankrishnan/oss-arch-gym.git
```
###### Install Anaconda / Miniconda if not already present
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
```

###### Create Conda Environment for ArchGym
```
conda env create -f oss-arch-gym/environment.yml
```

###### Install ACME Framework

Activate the arch-gym conda environment, then in `oss-arch-gym/acme` run: 
```
pip install .[tf,testing,envs,jax]
```

Now run `conda env list`, which will show the path to your arch-gym environment.

Add the following line to your bashrc:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<PATH_TO_ARCH-GYM_CONDA_ENV>/lib/
```

###### Install Vizier 
From the repository root `oss-arch-gym` run:
```
./install_sim.sh viz
```

## 2) Activate ArchGym Environment 
```
conda activate arch-gym
```

## 3) Verify Setup
Navigate to `sims/DRAMSys` and run:
```
python train_ga_DRAMSys.py
```
You should see the following output:
```

```
