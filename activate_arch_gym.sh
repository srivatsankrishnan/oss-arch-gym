# Use this to activate conda environment in FASRC cluster
module load Anaconda3/5.0.1-fasrc01

source activate arch-gym

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/arch-gym/lib/
