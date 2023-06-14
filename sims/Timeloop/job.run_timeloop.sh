#!/bin/bash -x

#SBATCH -n 32                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p seas_dgx1
#SBATCH -t 1-00:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem=32000           # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o joblogs/%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e joblogs/%A_%a.err  # File to which STDERR will be written, %j inserts jobid


set -x

date
cdir=$(pwd)

outputdir="${cdir}/output/${SLURM_JOB_ID}"
mkdir -p $outputdir
echo $outputdir

mapperdir="/scratch/susobhan/${SLURM_JOB_ID}/mapper"
mkdir -p $mapperdir
cp -r ./mapper/mapper.yaml $mapperdir

archdir="/scratch/susobhan/${SLURM_JOB_ID}/arch"
mkdir -p $archdir
cp -r ./arch/* $archdir

scriptdir="/scratch/susobhan/${SLURM_JOB_ID}/script"
mkdir -p $scriptdir
cp -r ./script/* $scriptdir

settingsdir="/scratch/susobhan/${SLURM_JOB_ID}/settings"
mkdir -p $settingsdir
cp /n/janapa_reddi_lab/Lab/susobhan/arch-gym/settings/default_timeloop.yaml $settingsdir

source /n/home12/susobhan/.bashrc
conda activate /n/home12/susobhan/.conda/envs/arch-gym

export USER_UID=$UID
export USER_GID=$(id -g)

python collect_data.py ${SLURM_ARRAY_TASK_ID} $outputdir $mapperdir $archdir $scriptdir $settingsdir

# To run first 4 experiments, use: sbatch --array=0-3 job.run_timeloop.sh