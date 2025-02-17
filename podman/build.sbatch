#!/bin/bash -l
#SBATCH --job-name="build"
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --gpus-per-node=4          # number of gpus per node
#SBATCH -c 288
#SBATCH --mem=400G
#SBATCH --exclusive
#SBATCH --time=3:00:00            # total run time limit (HH:MM:SS)

cd /capstor/scratch/cscs/ctianche/projects/crossdc/Megatron-LM/podman
bash rebuild.sh