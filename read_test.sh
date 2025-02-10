#!/bin/bash
#SBATCH --job-name read-test # your job name here
#SBATCH --gres=gpu:1 # if you need 4 GPUs, fixit to 4
#SBATCH --partition PB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# YOUR SCRIPT GOES HERE
echo $SHELL
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate base

export PATH="/home/n0/gihwan/miniconda3/bin:$PATH"  # commented out by conda initialize

torchrun --nproc_per_node 1 read_test.py 

