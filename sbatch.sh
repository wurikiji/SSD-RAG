#!/bin/bash
#SBATCH --job-name llama-transformer # your job name here
#SBATCH --gres=gpu:1 # if you need 4 GPUs, fixit to 4
#SBATCH --partition PB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# YOUR SCRIPT GOES HERE
python preprocessing.py --docs_dir=./documents --db_dir=./db --cache_dir=./cache