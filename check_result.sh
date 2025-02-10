#!/bin/bash
#SBATCH --job-name llama-ssd # your job name here
#SBATCH --gres=gpu:1 # if you need 4 GPUs, fixit to 4
#SBATCH --partition PB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# YOUR SCRIPT GOES HERE
echo $SHELL
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate base

export PATH="/home/n0/gihwan/miniconda3/bin:$PATH"  # commented out by conda initialize

# torchrun --nproc_per_node 1 check_result.py --db_dir=$HOME/data/db \
#   --cache_dir=$HOME/data/cache --query_file=./questions/query.jsonl  --top_k=2 --use_past_cache=False
torchrun --nproc_per_node 1 check_result.py --db_dir=data/db \
  --cache_dir=data/cache --query_file=./questions/query.jsonl --top_k=2 --use_past_cache=True

