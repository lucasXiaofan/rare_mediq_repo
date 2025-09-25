#!/bin/bash
#SBATCH -c 8
#SBATCH -p gpu               # generic GPU partition
#SBATCH --gpus=1             # request 1 GPU
#SBATCH --constraint=a100  
#SBATCH -t 48:00:00
#SBATCH --mem=100G
#SBATCH -o rare_out.txt
#SBATCH -e rare_err.txt

# Load CUDA (includes cuDNN on Unity cluster)
module load cuda/12.1

# Activate environment
source ~/rare_env/bin/activate

# Run
cd ~/RARE
bash scripts/run_generate_medqa_qwen0_6b.sh
