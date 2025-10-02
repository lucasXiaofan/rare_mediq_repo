#!/bin/bash
#SBATCH -c 8
#SBATCH -p gpu                # generic GPU partition
#SBATCH --gpus=1              # request 1 GPU
#SBATCH --constraint=a100     # request specifically A100 GPU
#SBATCH -t 48:00:00           # max runtime
#SBATCH --mem=100G            # memory
#SBATCH -o rare_out.txt       # STDOUT
#SBATCH -e rare_err.txt       # STDERR

# Load CUDA (includes cuDNN on Unity cluster)
module load cuda/12.6
module load cudnn/8.9.7.29-12-cuda12.6
module load conda/latest   # Unity cluster conda

# Initialize conda (if needed on your cluster)
source ~/.bashrc

# Activate conda environment
conda activate rare_mediq_p310

# Move to project directory
cd /project/pi_hongyu_umass_edu/zonghai/rare_mediq/rare_mediq || { echo "Project directory not found"; exit 1; }

# Run the training/generation script
bash scripts/run_generate_medqa_qwen0_6b.sh
