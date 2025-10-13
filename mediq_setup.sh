#!/bin/bash
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH -t 48:00:00
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH -o logs/mediq_run_out_%j.txt
#SBATCH -e logs/mediq_run_err_%j.txt
#SBATCH --job-name=mediq_benchmark

# ----------------------------
# CRITICAL: Set CUDA environment FIRST, before any modules
# ----------------------------
# export CUDA_VISIBLE_DEVICES=0

# ----------------------------
# Load required modules
# ----------------------------
module load cuda/12.6
module load cudnn/8.9.7.29-12-cuda12.6
module load conda/latest

# ----------------------------
# Project setup
# ----------------------------
PROJECT_DIR="/project/pi_hongyu_umass_edu/zonghai/rare_mediq/rare_mediq/mediQ"
CONDA_ENV_NAME="rare_mediq_p310"

cd $PROJECT_DIR

echo "Current directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA version:"
nvcc --version

# ----------------------------
# Conda environment setup
# ----------------------------
eval "$(conda shell.bash hook)"

echo "Activating conda environment: $CONDA_ENV_NAME"
conda activate $CONDA_ENV_NAME

echo "Python version: $(python --version)"
python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" || echo "Torch import failed!"

# ----------------------------
# GPU check
# ----------------------------
echo "GPU info:"
nvidia-smi

# ----------------------------
# Paths
# ----------------------------
mkdir -p outputs logs
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH

# ----------------------------
# Run benchmark
# ----------------------------
echo "Starting MediQ benchmark..."
python src/mediQ_benchmark.py \
    --expert_module expert \
    --expert_class ScaleExpert \
    --expert_model Qwen/Qwen3-4B-Instruct-2507 \
    --patient_module patient \
    --patient_model Qwen/Qwen3-4B-Instruct-2507 \
    --self_consistency 3\
    --max_tokens 1500\
    --patient_class FactSelectPatient \
    --data_dir data \
    --abstain_threshold 3\
    --dev_filename all_dev_good.jsonl \
    --output_filename outputs/mediq_results_$(date +%Y%m%d_%H%M%S).jsonl \
    --log_filename logs/mediq_log_$(date +%Y%m%d_%H%M%S).txt \
    --history_log_filename logs/mediq_history_$(date +%Y%m%d_%H%M%S).txt \
    --message_log_filename logs/mediq_messages_$(date +%Y%m%d_%H%M%S).txt \
    --max_questions 3 \
    --use_vllm

echo "MediQ benchmark completed!"