#!/bin/bash
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 48:00:00
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH -o logs/mediq_deepseek_out_%j.txt
#SBATCH -e logs/mediq_deepseek_err_%j.txt
#SBATCH --job-name=mediq_deepseek

# ----------------------------
# Load required modules
# ----------------------------
module load cuda/12.6
module load cudnn/8.9.7.29-12-cuda12.6
module load conda/latest

# ----------------------------
# Project setup
# ----------------------------
PROJECT_DIR="/project/pi_hongyu_umass_edu/zonghai/rare_mediq/rare_mediq"
CONDA_ENV_NAME="rare_mediq_p310"

cd "$PROJECT_DIR"

echo "Current directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "CUDA version:"
nvcc --version

# ----------------------------
# Conda environment setup
# ----------------------------
eval "$(conda shell.bash hook)"

echo "Activating conda environment: $CONDA_ENV_NAME"
conda activate "$CONDA_ENV_NAME"

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
export PYTHONPATH="$PROJECT_DIR/mediQ/src:$PYTHONPATH"

# ----------------------------
# Run DeepSeek benchmark
# ----------------------------
echo "Starting MediQ DeepSeek benchmark..."
python scripts/run_mediq_deepseek_10.py \
    --num_patients 10 \
    --model_name deepseek-chat \
    --data_dir mediQ/data \
    --dev_filename all_dev_good.jsonl \
    --output_dir outputs \
    --log_dir logs

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "MediQ DeepSeek benchmark completed!"
else
    echo "MediQ DeepSeek benchmark failed with exit code $exit_code"
fi

exit $exit_code
