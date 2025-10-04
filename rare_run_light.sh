#!/bin/bash
# Lightweight helper script for ad-hoc RARe runs.
# Mirrors the environment setup in rare_run_noRAG.sh but captures
# all three MediQ prompt variants in a single execution.

#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH -t 04:00:00
#SBATCH --mem=100G
#SBATCH -o rare_light_out.txt
#SBATCH -e rare_light_err.txt

module load cuda/12.6
module load cudnn/8.9.7.29-12-cuda12.6
module load conda/latest

source ~/.bashrc
conda activate rare_mediq_p310

cd /project/pi_hongyu_umass_edu/zonghai/rare_mediq/rare_mediq || { echo "Project directory not found"; exit 1; }

export VLLM_WORKER_MULTIPROC_METHOD=spawn

python run_src/rare_light_runner.py \
    --prompt-source all \
    --dataset-name MedQA \
    --test-json-filename test_all \
    --model-ckpt Qwen/Qwen3-0.6B \
    --api vllm \
    --use-vllm \
    --temperature 0.8 \
    --top-p 0.95 \
    --max-tokens 1024 \
    --num-rollouts 4 \
    --disable-a4 \
    --disable-a5 \
    --disable-a6 \
    --disable-a7 \
    --disable-a8 \
    --enable-chat-template \
    --num-retrieval 5 \
    --retrieval-corpus pubmed \
    --independent-modules
