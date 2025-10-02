#!/bin/bash
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH -t 48:00:00
#SBATCH --mem=100G
#SBATCH -o rare_out.txt
#SBATCH -e rare_err.txt

# Load CUDA modules
module load cuda/12.6
module load cudnn/8.9.7.29-12-cuda12.6

# Project root
PROJECT_DIR=/project/pi_hongyu_umass_edu/zonghai/rare_mediq

# Activate the venv that has uv installed
source $PROJECT_DIR/rare_mediq_venv/bin/activate

# Step 1: Prepare corpus
# cd $PROJECT_DIR/rare_mediq/MedRAG
# python -m uv run python prepare_corpus.py PubMed --db_dir ./corpus --output_dir ./chunks

# # Step 2: Build retrieval index
python -m uv run python index_colbert.py pubmed ./chunks/PubMed_chunks.pickle

# Step 3: Start ColBERT API (background process)
cd $PROJECT_DIR/rare_mediq/run_src
python -m python MedRAG/index_colbert.py pubmed MedRAG/chunks/PubMed_chunks.pickle

# Step 4: Run RARE
cd $PROJECT_DIR/rare_mediq
python -m uv run bash scripts/run_generate_medqa_qwen0_6b.sh
