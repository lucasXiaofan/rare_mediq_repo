#!/bin/bash
# Run the MediQ interactive benchmark with DeepSeek for every LLM call.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MEDIQ_SRC="$REPO_ROOT/mediQ/src"
DATA_DIR="${REPO_ROOT}/data"
CONDA_ENV="ml_env"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    echo "[INFO] Activating conda environment: ${CONDA_ENV}"
    conda activate "${CONDA_ENV}"
else
    echo "[WARNING] Conda not found on PATH; proceeding without activating ${CONDA_ENV}" >&2
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "[ERROR] Expected data directory at ${DATA_DIR}. Update DATA_DIR in this script if your data lives elsewhere." >&2
    exit 1
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "[INFO] DEEPSEEK_API_KEY is not set. The helper will try to load a key from mediQ/src/keys.py (api_account=deepseek)." >&2
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_NAME="${DEEPSEEK_MODEL_NAME:-deepseek-chat}"
OUTPUT_DIR="${REPO_ROOT}/outputs"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

OUTPUT_FILE="${OUTPUT_DIR}/mediq_deepseek_${TIMESTAMP}.jsonl"
LOG_FILE="${LOG_DIR}/mediq_deepseek_${TIMESTAMP}.log"
HISTORY_LOG="${LOG_DIR}/mediq_deepseek_history_${TIMESTAMP}.log"
DETAIL_LOG="${LOG_DIR}/mediq_deepseek_detail_${TIMESTAMP}.log"
MESSAGE_LOG="${LOG_DIR}/mediq_deepseek_messages_${TIMESTAMP}.log"

python3 "$MEDIQ_SRC/mediQ_benchmark.py" \
    --expert_module expert \
    --expert_class ScaleExpert \
    --expert_model "$MODEL_NAME" \
    --expert_model_question_generator "$MODEL_NAME" \
    --patient_module patient \
    --patient_class InstructPatient \
    --patient_model "$MODEL_NAME" \
    --data_dir "$DATA_DIR" \
    --dev_filename all_dev_good.jsonl \
    --output_filename "$OUTPUT_FILE" \
    --log_filename "$LOG_FILE" \
    --history_log_filename "$HISTORY_LOG" \
    --detail_log_filename "$DETAIL_LOG" \
    --message_log_filename "$MESSAGE_LOG" \
    --max_questions 10 \
    --temperature 0.6 \
    --max_tokens 1500 \
    --top_p 0.9 \
    --self_consistency 3 \
    --abstain_threshold 3 \
    --use_api deepseek \
    --api_account deepseek \
    --api_base_url "https://api.deepseek.com"

echo "Run completed. Results saved to ${OUTPUT_FILE}"
