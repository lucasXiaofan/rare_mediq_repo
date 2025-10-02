export VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 run_src/do_rare.py \
    --dataset_name MedQA \
    --test_json_filename test_all \
    --model_ckpt Qwen/Qwen3-0.6B \
    --note default \
    --num_rollouts 4 \
    --mode run \
    --disable_a5 \
    --disable_a8 \
    --disable_a4 \
    --disable_a6 \
    --disable_a7 \
    --enable_chat_template true \
    --num_retrieval 5 \
    --retrieval_corpus "pubmed" \
