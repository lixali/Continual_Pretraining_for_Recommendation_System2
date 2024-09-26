#!/bin/bash

cuda_device=0
# Loop through directories from rank0_sharded_0 to rank0_sharded_9
# for i in {0..9}; do
#     work_dir="rank0_sharded_${i}"
#     cuda_device=$(( i % 8 ))  # Calculate CUDA_VISIBLE_DEVICES from 0 to 7

experiment_run="sports_iter_2"
#model_dir="t5_${experiment_run}_best_dev"
model_dir="t5_sports_best_dev_from_iter_1"

CUDA_VISIBLE_DEVICES=0 python genEmbedding_seed.py \
    --seed_doc "seed_doc.jsonl" \
    --tokenizer "pretrained_model/t5-base/" \
    --model_dir "${model_dir}" \
    --seed_doc_output_embedding "seed_doc_embeddings_no_fusin_decoder/seed_doc_output_embedding_${experiment_run}.jsonl" \
    --split_num 503 \
    --num_passage 1 \
    --seed_doc_max_token 512 \
    --batch_size 128 



    # echo "launch ${work_dir} with CUDA device ${cuda_device}"
# done
