#!/bin/bash

cuda_device=0
# Loop through directories from rank0_sharded_0 to rank0_sharded_9
# for i in {0..9}; do
#     work_dir="rank0_sharded_${i}"
#     cuda_device=$(( i % 8 ))  # Calculate CUDA_VISIBLE_DEVICES from 0 to 7

run="beauty"
experiment_run="beauty_iter_2"
model_dir="t5_${run}_best_dev_from_iter_1"
#model_dir="pretrained_model/t5-base"

CUDA_VISIBLE_DEVICES=0 python genEmbedding_text_alike_size_2.py \
    --text_alike "text_alike_${experiment_run}_size_2.jsonl" \
    --tokenizer "pretrained_model/t5-base/" \
    --model_dir "${model_dir}" \
    --text_alike_output_embedding "text_alike_size_2_embeddings/text_alike_size_2_output_embedding_${experiment_run}.jsonl" \
    --split_num 502 \
    --num_passage 1 \
    --text_alike_max_token 512 \
    --batch_size 128 



    # echo "launch ${work_dir} with CUDA device ${cuda_device}"
# done
