#!/bin/bash

cuda_device=0
# Loop through directories from rank0_sharded_0 to rank0_sharded_9
# for i in {0..9}; do
#     work_dir="rank0_sharded_${i}"
#     cuda_device=$(( i % 8 ))  # Calculate CUDA_VISIBLE_DEVICES from 0 to 7

experiment_run="toys"
model_dir="t5_${experiment_run}_best_dev_no_fusin_decoder"
#model_dir="pretrained_model/t5-base"
leng=3
leng_plus_one=4

CUDA_VISIBLE_DEVICES=0 python genEmbedding_text_alike.py \
    --text_alike "text_alike_${experiment_run}_size_${leng}_test.jsonl" \
    --tokenizer "pretrained_model/t5-base/" \
    --model_dir "${model_dir}" \
    --text_alike_output_embedding "text_alike_size_${leng}_embeddings/text_alike_size_${leng}_output_embedding_${experiment_run}.jsonl" \
    --split_num 502 \
    --num_passage 1 \
    --text_alike_max_token 512 \
    --batch_size 128  \
    --leng 3



    # echo "launch ${work_dir} with CUDA device ${cuda_device}"
# done
