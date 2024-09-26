#!/bin/bash

cuda_device=0
# Loop through directories from rank0_sharded_0 to rank0_sharded_9
# for i in {0..9}; do
#     work_dir="rank0_sharded_${i}"
#     cuda_device=$(( i % 8 ))  # Calculate CUDA_VISIBLE_DEVICES from 0 to 7
Ks=6
fusin=0


run="toys"
experiment_run="toys_from_Jingyuan"
model_dir="t5_${experiment_run}_best_dev"
text_alike="text_alike_after_search/text_alike_${experiment_run}_taste_${set}_fusin_${fusin}.jsonl"
num_passage=1
split_num=499
text_alike_max_token=512
#model_dir="pretrained_model/t5-base"
set=train

text_alike="text_alike_after_search/text_alike_${experiment_run}_taste_${set}_fusin_${fusin}.jsonl"
CUDA_VISIBLE_DEVICES=0 python genEmbedding_taste.py \
    --text_alike "${text_alike}" \
    --tokenizer "pretrained_model/t5-base/" \
    --model_dir "${model_dir}" \
    --text_alike_output_embedding "text_alike_embeddings/text_alike_taste_${set}_output_embedding_${experiment_run}_fusin_${fusin}.jsonl" \
    --split_num ${split_num} \
    --num_passage ${num_passage} \
    --text_alike_max_token ${text_alike_max_token} \
    --batch_size 256  

echo "genEmbedding for ${set} set completed!!!!!" 


wait 

set=valid

text_alike="text_alike_after_search/text_alike_${experiment_run}_taste_${set}_fusin_${fusin}.jsonl"
cuda_visible_devices=0 python genEmbedding_taste.py \
    --text_alike "${text_alike}" \
    --tokenizer "pretrained_model/t5-base/" \
    --model_dir "${model_dir}" \
    --text_alike_output_embedding "text_alike_embeddings/text_alike_taste_${set}_output_embedding_${experiment_run}_fusin_${fusin}.jsonl" \
    --split_num ${split_num} \
    --num_passage ${num_passage} \
    --text_alike_max_token ${text_alike_max_token} \
    --batch_size 256  


echo "genembedding for ${set} set completed!!!!!" 

wait

file="data/${run}/pool_item.jsonl"
output_file="data/${run}/pool_item_embedding_${model_dir}_fusin_${fusin}.jsonl"
cuda_device=$(( counter % 8 ))
CUDA_VISIBLE_DEVICES=0 python genEmbedding_pool.py \
     --pool_doc $file \
     --tokenizer "pretrained_model/t5-base" \
     --model_dir "${model_dir}" \
     --pool_doc_output_embedding "${output_file}" \
     --pool_doc_max_token_length 32 \
     --batch_size 128

counter=$((counter + 1))
echo "$cuda_device"
echo "$file"

