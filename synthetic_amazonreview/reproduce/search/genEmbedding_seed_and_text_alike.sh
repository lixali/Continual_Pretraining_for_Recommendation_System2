#!/bin/bash

cuda_device=0
# Loop through directories from rank0_sharded_0 to rank0_sharded_9
# for i in {0..9}; do
#     work_dir="rank0_sharded_${i}"
#     cuda_device=$(( i % 8 ))  # Calculate CUDA_VISIBLE_DEVICES from 0 to 7

experiment_run="sports"
model_dir="t5_${experiment_run}_best_dev"
#model_dir="pretrained_model/t5-base"

max_leng=14
Ks=6
fusin=1
for (( leng=1; leng<=max_leng; leng++ )); do
    #text_alike="data/${experiment_run}/train_${leng}_item.jsonl"
    
    
    text_alike="text_alike_after_search/text_alike_${experiment_run}_size_${leng}_fusin_${fusin}.jsonl"
    CUDA_VISIBLE_DEVICES=1 python genEmbedding_seed_and_text_alike.py \
        --text_alike "${text_alike}" \
        --tokenizer "pretrained_model/t5-base/" \
        --model_dir "${model_dir}" \
        --text_alike_output_embedding "text_alike_embeddings/text_alike_size_${leng}_output_embedding_${experiment_run}_fusin_${fusin}.jsonl" \
        --split_num 243 \
        --num_passage 2 \
        --text_alike_max_token 256 \
        --batch_size 256  \
        --leng "${leng}"

    echo "leng is ${leng} : genEmbedding for leng ${leng} tasks completed!!!!!" 

    wait 
    
    
    leng_plus_one=$((leng+1))
    pool_embedding="data/${experiment_run}/pool_item_embedding_${experiment_run}_fusin_${fusin}.jsonl"
    text_alike_embedding="text_alike_embeddings/text_alike_size_${leng}_output_embedding_${experiment_run}_fusin_${fusin}.jsonl"
    output_file="text_alike_after_search/text_alike_${experiment_run}_size_${leng_plus_one}_fusin_${fusin}.jsonl"
   
    echo "leng is ${leng} : embedding similairy for leng ${leng} tasks is starting now!!!!!" 

    python embedding_similarity_seed_and_text_alike_vs_pool.py \
    	--pool_embedding "${pool_embedding}" \
    	--text_alike_embedding "${text_alike_embedding}" \
    	--output_file "${output_file}" \
    	--leng "${leng}" \
	--Ks ${Ks}

    
    echo "leng is ${leng} : embedding similairy for leng ${leng} tasks completed!!!!!" 
    
    wait


    Ks=$((Ks+1))
done
