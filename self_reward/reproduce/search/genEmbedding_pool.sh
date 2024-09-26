#!/bin/bash


folder="pool_doc_splitted_no_fusin_decoder"
counter=0
experiment_run="sports"
#model_dir="t5_${experiment_run}_best_dev"
model_dir="t5_sports_best_dev"

for letter in {a..j}; do
    file="${folder}/output_prefix_a${letter}"
    output_file="${folder}/output_prefix_a${letter}_pool_doc_output_embedding_${experiment_run}.jsonl"
    cuda_device=$(( counter % 8 ))
     CUDA_VISIBLE_DEVICES=0 python genEmbedding_pool.py \
         --pool_doc $file \
         --tokenizer "pretrained_model/t5-base" \
         --model_dir "${model_dir}" \
         --pool_doc_output_embedding "${output_file}" \
         --num_passage 1 \
         --pool_doc_max_token_length 512 \
         --batch_size 128  
 
     counter=$((counter + 1))
echo "$cuda_device"
echo "$file"
done

wait 
cat "${folder}/"*"${experiment_run}".jsonl >  "${folder}/pool_doc_output_embedding_${experiment_run}.jsonl"
