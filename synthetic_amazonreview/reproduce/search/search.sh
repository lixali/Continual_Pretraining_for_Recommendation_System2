#!/bin/bash
#
#folders=("text_alike_after_search" "text_alike_embeddings" "data")
#
#for folder in "${folders[@]}"; do
#  if [ ! -d "$folder" ]; then
#    mkdir "$folder"
#    echo "Folder created: $folder"
#  else
#    echo "Folder already exists: $folder"
#  fi
#done
#
#
experiment_run="sports"
leng=1
fusin=1
#INPUT_FILE="data/${experiment_run}/train.txt"
#OUTPUT_FILE="text_alike_after_search/text_alike_${experiment_run}_size_${leng}_fusin_${fusin}.jsonl"
#ITEM_FILE="data/${experiment_run}/item.txt"
#
#python grepFirstItem.py  \
#	--input_file "${INPUT_FILE}" \
#	--item_file "${ITEM_FILE}" \
#	--output_file "${OUTPUT_FILE}"
#
#wait
#
#OUTPUT_FILE="data/${experiment_run}/pool_item.jsonl"
#ITEM_FILE="data/${experiment_run}/item.txt"
#
#python grepPoolItem.py  \
#        --item_file "${ITEM_FILE}" \
#        --output_file "${OUTPUT_FILE}"
#
#
#
#counter=0
##experiment_run="vanilla_t5"
##model_dir="t5_${experiment_run}_best_dev"
model_dir="pretrained_model/t5-base"
#
#
#file="data/${experiment_run}/pool_item.jsonl"
#output_file="data/${experiment_run}/pool_item_embedding_${experiment_run}_fusin_${fusin}.jsonl"
#cuda_device=$(( counter % 8 ))
#CUDA_VISIBLE_DEVICES=0 python genEmbedding_pool.py \
#     --pool_doc $file \
#     --tokenizer "pretrained_model/t5-base" \
#     --model_dir "${model_dir}" \
#     --pool_doc_output_embedding "${output_file}" \
#     --pool_doc_max_token_length 32 \
#     --batch_size 128
#
#counter=$((counter + 1))
#echo "$cuda_device"
#echo "$file"
#
#
#
#wait


######################### search starts #################################
cuda_device=0


max_leng=14
Ks=6
fusin=1
for (( leng=2; leng<=max_leng; leng++ )); do
    #text_alike="data/${experiment_run}/train_${leng}_item.jsonl"
    
    
    text_alike="text_alike_after_search/text_alike_${experiment_run}_size_${leng}_fusin_${fusin}.jsonl"
    CUDA_VISIBLE_DEVICES=0 python genEmbedding_seed_and_text_alike.py \
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


wait


# List of input files
folder="text_alike_after_search"
input_files="${folder}/text_alike_${experiment_run}_size_2_fusin_${fusin}.jsonl ${folder}/text_alike_${experiment_run}_size_3_fusin_${fusin}.jsonl \
        ${folder}/text_alike_${experiment_run}_size_4_fusin_${fusin}.jsonl ${folder}/text_alike_${experiment_run}_size_5_fusin_${fusin}.jsonl \
        ${folder}/text_alike_${experiment_run}_size_6_fusin_${fusin}.jsonl ${folder}/text_alike_${experiment_run}_size_7_fusin_${fusin}.jsonl \
        ${folder}/text_alike_${experiment_run}_size_8_fusin_${fusin}.jsonl ${folder}/text_alike_${experiment_run}_size_9_fusin_${fusin}.jsonl \
        ${folder}/text_alike_${experiment_run}_size_10_fusin_${fusin}.jsonl ${folder}/text_alike_${experiment_run}_size_11_fusin_${fusin}.jsonl \
        ${folder}/text_alike_${experiment_run}_size_12_fusin_${fusin}.jsonl ${folder}/text_alike_${experiment_run}_size_13_fusin_${fusin}.jsonl \
        ${folder}/text_alike_${experiment_run}_size_14_fusin_${fusin}.jsonl"

# Output file
output_file="${folder}/${experiment_run}_combined_2_to_14_fusin_${fusin}.jsonl"

# Run the Python script
python combine_text_alike.py \
        --input_file $input_files \
        --output_file $output_file
