#!/bin/bash

run="sports"
experiment_run="sports"
fusin=0
set="train"
#set="valid"
INPUT_FILE="data/${run}/${set}.txt"
OUTPUT_FILE="text_alike_after_search/text_alike_${experiment_run}_taste_${set}_fusin_${fusin}.jsonl"
ITEM_FILE="data/${run}/item.txt"
max_len=512
num_passage=1


folders=("text_alike_after_search" "text_alike_embeddings" "data")

for folder in "${folders[@]}"; do
  if [ ! -d "$folder" ]; then
    mkdir "$folder"
    echo "Folder created: $folder"
  else
    echo "Folder already exists: $folder"
  fi
done




python grepTaste.py  \
	--input_file "${INPUT_FILE}" \
	--item_file "${ITEM_FILE}" \
	--output_file "${OUTPUT_FILE}" \
	--max_len "${max_len}" \
	--num_passage "${num_passage}"


set="valid"
OUTPUT_FILE="text_alike_after_search/text_alike_${experiment_run}_taste_${set}_fusin_${fusin}.jsonl"
python grepTaste.py  \
	--input_file "${INPUT_FILE}" \
	--item_file "${ITEM_FILE}" \
	--output_file "${OUTPUT_FILE}" \
	--max_len "${max_len}" \
	--num_passage "${num_passage}"


Ks=6


text_alike="text_alike_after_search/text_alike_${experiment_run}_taste_${set}_fusin_${fusin}.jsonl"
split_num=499
text_alike_max_token=512
model_dir="pretrained_model/t5-base"
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



len=1
myset="train"
pool_embedding="data/${run}/pool_item_embedding_t5_${experiment_run}_fusin_${fusin}.jsonl"
text_alike_embedding="text_alike_embeddings/text_alike_taste_${myset}_output_embedding_${experiment_run}_fusin_${fusin}.jsonl"
output_file="text_alike_after_search/text_alike_${experiment_run}_taste_${myset}_fusin_${fusin}_all_sizes.jsonl"

python embedding_similarity_taste.py \
	--pool_embedding "${pool_embedding}" \
	--text_alike_embedding "${text_alike_embedding}" \
	--output_file "${output_file}" \
	--leng "${len}" \
        --Ks 6



wait
myset="valid"
text_alike_embedding="text_alike_embeddings/text_alike_taste_${myset}_output_embedding_${experiment_run}_fusin_${fusin}.jsonl"
output_file="text_alike_after_search/text_alike_${experiment_run}_taste_${myset}_fusin_${fusin}_all_sizes.jsonl"

python embedding_similarity_taste.py \
        --pool_embedding "${pool_embedding}" \
        --text_alike_embedding "${text_alike_embedding}" \
        --output_file "${output_file}" \
        --leng "${len}" \
        --Ks 6
