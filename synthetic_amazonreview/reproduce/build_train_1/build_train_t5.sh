#!/bin/sh





# T5 Family
model="t5-base"
data_set="sports"
experiment_run="sports"
myset="train"
num_of_passage=2

set="Amazon"
folder="name"
fusin=1

base_dir="data/${data_set}"
tokenizer_path="pretrained_model/${model}"
output_dir="data_taste_synthetic/${data_set}/"
text_alike="text_alike_after_search/text_alike_${experiment_run}_taste_${myset}_fusin_${fusin}_all_sizes.jsonl"



folders=("data_taste_synthetic/${data_set}")

for folder in "${folders[@]}"; do
  if [ ! -d "$folder" ]; then
    mkdir -p "$folder"
    echo "Folder created: $folder"
  else
    echo "Folder already exists: $folder"
  fi
done




echo $base_dir

python build_train.py  \
    --data_name $set  \
    --sample_num 100 \
    --train_file "${base_dir}/${myset}.txt"  \
    --item_file "${base_dir}/item.txt"  \
    --item_ids_file "${base_dir}/item_${folder}.jsonl"  \
    --output ${myset}_${folder}_fusin_${fusin}.jsonl \
    --output_dir ${output_dir}  \
    --seed 2022  \
    --tokenizer $tokenizer_path \
    --num_passage ${num_of_passage} \
    --text_alike ${text_alike}


myset="valid"
text_alike="text_alike_after_search/text_alike_${experiment_run}_taste_${myset}_fusin_${fusin}_all_sizes.jsonl"

python build_train.py  \
    --data_name $set  \
    --sample_num 100 \
    --train_file "${base_dir}/${myset}.txt"  \
    --item_file "${base_dir}/item.txt"  \
    --item_ids_file "${base_dir}/item_${folder}.jsonl"  \
    --output ${myset}_${folder}_fusin_${fusin}.jsonl \
    --output_dir ${output_dir}  \
    --seed 2022  \
    --tokenizer $tokenizer_path \
    --num_passage ${num_of_passage} \
    --text_alike ${text_alike}

