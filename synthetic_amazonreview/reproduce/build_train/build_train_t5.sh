#!/bin/sh


# T5 Family
model="t5-base"
experiment_run="sports"

num_of_passage=2
fusin=1

set="Amazon"
folder="name"

base_dir="data/${experiment_run}"
tokenizer_path="pretrained_model/${model}"
output_dir="data_synthetic/${experiment_run}/"

echo $base_dir

python build_train.py  \
    --data_name $set  \
    --sample_num 100 \
    --train_file "text_alike_after_search/${experiment_run}_combined_2_to_14_fusin_${fusin}.jsonl"  \
    --item_file "${base_dir}/item.txt"  \
    --item_ids_file "${base_dir}/item_${folder}.jsonl"  \
    --output train_${folder}_fusin_${fusin}.jsonl  \
    --output_dir ${output_dir}  \
    --seed 2022  \
    --tokenizer $tokenizer_path \
    --num_passage ${num_of_passage} \


python build_train.py  \
    --data_name $set  \
    --sample_num 100 \
    --train_file "text_alike_after_search/text_alike_${experiment_run}_size_15_fusin_${fusin}.jsonl"  \
    --item_file "${base_dir}/item.txt"  \
    --item_ids_file "${base_dir}/item_${folder}.jsonl"  \
    --output valid_${folder}_fusin_${fusin}.jsonl  \
    --output_dir ${output_dir}  \
    --seed 2022  \
    --tokenizer $tokenizer_path \
    --num_passage ${num_of_passage} \

