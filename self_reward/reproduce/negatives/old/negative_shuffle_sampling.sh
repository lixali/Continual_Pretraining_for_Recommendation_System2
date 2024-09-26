#!/bin/bash

conda activate taste_run


dataset_path="/data/datasets/hf_cache/sample/350BT/"
output_file="negatives_shuffle_samples/negative_13_millions.jsonl"
python negative_shuffle_sampling.py --dataset_path "$dataset_path" \
	--num_of_negatives 13_000_000 \
	--output_file "$output_file"



echo "$file"
