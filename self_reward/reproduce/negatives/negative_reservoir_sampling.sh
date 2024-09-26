#!/bin/bash

conda activate taste_run

for prefix in $(seq -w 01 02); do
    # Inner loop for file numbers
    for num in $(seq -w 01 02); do
        file="/data/datasets/hf_cache/sample/350BT/0${prefix}_000${num}.parquet"
	output_file="negatives_samples_sharded/negatives_0${prefix}_000${num}.jsonl"
	python negative_reservoir_sampling.py --dataset_file "$file" \
		--num_of_negatives 25_000 \
		--output_file "$output_file"

	echo "$file"

    done
done
