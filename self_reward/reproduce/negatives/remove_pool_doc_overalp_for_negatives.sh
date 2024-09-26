#!/bin/bash

python remove_pool_doc_overalp_for_negatives.py \
	--negatives_unfiltered "t5_tokenized_negatives_samples_sharded_total.jsonl" \
	--pool_doc "pool_doc.jsonl" \
	--output_file "t5_tokenized_negatives_samples_sharded_total_filtered.jsonl" \

