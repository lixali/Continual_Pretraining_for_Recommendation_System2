#!/bin/sh


experiment_run="sports"

CUDA_VISIBLE_DEVICES=0 python query_pos_neg_tokens_id_gen.py \
	--text_alike_size_2 "text_alike_${experiment_run}_size_2.jsonl" \
    	--output_dir "data_proposed/${experiment_run}/" \
    	--output_file "data_proposed/${experiment_run}/train_valid_name.jsonl" \
    	--tokenizer "pretrained_model/t5-base"   \
    	--query_token_length 64  \
    	--positive_token_length 512  \
    	--tokenizer_max_length 512  \
    	--total_negative_samples_per_training_record 100  \
    	--random_sampled_negative_documents_file "t5_tokenized_negatives_samples_sharded_total_filtered.jsonl"
