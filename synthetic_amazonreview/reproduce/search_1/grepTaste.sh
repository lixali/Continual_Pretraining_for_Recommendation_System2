#!/bin/bash

run="toys"
experiment_run="toys_from_Jingyuan"
fusin=0
set="train"
#set="valid"
INPUT_FILE="data/${run}/${set}.txt"
OUTPUT_FILE="text_alike_after_search/text_alike_${experiment_run}_taste_${set}_fusin_${fusin}.jsonl"
ITEM_FILE="data/${run}/item.txt"
max_len=512
num_passage=1

python grepTaste.py  \
	--input_file "${INPUT_FILE}" \
	--item_file "${ITEM_FILE}" \
	--output_file "${OUTPUT_FILE}" \
	--max_len "${max_len}" \
	--num_passage "${num_passage}"


set="valid"
INPUT_FILE="data/${run}/${set}.txt"
OUTPUT_FILE="text_alike_after_search/text_alike_${experiment_run}_taste_${set}_fusin_${fusin}.jsonl"
python grepTaste.py  \
	--input_file "${INPUT_FILE}" \
	--item_file "${ITEM_FILE}" \
	--output_file "${OUTPUT_FILE}" \
	--max_len "${max_len}" \
	--num_passage "${num_passage}"

