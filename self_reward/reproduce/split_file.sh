#!/bin/bash


folders=("search/pool_doc_splitted_no_fusin_decoder")

for folder in "${folders[@]}"; do
  if [ ! -d "$folder" ]; then
    mkdir "$folder"
    echo "Folder created: $folder"
  else
    echo "Folder already exists: $folder"
  fi
done


total_lines=$(wc -l < pool_doc.jsonl)
lines_per_file=$((total_lines / 10))

split -l $lines_per_file pool_doc.jsonl search/pool_doc_splitted_no_fusin_decoder/output_prefix_
