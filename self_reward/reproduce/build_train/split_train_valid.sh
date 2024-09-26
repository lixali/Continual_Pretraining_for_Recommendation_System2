#!/bin/bash

experiment_run="beauty"
input_file="data_proposed/${experiment_run}/train_valid_name.jsonl"
train_file="data_proposed/${experiment_run}/train_name.jsonl"
validation_file="data_proposed/${experiment_run}/valid_name.jsonl"


total_lines=$(wc -l < "$input_file")
num_of_validation=$(( (total_lines * 15 + 99) / 100 ))

num_of_train=$((total_lines - num_of_validation))

# Extract the lines and write to the new file
head -n $num_of_train "$input_file" > "$train_file"
echo "Extraction complete. First $num_of_train lines written to $train_file"

tail -n $num_of_validation "$input_file" > "$validation_file"
echo "Last $num_of_validation lines extracted to $validation_file_name"
