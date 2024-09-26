

for i in $(seq -w 16 16); do
	for j in $(seq -w 10 13); do
		file="negatives_samples_sharded/negatives_0${i}_000${j}.jsonl"
		output_file="t5_tokenzied_${file}"
		python negative_samples_tokenized_ID_gen.py \
			--input_file "$file" \
			--tokenizer pretrained_model/t5-base \
			--max_token_length 512 \
			--output_file "$output_file" &

echo "$file"
echo "$output_file"
done
done
