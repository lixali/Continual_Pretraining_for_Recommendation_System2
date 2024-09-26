

len=1
fusin=0
myset="train"
run="toys"
experiment_run="toys_from_Jingyuan"
pool_embedding="data/${run}/pool_item_embedding_t5_${experiment_run}_fusin_${fusin}.jsonl"
text_alike_embedding="text_alike_embeddings/text_alike_taste_${myset}_output_embedding_${experiment_run}_fusin_${fusin}.jsonl"
output_file="text_alike_after_search/text_alike_${experiment_run}_taste_${myset}_fusin_${fusin}_all_sizes.jsonl"

python embedding_similarity_taste.py \
	--pool_embedding "${pool_embedding}" \
	--text_alike_embedding "${text_alike_embedding}" \
	--output_file "${output_file}" \
	--leng "${len}" \
        --Ks 6



wait
myset="valid"
text_alike_embedding="text_alike_embeddings/text_alike_taste_${myset}_output_embedding_${experiment_run}_fusin_${fusin}.jsonl"
output_file="text_alike_after_search/text_alike_${experiment_run}_taste_${myset}_fusin_${fusin}_all_sizes.jsonl"

python embedding_similarity_taste.py \
        --pool_embedding "${pool_embedding}" \
        --text_alike_embedding "${text_alike_embedding}" \
        --output_file "${output_file}" \
        --leng "${len}" \
        --Ks 6
