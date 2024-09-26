

len=2
leng=3
experiment_run="beauty_iter_2"
pool_embedding="pool_doc_embeddings_no_fusin_decoder/pool_doc_output_embedding_${experiment_run}.jsonl"
text_alike_embedding="text_alike_size_${len}_embeddings/text_alike_size_${len}_output_embedding_${experiment_run}.jsonl"
output_file="text_alike_${experiment_run}_size_${leng}.jsonl"

python embedding_similarity_text_alike_vs_pool.py \
	--pool_embedding "${pool_embedding}" \
	--text_alike_embedding "${text_alike_embedding}" \
	--output_file "${output_file}"
