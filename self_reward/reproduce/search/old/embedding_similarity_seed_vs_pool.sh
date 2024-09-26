

experiment_run="sports_iter_2"
pool_embedding="pool_doc_embeddings_no_fusin_decoder/pool_doc_output_embedding_${experiment_run}.jsonl"
seed_embedding="seed_doc_embeddings_no_fusin_decoder/seed_doc_output_embedding_${experiment_run}.jsonl"
output_file="text_alike_${experiment_run}_size_2.jsonl"

python embedding_similarity_seed_vs_pool.py \
	--pool_embedding "${pool_embedding}" \
	--seed_embedding "${seed_embedding}" \
	--output_file "${output_file}"
