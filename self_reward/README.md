Prerequisite files in this stage:

"pool_doc.jsonl" and "seed_doc.jsonl"


There are steps in this stage:


(1) "seed_doc.jsonl" and "pool_doc.jsonl" is coming from previous stage. We split the "pool_doc.jsonl" into smaller chunks because it is too big. 

Run script "bash split_file.sh"


(2) run command "bash genEmbedding_seed.sh" and "bash genEmbedding_pool.sh"

Need to change the "experiment_run" name to "beauty"/"toys"/"sports" inside the .sh script
and change the corresponding model to generate embedding; each embedding generation needs the best finetuned model to generate

P.S because CPU memory might not be enough (it also uses CPU in the code), even it is running in parallel for pool document generation, also need to be careful whether all 
of 10 splits are generated successfully; if not, make sure to rerun the ones that are not generated

(3) run the command "bash combined_pool_doc_embedding.sh" to combine the pool document embeddings

need to change the "experiment_run" name to "beauty"/"toys"/"sports" inside the .sh script


(4) search similar emebddings and run the following command

"bash embedding_similarity_seed_vs_pool.sh"

Need to change the "experiment_run" name to "beauty"/"toys"/"sports" inside the .sh script

Implement the removal of duplicate positive pairs in here; (A, B) and (B, A) are considered as one positive pair; the removal logic is
using a dictionary and it is implemented in embedding_similarity_seed_vs_pool.py; the text_alike_*jsonl file will not have duplicate pairs
any more



(5) Next is to generate Embedding document seuqnce with lenght = 2. Run the following command

"bash genEmbedding_text_alike_size_2.sh"


(6) And then is run similarity search against the pool document. Run the following command

"bash embedding_similarity_text_alike_vs_pool.sh"
