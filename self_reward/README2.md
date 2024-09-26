This is after running TASTE_flavour_stage_2.

This is to generate the jsonl files that are input to the TASTE_checkin to do contrastive leanring for pythia


Following are the steps:

(1) randomly sample from sample-350BT using reservoir sampling

Run commmand "bash negative_reservoir_sampling.sh"

This might run for long time. And the sample-350BT are sharded into over 500 parquet files in the begging, 
each sharded parqeut file will be processed concurrently 



(2) generate tokenized ID for negative samples

After the sharded parquet files are sampled into jsonl files, these jsonl files will be used to generate tokens.
There are over 500 jsonl files, they will be processed concurrently 

Run command "bash t5_negative_samples_tokenized_ID_gen.sh"


(3) combine all the small tokenzied ID jsonl files together into a larger file

Run command "bash combine_all_tokeinzed_negatives_jsonl.sh "

(4) remove the overlap of sampled negatives and pool document
(we don't want any pool document to be in the negatives)

Run command "bash remove_pool_doc_overalp_for_negatives.sh"



(5) generate tokenzied ID for query and positive sample

For query, adding following prefix and suffix so that the Prefix's first token's embedding can be used

Prefix: "Here is the document: "

Suffix: " recommand next document "


Prefix has 5 tokens, suffix has 3 tokens. In total, there are 8 tokens
And then 512 - 8 - 1 = 503 is the max lenght for the real text
why minus 1? This is what TASTE framework does. Not sure why. 

Run command "bash t5_query_pos_neg_tokens_id_gen.sh"

In this stage, need to modify .sh file and change "$experiment_run" to "beauty"/"toys"/"sports"; they all share the same script

(6) Split the generated dataset into train and validation split 

Run command "bash split_train_valid.sh"

In this stage, need to modify .sh file and change "$experiment_run" to "beauty"/"toys"/"sports"; they all share the same split_train_valid.sh script
