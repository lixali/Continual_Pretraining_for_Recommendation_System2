
(1) generate the seed items (text)

Run "bash grepFirstItem.sh"

Output is in text_aliek_after_search/text_alike_toys_1.jsonl

And the name product (even in the original TASTE dataset will appear difference times with difference users) will only appears once

(2) generate the pool items (text)

Run "bash grepPoolItem.sh"


(3) Convert pool document into embeddings

Run "bash genEmbedding_pool.sh"



(4) Convert seed items/text_alike (text) into embedding

Run "bash genEmbedding_seed_and_alike.sh"

It has a for loop to loop the length from 1 to any length
 
- Need to the experiment_run to toys/sports/beauty

The loop will also do similarity search and generate the text_alike*jsonl files in the text_alike_after_search folder

(5) combine the text_alike*jsonl files together

Run "bash combine_text_alike.sh"

- Need to change the experiment_run variable name
- Need to change the length 2, 3, .... 15... (everyone of them; it starts at 2 because the sequence length is at minimum 2)









#################################################################################################################


Following is a second work flow that takes the same input of TASTE and only generate one output label 

- Different from above work flow because there is no sequential generation over and over again with different sequence length


(1) Get all the sequence input from TASTE 

- run "bash grepTaste.sh"
- Need to change script to do run for "beauty"/"toys"/"sports" seperately (change the experiment_run argument to "beauty" or "toys" or "sports" in the .sh file)
- Need to do for both train.txt and valid.txt from TASTE (change the set argument to "train" or "valid" in the .sh file)
 
(2) Generate embedding for the sequence and TASTE pool (using TATSTE checkpoint model to generate sequence) 

- run "bash genEmbedding_taste.sh"
- Need to change script to do run for "beauty"/"toys"/"sports" seperately (change the experiment_run argument to "beauty" or "toys" or "sports" in the .sh file)
- Need to do for both train.txt and valid.txt from TASTE (change the set argument to "train" or "valid" in the .sh file)
- This step will run also generate embedding for TASTE pool (beauty/toys/sports). I need to run have data/beauty/pool_item.jsonl file ready (data/beauty/pool_item.jsonl is generated by previous workflow in step 2 "bash grepPoolItem.sh")

 

(3) Run embedding similarity search against the AmazonReview beauty/toys/sports to find the cloest item (this pool is generated in above work flow step 2)

- run "bash embedding_similarity_taste.sh" 
- Need to change script to do run for "beauty"/"toys"/"sports" seperately (change the experiment_run argument to "beauty" or "toys" or "sports" in the .sh file)
- Need to do for both train.txt and valid.txt from TASTE (change the set argument to "train" or "valid" in the .sh file)
 
(4) Build training and validatin data (including generating negatives)

- run "bash build_train_t5.sh" (this "build_train_t5.sh" is basically copied from TASTE, the only added code is that the postive needs to be replaced by the postive generated in step (3))
- Need to change script to do run for "beauty"/"toys"/"sports" seperately (change the experiment_run argument to "beauty" or "toys" or "sports" in the .sh file)
- Need to do for both train.txt and valid.txt from TASTE (change the set argument to "train" or "valid" in the .sh file)
 

P.S In this work flow, there is no 2.5 stage (unlike the work flow above)
