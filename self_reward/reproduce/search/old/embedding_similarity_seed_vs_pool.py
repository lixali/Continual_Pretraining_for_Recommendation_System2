import json
import random
import numpy as np
import faiss
import torch
import csv
import time 
import os
import argparse
# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
start_time = time.time()
start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
print(f"Start time: {start_time_readable}")

def main(args):
    seed_embedding = []
    seed_text = []
    seed_id = []
    with open(args.seed_embedding, "r") as f:
        for line in f:
            data = json.loads(line)
            seed_embedding.append(data["embedding"])
            seed_text.append(data["text"])
            seed_id.append(data["id"])
    
    num_of_seed_doc = len(seed_text)
    pool_embedding = []
    pool_text = []
    pool_id = []
    #seed_text_set = set(seed_text)
    line_count = 0
    with open(args.pool_embedding, "r") as f:
        for line in f:
            line_count += 1
            data = json.loads(line)
            pool_embedding.append(data["embedding"])
            pool_text.append(data["text"])
            pool_id.append(data["id"])
            if line_count % 100000 == 0: print(f"{line_count} in pool_doc read into python list")
    #pool_minus_seed = [text for text in pool_text if text not in seed_text_set]
    
    #pool_embedding_narray = np.array(pool_embedding, dtype=np.float32)
    pool_embedding_narray = np.array(pool_embedding)
    seed_embedding_narray = np.array(seed_embedding)
    
    #breakpoint()
    # FAISS indexing and searching
    faiss.omp_set_num_threads(128)
    cpu_index = faiss.IndexFlatIP(seed_embedding_narray.shape[1])
    cpu_index.add(pool_embedding_narray)
    
    query_embeds_narray = seed_embedding_narray.astype(np.float32)
    K = 4  
    
    ### D and I are numpy array list type; D is the dot product value; I is the index of the position of matching doc in pool_doc
    D, I = cpu_index.search(query_embeds_narray, K)
    output_data = []
    

    nearest_neighbour = {}
    for i in range(num_of_seed_doc):
        # The first index will be the text itself, so we take the second closest
        if i % 10000 == 0: print(f"{i} processed")
    
        for j in range(K):
    
            Index_picked = I[i][j] ## Index_picked is the index in the pool_text that is picked to be close to the current seed document
            if seed_id[i] != pool_id[Index_picked]: # we want the cloest to exclude itself 
               



                # can not have duplicate pairs (A, B) is same as (B, A), record the pair in dictionary nearest_neighbour
                if pool_id[Index_picked] in nearest_neighbour and nearest_neighbour[pool_id[Index_picked]] == seed_id[i]: break
                
                nearest_neighbour[seed_id[i]] = pool_id[Index_picked]

                output_data.append({
                    'seed_text': seed_text[i],
                    'seed_id': seed_id[i],
                    'seed_embedding': seed_embedding[i],
                    '1st_nearest_text': pool_text[Index_picked],
                    '1st_nearest_id': pool_id[Index_picked],
                    '1st_nearest_embedding': pool_embedding[Index_picked]
                })
    
                break
        
    count = 0 
    with open(args.output_file, 'w') as jsonl_file:
        for item in output_data:
            count += 1

            if count % 1000 == 0: print(f"write {count} lines into the output file now")
            jsonl_file.write(json.dumps(item) + '\n')
    
    
    end_time = time.time()
    end_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print(f"End time: {end_time_readable}")
    
    elapsed_time = end_time - start_time
    
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken: {int(minutes)} minutes and {seconds:.2f} seconds")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool_embedding', type=str, required=True, help="pool embedding")
    parser.add_argument('--seed_embedding', type=str, required=True, help="seed embedding")
    parser.add_argument('--output_file', type=str, required=True, help="output file")
    args = parser.parse_args()
    main(args)
