import json
import random
import numpy as np
import faiss
import torch
import csv
import time 
import os
import argparse
from collections import defaultdict
# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
start_time = time.time()
start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
print(f"Start time: {start_time_readable}")

def main(args):
    text_alike_embedding = []
    doc_text = defaultdict(list)
    doc_id = defaultdict(list)
    user_ids = []
    
    with open(args.text_alike_embedding, "r") as f:
        for line in f:
            data = json.loads(line)
            text_alike_embedding.append(data["embedding"])
            user_ids.append(data["user_id"])
            for i in range(args.leng):
                doc_text[f"{i+1}_text"].append(data[f"{i+1}_text"])
                doc_id[f"{i+1}_id"].append(data[f"{i+1}_id"])
            
    
    num_of_seed_doc = len(doc_text["1_text"])
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
    text_alike_embedding_narray = np.array(text_alike_embedding)
    
    #breakpoint()
    # FAISS indexing and searching
    faiss.omp_set_num_threads(128)
    cpu_index = faiss.IndexFlatIP(text_alike_embedding_narray.shape[1])
    cpu_index.add(pool_embedding_narray)
    
    query_embeds_narray = text_alike_embedding_narray.astype(np.float32)
    K = args.Ks
    
    ### D and I are numpy array list type; D is the dot product value; I is the index of the position of matching doc in pool_doc
    D, I = cpu_index.search(query_embeds_narray, K)
    output_data = []
    

    nearest_neighbour = {}
    for i in range(num_of_seed_doc):
        # The first index will be the text itself, so we take the second closest
        curr_dict = {}
        if i % 10000 == 0: print(f"{i} processed")
        curr_dict["user_id"] = user_ids[i]
        for j in range(K):
    
            Index_picked = I[i][j] ## Index_picked is the index in the pool_text that is picked to be close to the current seed document
            if all(pool_id[Index_picked] != doc_id[f"{x+1}_id"][i] for x in range(args.leng)):
               

                for m in range(args.leng):
                    curr_dict[f"{m+1}_text"] = doc_text[f"{m+1}_text"][i]
                    curr_dict[f"{m+1}_id"] = doc_id[f"{m+1}_id"][i]

                curr_dict[f"{args.leng+1}_text"] = pool_text[Index_picked]
                curr_dict[f"{args.leng+1}_id"] = pool_id[Index_picked]
                output_data.append(curr_dict)
                
                    
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
    parser.add_argument('--text_alike_embedding', type=str, required=True, help="seed embedding")
    parser.add_argument('--output_file', type=str, required=True, help="output file")
    parser.add_argument('--leng', type=int, required=True, help="output file")
    parser.add_argument('--Ks', type=int, required=True, help="top K it is looking")
    args = parser.parse_args()
    main(args)
