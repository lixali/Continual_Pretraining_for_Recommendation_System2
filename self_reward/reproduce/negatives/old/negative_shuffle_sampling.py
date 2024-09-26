import random
from datasets import load_dataset
import time
import json
import argparse


start_time = time.time()
start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
print(f"Start time: {start_time_readable}")



#dataset_path = "/data/datasets/hf_cache/sample/10BT/"

def shuffle_sample(iterable, k):
    # Initialize a shuffle with the first k elements
    shuffle_text = []
    shuffle_id = []
    count = 0
    for i, element in enumerate(iterable):
        count += 1
        
        shuffle_text.append(element["text"])
        shuffle_id.append(element["id"])
        if count % 100_000 == 0: print(f"count is {count}")
        
        if count == k: return shuffle_text, shuffle_id



def main(args):
    dataset = load_dataset(args.dataset_path, streaming=True)
    num_of_negatives = args.num_of_negatives
     
    sampled_text, sampled_id = shuffle_sample(dataset["train"].shuffle(seed=42), num_of_negatives)
    
    with open(args.output_file, 'w') as f:
        for text, id in zip(sampled_text, sampled_id):
            json_line = json.dumps({'text': text, 'id': id})
            f.write(json_line + '\n')


    # 
    # pool_doc_id = []
    # with open("pool_doc.jsonl", "r") as f:
    #     pool_doc_id = [json.loads(line)['id'] for line in f]
    # 
    # pool_doc_id_set = set(pool_doc_id)
    # 
    end_time = time.time()
    end_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print(f"End time: {end_time_readable}")
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken: {int(minutes)} minutes and {seconds:.2f} seconds")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help="dataset parquet file name")
    parser.add_argument('--num_of_negatives', type=int, required=True, help="numebr of negatives in this parquet")
    parser.add_argument('--output_file', type=str, required=True, help="output file")
    args = parser.parse_args()    
    main(args)

