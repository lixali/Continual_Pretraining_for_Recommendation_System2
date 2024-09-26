import random
from datasets import load_dataset
import time
import json
import argparse


start_time = time.time()
start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
print(f"Start time: {start_time_readable}")



#dataset_path = "/data/datasets/hf_cache/sample/10BT/"

def reservoir_sample(iterable, k):
    # Initialize a reservoir with the first k elements
    reservoir_text = []
    reservoir_id = []
    count = 0
    for i, element in enumerate(iterable):
        count += 1

        if i < k:
            reservoir_text.append(element["text"])
            reservoir_id.append(element["id"])
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir_text[j] = element["text"]
                reservoir_id[j] = element["id"]
        if count % 100_000 == 0: print(f"count is {count}")
    return reservoir_text, reservoir_id



def main(args):
    #dataset = load_dataset(dataset_path, streaming=True)
    dataset = load_dataset('parquet', data_files=args.dataset_file, streaming=True)
    num_of_negatives = args.num_of_negatives
    
    sampled_text, sampled_id = reservoir_sample(dataset["train"], num_of_negatives)
    
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
    parser.add_argument('--dataset_file', type=str, required=True, help="dataset parquet file name")
    parser.add_argument('--num_of_negatives', type=int, required=True, help="numebr of negatives in this parquet")
    parser.add_argument('--output_file', type=str, required=True, help="output file")
    args = parser.parse_args()    
    main(args)

