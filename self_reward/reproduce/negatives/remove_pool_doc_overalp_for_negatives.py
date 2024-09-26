import json
import argparse
import time
start_time = time.time()
start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
print(f"Start time: {start_time_readable}")
def main(args):
    pool_doc_id = []
    with open(args.pool_doc, "r") as f:
        pool_doc_id = [json.loads(line)["id"] for line in f]
    
    pool_doc_id_set = set(pool_doc_id)
    
    
    with open(args.negatives_unfiltered, "r") as f:
        filtered_negatives = [json.loads(line) for line in f if json.loads(line)["id"] not in pool_doc_id_set]
    
    count = 0
    with open(args.output_file, "w") as f:
        for record in filtered_negatives:
            count += 1
            if count % 100_000 == 0: print(f"process ${count} negatives")
            json.dump(record, f)
            f.write("\n")
    end_time = time.time()
    end_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print(f"End time: {end_time_readable}")
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken: {int(minutes)} minutes and {seconds:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--negatives_unfiltered', type=str, required=True, help="negatives with overlaps")
    parser.add_argument('--pool_doc', type=str, required=True, help="pool documents")
    parser.add_argument('--output_file', type=str, required=True, help="output file")
    args = parser.parse_args()    
    main(args)
