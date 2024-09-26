
import json
from transformers import AutoTokenizer
import argparse
import time 
start_time = time.time()
start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
print(f"Start time: {start_time_readable}")

def tokenize_document(doc, tokenizer, tokenizer_max_length=512):
    return tokenizer.encode(doc, max_length=tokenizer_max_length, add_special_tokens=False, padding=False,  truncation=False)

def main(args):
    
    data = []
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    count = 0
    with open(args.input_file, "r") as f:
        for line in f:
            
            count += 1
            tokenized_id = tokenize_document(json.loads(line)["text"], tokenizer, args.max_token_length)
            
            curr = {
                "text": json.loads(line)["text"],
                "id": json.loads(line)["id"],
                "tokenized_id": tokenized_id
                    
            }

            if count % 10_000 == 0: print(f"tokneized {count} lines now")


            data.append(curr)



    with open(args.output_file, "w") as f:

        for record in data:
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
    parser.add_argument('--input_file', type=str, required=True, help="input json file")
    parser.add_argument('--tokenizer', type=str, required=True, help="tokenizer local file")
    parser.add_argument('--max_token_length', type=int, required=True, help="max token length")
    parser.add_argument('--output_file', type=str, required=True, help="output file")
    args = parser.parse_args()
    main(args)


            
        
        
        



tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
