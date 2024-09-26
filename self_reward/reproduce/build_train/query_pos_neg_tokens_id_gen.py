import pickle
import json
from transformers import AutoTokenizer

import logging
import os
import argparse
import time
start_time = time.time()
start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
print(f"Start time: {start_time_readable}")


def main(args):


    tokenizer_max_length=args.tokenizer_max_length
    total_negative_samples_per_training_record=args.total_negative_samples_per_training_record
    random_sampled_negative_documents_file=args.random_sampled_negative_documents_file
    output_dir=args.output_dir
    text_alike_size_2=args.text_alike_size_2

    # Load the original data

    # Set up logging configuration
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(output_dir,'train_valid_name.log'), filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)


    # Log the important variable
    logger.info(f'tokenizer_max_length is {tokenizer_max_length}')
    logger.info(f'tokenizer is {args.tokenizer}')
    logger.info(f'total_negative_samples_per_training_record is {total_negative_samples_per_training_record}')
    logger.info(f'output_dir is {output_dir}')
    logger.info(f'text_alike_size_2 is {text_alike_size_2}')




    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    
    def tokenize_document(doc, tokenizer, max_length=tokenizer_max_length): ### t5 base is 512, 
        return tokenizer.encode(doc, max_length=max_length, add_special_tokens=False, padding=False, truncation=True) ## double checked on it, it should be truncation=True
   
    prefix = "Here is the document: "
    suffix = " recommend next document "
    
    prefix_tokens = tokenize_document(prefix, tokenizer, 512)
    suffix_tokens = tokenize_document(suffix, tokenizer, 512)
    prefix_token_length = len(prefix_tokens)
    suffix_token_length = len(suffix_tokens)

    query_text_length = args.query_token_length - prefix_token_length - suffix_token_length 
   
    with open(text_alike_size_2, 'r') as f: text_alike_size_2_data = [json.loads(line) for line in f]
    
    with open(random_sampled_negative_documents_file, 'r') as f: random_sampled_negative_documents_tokenized_ID = [json.loads(line)["tokenized_id"][:512] for line in f]

    # Process each row
    processed_data = []
    for i in range(len(text_alike_size_2_data)):

        if i % 1000 == 0: print(f"processed {i} rows now")
        row = text_alike_size_2_data[i]
        
        query_ids = [prefix_tokens + tokenize_document(row['seed_text'], tokenizer, query_text_length) + suffix_tokens]
        positive_ids = [tokenize_document(row['1st_nearest_text'], tokenizer, args.positive_token_length)]
        negatives_ids = random_sampled_negative_documents_tokenized_ID[i*total_negative_samples_per_training_record:(i+1)*total_negative_samples_per_training_record] ## negatives have been tokenized in previous steps, I am just reading it here
        
        processed_row = {
            "query": query_ids,
            "positives": positive_ids,
            "negatives": negatives_ids
        }
        
        processed_data.append(processed_row)

    # Save to a JSONL file
    count = 0
    with open(args.output_file, 'w') as f:
        for row in processed_data:
            count += 1

            if count % 1000 == 0: print(f"{count} lines have been written to file")
            json.dump(row, f)
            f.write('\n')

    logger.info("Processing and saving complete!")

    end_time = time.time()
    end_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print(f"End time: {end_time_readable}")
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken: {int(minutes)} minutes and {seconds:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_alike_size_2', type=str, required=True, help="a file that contains the closest text for its corresponding doc")
    parser.add_argument('--output_dir', type=str, required=True, help="output directory to save the results")
    parser.add_argument('--output_file', type=str, required=True, help="output directory to save the results")
    parser.add_argument('--tokenizer', type=str, required=True, help="tokenizer")
    parser.add_argument('--query_token_length', type=int, required=True, help="tokenizer max length")
    parser.add_argument('--positive_token_length', type=int, required=True, help="tokenizer max length")
    parser.add_argument('--tokenizer_max_length', type=int, required=True, help="tokenizer max length")
    parser.add_argument('--total_negative_samples_per_training_record', type=int, required=True, help="Directory to save the tensorboard log results")
    parser.add_argument('--random_sampled_negative_documents_file', type=str, required=True, help="pythia or t5 or others")
    args = parser.parse_args()
    main(args)
