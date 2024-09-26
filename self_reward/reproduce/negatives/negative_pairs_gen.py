import pandas as pd
import numpy as np
import random
import pickle
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import logging
import argparse


def main(args):

    dataset_name = args.dataset_name
    dataset = load_dataset("/data/datasets/hf_cache/sample/350BT", streaming=True)
    count = 0    
    for sample in dataset["train"]:
        count += 1
        #print(sample["text"])
        if count % 10000000 == 0: 
            print(count)
            breakpoint()
    breakpoint()
    num_of_negatives_to_generate = args.num_of_negatives_to_generate ## I want to generate 1900000 negative samples; the more, the better, this number is larger than total number of negative samples needed
    model = args.model
    output_dir=args.output_dir
    tokenizer_max_length=args.tokenizer_max_length
    total_negative_samples_per_training_record=args.total_negative_samples_per_training_record

    # Set up logging configuration
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(output_dir,'negative_pair_gen.log'), filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')


    # Create a logger object
    logger = logging.getLogger(__name__)

    # Example variable to log

    # Log the important variable
    logger.info(f'model is {model}')
    logger.info(f'tokenizer_max_length is {tokenizer_max_length}')
    logger.info(f'output_dir is {output_dir}')
    logger.info(f'dataset_name is {dataset_name}')
    logger.info(f'total_negative_samples_per_training_record is {total_negative_samples_per_training_record}')

    # Randomly sample
    sampled_dataset = dataset['train'].shuffle(seed=42).select(range(num_of_negatives_to_generate))

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Tokenize documents with a max token length 
    def tokenize_document(doc, tokenizer, max_length=tokenizer_max_length):
        return tokenizer.encode(doc, max_length=tokenizer_max_length, truncation=True)

    tokenized_documents = [tokenize_document(doc['text'], tokenizer) for doc in sampled_dataset]
    documents_text = [doc['text'] for doc in sampled_dataset]

    # Organize into rows of 100 documents each
    tokenized_rows = [tokenized_documents[i:i+total_negative_samples_per_training_record] for i in range(0, len(tokenized_documents), total_negative_samples_per_training_record)]

    # Save to a pickle file
    with open(f"{model}_random_sampled_tokenized_documents_{num_of_negatives_to_generate}.pkl", 'wb') as f:
        pickle.dump(tokenized_rows, f)

    with open(f"{model}_random_sampled_documents_text_{num_of_negatives_to_generate}.pkl", 'wb') as f:
        pickle.dump(documents_text, f)

    logger.info("Tokenization and saving complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help="dataset name, e.g sample-350BT or sample-10BT")
    parser.add_argument('--num_of_negatives_to_generate', type=int, required=True, help="number of negatives in total to sample from fineweb")
    parser.add_argument('--model', type=str, required=True, help="pythia or t5 or others")
    parser.add_argument('--output_dir', type=str, required=True, help="the output path for generated sample negative document files; for both raw text and tokenzied doucment")
    parser.add_argument('--tokenizer_max_length', type=int, required=True, help="tokenizer max length")
    parser.add_argument('--total_negative_samples_per_training_record', type=int, required=True, help="Directory to save the tensorboard log results")
    args = parser.parse_args()
    main(args)
