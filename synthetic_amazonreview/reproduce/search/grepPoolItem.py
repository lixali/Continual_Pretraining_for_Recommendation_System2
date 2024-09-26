import json
import argparse
from collections import defaultdict
import numpy as np
from utils.data_loader import list_split, load_item_address, load_item_name


def process_file(input_path, output_path):
    user_seq_dict = defaultdict(list)
   

    item_desc = load_item_name(args.item_file)
    
    
    with open(output_path, 'w') as output_file:
        for pool_id, pool_item in item_desc.items():

            data = {
                "id": int(pool_id),
                "text": pool_item
            }
            
            # Write the JSON object as a line to the jsonl file
            output_file.write(json.dumps(data) + '\n')

def main(args):
    # Call the process_file function with the provided arguments
    process_file(args.item_file, args.output_file)

if __name__ == "__main__":
    # Setup argparse to handle input and output file arguments
    parser = argparse.ArgumentParser(description='Process a text file and output to jsonl format.')
    parser.add_argument('--item_file', type=str, required=True, help='Path to the output jsonl file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output jsonl file')
    
    args = parser.parse_args()
    
    main(args)

