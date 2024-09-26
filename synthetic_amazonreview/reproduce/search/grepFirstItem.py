import json
import argparse
from collections import defaultdict

from utils.data_loader import list_split, load_item_address, load_item_name


def process_file(input_path, output_path):
    user_seq_dict = defaultdict(list)
   

    item_desc = load_item_name(args.item_file)
    first_item_set = set()
    # Read the file and populate the dictionary
    with open(input_path, 'r') as file:
        next(file)  # Skip the header
        for line in file:
            curr_data = line.split()
            user_id = int(curr_data[0])

            first_item = int(curr_data[1])
            # if user_id not in user_seq_dict or first_item != user_seq_dict[user_id][-1]: 
            if user_id not in user_seq_dict:
                user_seq_dict[user_id].append(first_item)
    # Write the output to a jsonl file
    with open(output_path, 'w') as output_file:
        for user_id, first_items in user_seq_dict.items():


            # it is possible to have two or more different first items, make a for loop to use all of them

            
            for first_item in first_items:

                if first_item not in first_item_set:
                    first_item_set.add(first_item)
                    data = {
                        "user_id": user_id,
                        "1_id": first_item,
                        "1_text": item_desc[first_item]
                    }
                
                # Write the JSON object as a line to the jsonl file
                    output_file.write(json.dumps(data) + '\n')
    
def main(args):
    # Call the process_file function with the provided arguments
    process_file(args.input_file, args.output_file)

if __name__ == "__main__":
    # Setup argparse to handle input and output file arguments
    parser = argparse.ArgumentParser(description='Process a text file and output to jsonl format.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input text file')
    parser.add_argument('--item_file', type=str, required=True, help='Path to the output jsonl file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output jsonl file')
    
    args = parser.parse_args()
    
    main(args)

