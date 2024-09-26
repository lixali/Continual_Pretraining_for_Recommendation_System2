import json
import argparse
from collections import defaultdict
import sys
import os
from transformers import AutoTokenizer, T5Tokenizer
from tqdm import tqdm
import json


from utils.data_loader import list_split, load_item_address, load_item_name
sys.path.append(os.path.abspath('../TASTE_checkin'))

#from build_train import load_data



def load_data(filename, item_desc):
    data = []
    data_ids = []
    lines = open(filename, "r").readlines()
    users = []
    for line in lines[1:]:
        example = list()
        example2 = list()
        line = line.strip().split("\t")
        target = int(line[-1])  # target item at this step
        seq_id = line[1:-1]  # the history of the user

        ### added by Lixiang ###
        users.append(line[0])
        ### ended added by Lixiang ###

        text_list = []
        # convert the item ids to the item text
        for id in seq_id:
            id = int(id)
            if id == 0:  # get the actions till current step
                break
            text_list.append(item_desc[id])
            example2.append(id)
        text_list.reverse()  # most recent to the oldest
        seq_text = ", ".join(text_list)
        example.append(seq_text)
        example.append(target)
        example2.append(target)
        data.append(example)  # input txt list
        data_ids.append(example2)  # input id list
    return data, data_ids, users



def process_file(input_file,item_file, output_file, args):
    
    
    tokenizer = T5Tokenizer.from_pretrained("pretrained_model/t5-base", local_files_only=True)
    item_desc = load_item_name(item_file)

    train_data, train_data_ids, user_ids = load_data(input_file, item_desc)
    data_num = len(train_data)


    template1 = "Here is the visit history list of user: "
    template2 = " recommend next item "
    t1 = tokenizer.encode(template1, add_special_tokens=False, truncation=False)
    t2 = tokenizer.encode(template2, add_special_tokens=False, truncation=False)
    split_num = args.max_len - len(t1) - len(t2) - 1


    with open(output_file, "w") as f:
        for idx, data in enumerate(tqdm(train_data)):

            query = data[0]
            query = tokenizer.encode(
                query, add_special_tokens=False, padding=False, truncation=False
            )

            query_list = list_split(
                query, split_num
            )  # cut the history into 2 pieces, the first within seq max_len limit

            query_list[0] = t1 + query_list[0] + t2  # first seq fit into template
            if args.num_passage == 1:
                # split into n_passages for t5, otherwise, only keep the first seq that fit into template
                query_list = query_list[:1]

            group = {}
            group["query"] = query_list  # a list of lists of query_ids
            group["item_sequence"] = train_data_ids[idx]
            group["user_id"] = user_ids[idx]


            f.write(json.dumps(group) + "\n")

def main(args):
    # Call the process_file function with the provided arguments
    process_file(args.input_file, args.item_file, args.output_file, args)

if __name__ == "__main__":
    # Setup argparse to handle input and output file arguments
    parser = argparse.ArgumentParser(description='Process a text file and output to jsonl format.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input text file')
    parser.add_argument('--item_file', type=str, required=True, help='Path to the output jsonl file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output jsonl file')
    parser.add_argument('--max_len', type=int, required=True, help='Path to the output jsonl file')
    parser.add_argument('--num_passage', type=int, required=True, help='Path to the output jsonl file')
    
    args = parser.parse_args()
    
    main(args)

