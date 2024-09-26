import sys

sys.path.append("/data1/meisen/TASTE-main")
import json
import os.path
from argparse import ArgumentParser

import jsonlines
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, T5Tokenizer

from utils.data_loader import list_split, load_item_address, load_item_name


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_name", type=str, default="Amazon", help="choose Amazon or yelp"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="/data1/meisen/TASTE-main/Data/beauty/valid.txt",
        help="Path of the train/valid.txt file",
    )
    parser.add_argument(
        "--item_file",
        type=str,
        default="/data1/meisen/TASTE-main/Data/beauty/item.txt",
        help="Path of the item.txt file",
    )
    parser.add_argument(
        "--item_ids_file",
        type=str,
        default="/data1/meisen/TASTE-main/Data/beauty/item_name.jsonl",
        help="Path of the item token file",
    )
    parser.add_argument("--output", type=str, default="valid_name.jsonl")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data1/meisen/TASTE-main/Data/beauty",
        help="Output data path.",
    )
    parser.add_argument(
        "--split_num",
        type=int,
        default=243,
        help="token num of seq text without prompt, total num equals to 256",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=256,
        help="token num of seq text allowed for the model",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=100,
        help="the sample num of random negatives ",  # might need to adjust for ml-1m -> might have out of bounds issur for items (3417)
    )
    parser.add_argument(
        "--num_passage",
        type=int,
        default=1,
        help="use fusin decoder or not",  # might need to adjust for ml-1m -> might have out of bounds issur for items (3417)
    )

    parser.add_argument("--seed", type=int, default=2022, help="random seed")
    parser.add_argument(
        "--tokenizer", type=str, default="pretrained_model/t5-base"
    )
    parser.add_argument("--t5", action="store_true")
    args = parser.parse_args()
    return args


def load_item_input_ids(filename):
    item_input_ids_dict = dict()
    with open(filename, "r", encoding="utf-8") as f:
        for example in jsonlines.Reader(f):
            id = example["id"]
            item_ids = example["item_ids"]
            item_input_ids_dict[id] = item_ids
    return item_input_ids_dict


def load_data(filename, item_desc, args):
    data = []
    data_ids = []

    with open(filename, "r") as file:

        for line in file:
            curr_data = []
            s = ""
            row = json.loads(line)
            dict_len = len(row)
            sequence_length = (dict_len - 1) // 2 ## total length of each row in jsonl , minus the "user_id", divided by 2 because of pairs such as "1_text" "1_id", "2_text", "2_id"
            for i in range(sequence_length-2, -1, -1):
                k = i + 1

                s = row[f"{k}_text"] if not s else s + ", " + row[f"{k}_text"] 
            
            curr_data.append(s)
            curr_data.append(row[f"{k}_id"])

            curr_ids = []

            for i in range(sequence_length):
                k = i + 1
                curr_ids.append(row[f"{k}_id"])
    
            data.append(curr_data)
            data_ids.append(curr_ids)
    return data, data_ids


def load_random_neagtive_items(args, item_num, data_num, train_data_ids):
    np.random.seed(args.seed)
    negative_samples = {}
    print(
        "going to sample: %d negatives for each training datapoint"
        % int(args.sample_num)
    )
    for i in range(data_num):
        samples = []
        for _ in range(args.sample_num):
            item = np.random.choice(item_num) + 1  # one-indexing
            while (
                item in train_data_ids[i]
                or item in samples
                or item == item_num  # extra check for keyerror in small datasets
            ):  # hash to the next one
                item = np.random.choice(item_num) + 1
                if len(train_data_ids[i]) + len(samples) == item_num:  # saturated
                    breakpoint()
            samples.append(item)
        negative_samples[i] = samples
    print("length of negative samples is %d" % len(negative_samples))
    print("sample of the first data point: %d" % len(negative_samples[0]))
    return negative_samples


def main():
    args = get_args()
    if args.t5:
        print("T5 tokenizer..")
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    else:
        print("Fast tokenizer..")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    item_input_ids_dict = load_item_input_ids(args.item_ids_file)
    item_num = len(item_input_ids_dict)
    print("item num is %d" % item_num)
    # We only use the title attribute in the Amazon dataset, and the title and address attributes in the Yelp dataset.
    if args.data_name == "Amazon":
        item_desc = load_item_name(args.item_file)
    elif args.data_name == "yelp":
        item_desc = load_item_address(args.item_file)
    train_data, train_data_ids = load_data(args.train_file, item_desc, args)
    
    #breakpoint()
    data_num = len(train_data)
    print("data num is %d" % data_num)
    random_neg_dict = load_random_neagtive_items(
        args, item_num, data_num, train_data_ids
    )
    
    #breakpoint()
    output_file = os.path.join(args.output_dir, args.output)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    template1 = "Here is the visit history list of user: "
    template2 = " recommend next item "
    t1 = tokenizer.encode(template1, add_special_tokens=False, truncation=False)
    t2 = tokenizer.encode(template2, add_special_tokens=False, truncation=False)
    split_num = args.max_len - len(t1) - len(t2) - 1
    # all "query" are the interaction history in text
    with open(output_file, "w") as f:
        for idx, data in enumerate(tqdm(train_data)):
            pos_list = []
            neg_list = []
            query = data[0]
            query = tokenizer.encode(
                query, add_special_tokens=False, padding=False, truncation=False
            )
            query_list = list_split(
                query, split_num
            )  # cut the history into 2 pieces, the first within seq max_len limit
            query_list[0] = t1 + query_list[0] + t2  # first seq fit into template
            #breakpoint()
            
            if args.num_passage==1:
                # split into n_passages for t5, otherwise, only keep the first seq that fit into template
                #breakpoint()
                query_list = query_list[:1]
            pos = data[1]
            group = {}
            pos_list.append(item_input_ids_dict[pos])
            for id in random_neg_dict[idx]:
                neg_list.append(item_input_ids_dict[id])
            group["query"] = query_list  # a list of lists of query_ids
            group["positives"] = pos_list
            group["negatives"] = neg_list
            f.write(json.dumps(group) + "\n")

    print("-----finish------")


if __name__ == "__main__":
    main()
