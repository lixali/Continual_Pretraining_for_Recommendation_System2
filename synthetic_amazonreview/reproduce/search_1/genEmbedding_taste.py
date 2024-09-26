import os

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, T5Tokenizer
from src.model import TASTEModel
from torch.utils.data import DataLoader, SequentialSampler
import argparse
from datasets import load_dataset, Features, Value
import json
import logging
import sys
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.setLevel(logging.INFO)

import time
start_time = time.time()
start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
print(f"Start time: {start_time_readable}")


class text_alike_Dataset(Dataset):
    def __init__(self, data, tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


    def collation_fn(self, data):
        sequence_ids = []
        sequence_masks = []
        ids = defaultdict(list)        
        output_dict = {}
        for seq in data:
            
            #breakpoint()
            s_ids = []
            s_masks = []

            for s in seq:
    
                outputs = self.tokenizer.encode_plus(
                    s,
                    max_length=self.args.text_alike_max_token,
                    pad_to_max_length=True,
                    return_tensors="pt",
                    truncation=True,
                )
    
                input_ids = outputs["input_ids"]
                attention_mask = outputs["attention_mask"]
                
                
                s_ids.append(input_ids)
                s_masks.append(attention_mask)
                
            s_ids = torch.cat(s_ids, dim=0)
            s_masks = torch.cat(s_masks, dim=0)
            cur_item = s_ids.size(0)
            
            if cur_item < self.args.num_passage: ### make all 0 tensor [0,0,0,0 ... 0,0,0] for the second list because we num_passage is 2, and the first list is less than 256, and it can not be splitted into two lists
                b = self.args.num_passage - cur_item
                l = s_ids.size(1)
                pad = torch.zeros([b, l], dtype=s_ids.dtype)
                s_ids = torch.cat((s_ids, pad), dim=0)
                s_masks = torch.cat((s_masks, pad), dim=0)            
            
            sequence_ids.append(s_ids[None])
            sequence_masks.append(s_masks[None])

        sequence_ids = torch.cat(sequence_ids, dim=0)
        sequence_masks = torch.cat(sequence_masks, dim=0)
        
        output_dict["text_alike_ids"] = sequence_ids
        output_dict["text_alike_masks"] = sequence_masks
        
        return output_dict



def list_split(array, n):
    split_list = []
    s1 = array[:n]
    s2 = array[n:]
    split_list.append(s1)
    if len(s2) != 0:
        split_list.append(s2)
    return split_list


def genEmbedding(model, text_alike_dataloader, device, logging, args, item_sequence_list=None, users_list=None):
    logging.info("***** Running testing *****")
    model.eval()
    model = model.module if hasattr(model, "module") else model
    text_alike_emb_list = []
    text_alike_text = defaultdict(list)
    doc_ids = defaultdict(list)
    batch_leng = len(text_alike_dataloader)
    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(text_alike_dataloader), total=len(text_alike_dataloader)
        ):
            text_alike_inputs = batch["text_alike_ids"].to(device)
            text_alike_masks = batch["text_alike_masks"].to(device)
            _, text_alike_emb = model(text_alike_inputs, text_alike_masks)
            text_alike_emb_list.append(text_alike_emb.cpu().numpy())

        if text_alike_emb_list is not []: text_alike_emb_list = np.concatenate(text_alike_emb_list, 0)


        text_alike_text_emebdding = []
        
        for j in range(text_alike_emb_list.shape[0]):
            current_dict = {}
            current_dict["item_sequence"] = item_sequence_list[j]
            current_dict["user_id"] = users_list[j]
            # Add the embedding
            if text_alike_emb_list is not []: current_dict["embedding"] = text_alike_emb_list[j].tolist()
            # Append the current dictionary to the list
            text_alike_text_emebdding.append(current_dict)
    
    count = 0
    with open(f'{args.text_alike_output_embedding}', 'w') as f:
        for entry in text_alike_text_emebdding:
            if count % 10000 == 0:print(f"Processed seed {count} rows")
            count += 1
            json.dump(entry, f)
            f.write('\n')


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer =  T5Tokenizer.from_pretrained(args.tokenizer, local_files_only=True) 
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model_class = TASTEModel
    model = model_class.from_pretrained(args.model_dir)
    model.to(device)
    # collator=collator_class(args)


    text_alike = load_dataset('json', data_files=args.text_alike, split='train')
    
    text_alike_list = []
    item_sequence_list = []
    users_list = []
    for _, row in enumerate(tqdm(text_alike)):

        text_alike_list.append(row["query"])
        item_sequence_list.append(row["item_sequence"])
        users_list.append(row["user_id"])
    
    text_alike_dataset = text_alike_Dataset(text_alike_list, tokenizer=tokenizer, args=args)

    text_alike_sampler = SequentialSampler(text_alike_dataset)

    text_alike_dataloader = DataLoader(
        text_alike_dataset,
        sampler=text_alike_sampler,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=text_alike_dataset.collation_fn,
    )

    logger.info(f"############ Finish loading dataset into dataloader; start generating Embeddings ###########")
    
    genEmbedding(model, text_alike_dataloader, device, logging, args, item_sequence_list, users_list)

    logger.info(f"############ Finish generating the embedding jsonl files ###########")
    end_time = time.time()
    end_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print(f"End time: {end_time_readable}")
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken: {int(minutes)} minutes and {seconds:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_alike', type=str, required=True, help="Path to the input file")
    parser.add_argument('--tokenizer', type=str, required=True, help="tokenizer")
    parser.add_argument('--model_dir', type=str, required=True, help="model directory")
    parser.add_argument('--text_alike_output_embedding', type=str, required=True, help="text_alike_output_embedding")
    parser.add_argument('--split_num', type=int, required=True, help="split_num")
    parser.add_argument('--num_passage', type=int, required=True, help="num_passage")
    parser.add_argument('--text_alike_max_token', type=int, required=True, help="seq_size")
    parser.add_argument('--batch_size', type=int, required=True, help="batch_size")
    
    args = parser.parse_args()
    main(args)
