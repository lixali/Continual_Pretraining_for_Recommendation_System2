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
        self.template = self.toke_template()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def toke_template(self):
        t_list = []
        template1 = "Here is the visit history list of user:"
        template2 = " recommend next item"
        t1 = self.tokenizer.encode(
            template1, add_special_tokens=False, truncation=False
        )
        t_list.append(t1)
        t2 = self.tokenizer.encode(
            template2, add_special_tokens=False, truncation=False
        )

        t_list.append(t2)
        return t_list

    def collect_fn(self, data):
        sequence_ids = []
        sequence_masks = []
        user_ids = [] 
        sequence_text = defaultdict(list)
        ids = defaultdict(list)
        output_dict = {}
        for example in data:
            seq = []
            
            doc_text = []
            doc_text_id = []
            doc_text_token_id = []
            
            m = 0 
            for i in range(args.leng-1, -1, -1): ### starting from last, reversed order
                k = i + 1
                doc_text.append(example[f"{k}_text"])
                doc_text_id.append(example[f"{k}_id"])
             
                doc_text_token_id.append(self.tokenizer.encode(
                    doc_text[m], add_special_tokens=False, truncation=False
                ))
                
                m+=1


            # Segment the user sequence text first, then add prompt to the first subsequence; this is for self reward different doc seq length
                #breakpoint()
                list_id = args.leng - i - 1 
                doc_text_token_id[list_id] = list_split(doc_text_token_id[list_id], self.args.split_num//args.leng)

                seq = seq + doc_text_token_id[list_id][0]  + [117] ## 117 is the semi conlon ";"
            seq = list_split(seq, self.args.split_num)
            
            # Following one line are for in-domain taste synthetic data generation different item sequence length
            #flattern_seq = [token for l in doc_text_token_id for token in l]
            #seq = list_split(flattern_seq, self.args.split_num)
            
            seq[0] = self.template[0] + seq[0] + self.template[1] 
            
            if self.args.num_passage == 1: seq = seq[:1]
                 
            #if len(seq) == 2 and seq[1] > 512: seq[1] = seq[1][:512]          

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

            for i in range(args.leng):
                sequence_text[i].append(doc_text[i])
                ids[i].append(doc_text_id[i])
        
        
            if "user_id" in example: user_ids.append(example["user_id"])

        sequence_ids = torch.cat(sequence_ids, dim=0)
        sequence_masks = torch.cat(sequence_masks, dim=0)
        if user_ids: user_ids = torch.tensor(user_ids) 
        for i in range(args.leng):
            output_dict[f"{i+1}_text"] = sequence_text[i]
            output_dict[f"{i+1}_id"] = ids[i]
        output_dict["text_alike_ids"] = sequence_ids
        output_dict["text_alike_masks"] = sequence_masks
        if user_ids !=  []: 
            output_dict["user_ids"] = user_ids

        #breakpoint()         
        return output_dict



def list_split(array, n):
    split_list = []
    s1 = array[:n]
    s2 = array[n:]
    split_list.append(s1)
    if len(s2) != 0:
        split_list.append(s2)
    return split_list


def genEmbedding(model, text_alike_dataloader, device, logging, args):
    logging.info("***** Running testing *****")
    model.eval()
    model = model.module if hasattr(model, "module") else model
    text_alike_emb_list = []
    text_alike_text = defaultdict(list)
    doc_ids = defaultdict(list)
    user_ids = []
    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(text_alike_dataloader), total=len(text_alike_dataloader)
        ):
            text_alike_inputs = batch["text_alike_ids"].to(device)
            text_alike_masks = batch["text_alike_masks"].to(device)
            _, text_alike_emb = model(text_alike_inputs, text_alike_masks)
            text_alike_emb_list.append(text_alike_emb.cpu().numpy())
            #breakpoint()
            if "user_ids" in batch and batch["user_ids"] != []:
                #if args.leng == 2: breakpoint()
                print("bath['user_id'] is", batch["user_ids"])
                user_ids.append(batch["user_ids"].cpu())
            
            for i in range(args.leng):
                text_alike_text[f"{i+1}_text"].append(batch[f"{i+1}_text"])
                doc_ids[f"{i+1}_id"].append(batch[f"{i+1}_id"])    

        if text_alike_emb_list !=  []: text_alike_emb_list = np.concatenate(text_alike_emb_list, 0)
        if user_ids != []: 

            user_ids = np.concatenate(user_ids, 0)

        for i in range(args.leng):
            text_alike_text[f"{i+1}_text"] = np.concatenate(text_alike_text[f"{i+1}_text"], 0) 
            doc_ids[f"{i+1}_id"] = np.concatenate(doc_ids[f"{i+1}_id"], 0)


        #breakpoint()
        text_alike_text_emebdding = []
        
        for j in range(len(text_alike_text["1_text"])):
            current_dict = {}
            
            # Add text and id for each 'i'
            for i in range(args.leng):
                current_dict[f"{i+1}_text"] = text_alike_text[f"{i+1}_text"][j]
                current_dict[f"{i+1}_id"] = int(doc_ids[f"{i+1}_id"][j]) if isinstance(doc_ids[f"{i+1}_id"][j], np.int64) else doc_ids[f"{i+1}_id"][j]
            
            # Add the embedding
            if text_alike_emb_list.size != 0: current_dict["embedding"] = text_alike_emb_list[j].tolist()
            if user_ids !=  []: current_dict["user_id"] = int(user_ids[j])
            # Append the current dictionary to the list
            text_alike_text_emebdding.append(current_dict)
        
    #breakpoint()
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

    for _, row in enumerate(tqdm(text_alike)):
        my_dict = {}
        for i in range(args.leng):
            my_dict[f"{i+1}_text"] = row[f"{i+1}_text"]
            my_dict[f"{i+1}_id"] = row[f"{i+1}_id"]

        if "user_id" in row: my_dict["user_id"] = row["user_id"]
        text_alike_list.append(my_dict)
    
    text_alike_dataset = text_alike_Dataset(text_alike_list, tokenizer=tokenizer, args=args)

    text_alike_sampler = SequentialSampler(text_alike_dataset)

    text_alike_dataloader = DataLoader(
        text_alike_dataset,
        sampler=text_alike_sampler,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=text_alike_dataset.collect_fn,
    )

    logger.info(f"############ Finish loading dataset into dataloader; start generating Embeddings ###########")
    
    genEmbedding(model, text_alike_dataloader, device, logging, args)

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
    parser.add_argument('--leng', type=int, required=True, help="length")
    
    
    args = parser.parse_args()
    main(args)
