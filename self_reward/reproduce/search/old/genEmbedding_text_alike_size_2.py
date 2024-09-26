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
        template1 = "Here is the documents history: "
        template2 = " recommend next document "
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
        sequence_text_1 = []
        id_1 = []
        sequence_text_2 = []
        id_2 = []
        
        for example in data:
            doc_text_1 = example["seed_text"]
            doc_text_1_id = example["seed_id"]
            doc_text_2 = example["1st_nearest_text"]
            doc_text_2_id = example["1st_nearest_id"]
             
            doc_text_1_token_id = self.tokenizer.encode(
                doc_text_1, add_special_tokens=False, truncation=False
            )

            doc_text_2_token_id = self.tokenizer.encode(
                doc_text_2, add_special_tokens=False, truncation=False
            )

            # Segment the user sequence text first, then add prompt to the first subsequence
            
            doc_text_1_token_id = list_split(doc_text_1_token_id, self.args.split_num//2)
            doc_text_2_token_id = list_split(doc_text_2_token_id, self.args.split_num//2)

            seq = self.template[0] + doc_text_1_token_id[0]  + [117] +  doc_text_2_token_id[0] + self.template[1] ## 117 is the semi conlon ";"
            
            s_ids = []
            s_masks = []
            s = seq

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
            
            
            sequence_ids.append(s_ids[None])
            sequence_masks.append(s_masks[None])
            sequence_text_1.append(doc_text_1)
            sequence_text_2.append(doc_text_2)
            id_1.append(doc_text_1_id)
            id_2.append(doc_text_2_id)
        sequence_ids = torch.cat(sequence_ids, dim=0)
        sequence_masks = torch.cat(sequence_masks, dim=0)
        
        return {
            "text_alike_ids": sequence_ids,
            "text_alike_masks": sequence_masks,
            "seed_doc": sequence_text_1,
            "seed_id": id_1, 
            "1st_closest_text": sequence_text_2,
            "1st_closest_id": id_2,
        }



def list_split(array, n):
    split_list = []
    s1 = array[:n]
    s2 = array[n:]
    split_list.append(s1)
    if len(s2) != 0:
        split_list.append(s2)
    return split_list


def genEmbedding(model, text_alike_dataloader, device, logging):
    logging.info("***** Running testing *****")
    model.eval()
    model = model.module if hasattr(model, "module") else model
    text_alike_emb_list = []
    target_item_list = []
    text_alike_text_list_1 = []
    text_alike_text_list_2 = []
    text_alike_key_list = []
    seed_doc_id = []
    second_doc_id = []
    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(text_alike_dataloader), total=len(text_alike_dataloader)
        ):
            text_alike_inputs = batch["text_alike_ids"].to(device)
            text_alike_masks = batch["text_alike_masks"].to(device)
            _, text_alike_emb = model(text_alike_inputs, text_alike_masks)
            text_alike_emb_list.append(text_alike_emb.cpu().numpy())
            text_alike_text_list_1.append(batch["seed_doc"])
            text_alike_text_list_2.append(batch["1st_closest_text"])
            seed_doc_id.append(batch["seed_id"])
            second_doc_id.append(batch["1st_closest_id"])

        text_alike_emb_list = np.concatenate(text_alike_emb_list, 0)
        text_alike_text_list_1 = np.concatenate(text_alike_text_list_1, 0)
        text_alike_text_list_2 = np.concatenate(text_alike_text_list_2, 0)
        seed_doc_id = np.concatenate(seed_doc_id, 0)
        second_doc_id = np.concatenate(second_doc_id, 0)



        text_alike_text_emebdding = [{"seed_doc": text_alike_text_list_1[i],"1_id": seed_doc_id[i], "1st_closest_text": text_alike_text_list_2[i], "2_id": second_doc_id[i], "embedding": text_alike_emb_list[i].tolist()} for i in range(len(text_alike_text_list_1))]

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
    text_alike_list = [{"seed_text":row["seed_text"], "seed_id": row["seed_id"], "1st_nearest_text":row["1st_nearest_text"], "1st_nearest_id": row["1st_nearest_id"]} for row in text_alike]
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
    
    genEmbedding(model, text_alike_dataloader, device, logging)

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
