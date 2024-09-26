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

class PoolDocDocument(Dataset):
    def __init__(self, data, tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collect_fn(self, batch):

        pool_doc_text = []
        pool_doc_key = []
        for example in batch:
            pool_doc_text.append(example["text"])
            pool_doc_key.append(example["id"])
        item_ids, item_masks = encode_batch(pool_doc_text, self.tokenizer, self.args.pool_doc_max_token_length)
        return {
            "pool_doc_ids": item_ids,
            "pool_doc_masks": item_masks,
            "text": pool_doc_text,
            "id": pool_doc_key,        
            }


def encode_batch(batch_text, tokenizer, max_length):
    outputs = tokenizer.batch_encode_plus(
        batch_text,
        max_length=max_length,
        pad_to_max_length=True,
        return_tensors="pt",
        truncation=True,
    )
    input_ids = outputs["input_ids"]
    input_ids = torch.unsqueeze(input_ids, 1)
    attention_mask = outputs["attention_mask"]
    attention_mask = torch.unsqueeze(attention_mask, 1)

    return input_ids, attention_mask



def genEmbedding(model, pool_documents_dataloader, device, logging):
    logging.info("***** Running testing *****")
    model.eval()
    model = model.module if hasattr(model, "module") else model
    pool_doc_emb_list = []
    target_item_list = []
    pool_document_text_list = []
    pool_document_key_list = []
    

    with torch.no_grad():
        for _i, batch in tqdm(
            enumerate(pool_documents_dataloader), total=len(pool_documents_dataloader)
        ):
            pool_doc_inputs = batch["pool_doc_ids"].to(device)
            pool_doc_masks = batch["pool_doc_masks"].to(device)
            batch_target = batch["text"]
            _, pool_doc_emb = model(pool_doc_inputs, pool_doc_masks)
            pool_doc_emb_list.append(pool_doc_emb.cpu().numpy())
            pool_document_text_list.append(batch["text"])
            pool_document_key_list.append(batch["id"])
            target_item_list.extend(batch_target)
        
        
        pool_doc_emb_list = np.concatenate(pool_doc_emb_list, 0)
        pool_document_text_list = np.concatenate(pool_document_text_list, 0)
        pool_document_key_list = np.concatenate(pool_document_key_list, 0)
        pool_documents_text_emebdding = [{"id": pool_document_key_list[i], "text": pool_document_text_list[i], "embedding": pool_doc_emb_list[i].tolist()} for i in range(len(pool_document_text_list))]

    # breakpoint()
    count = 0
   
    with open(args.pool_doc_output_embedding, 'w') as f:
        for entry in pool_documents_text_emebdding:
            if count % 10000 == 0:print(f"Processed pool {count} rows")
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

    pool_doc = load_dataset('json', data_files=args.pool_doc, split='train')
    pool_doc_list = [{"text":pool["text"], "id":pool["id"]} for pool in pool_doc]
    pool_doc_dataset = PoolDocDocument(pool_doc_list, tokenizer=tokenizer, args=args)

    test_pool_doc_sampler = SequentialSampler(pool_doc_dataset)

    pool_doc_dataloader = DataLoader(
        pool_doc_dataset,
        sampler=test_pool_doc_sampler,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=pool_doc_dataset.collect_fn,
    )
    # for i, batch in enumerate(documents_dataloader): 
    #     # breakpoint()
    #     logger.info(batch["input_ids"])

    logger.info(f"############ Finish loading dataset into dataloader; start generating Embeddings ###########")
    
    genEmbedding(model, pool_doc_dataloader, device, logging)

    logger.info(f"############ Finish generating the embedding jsonl file ###########")
    end_time = time.time()
    end_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print(f"End time: {end_time_readable}")
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken: {int(minutes)} minutes and {seconds:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool_doc', type=str, required=True, help="Path to the input file")
    parser.add_argument('--tokenizer', type=str, required=True, help="tokenizer")
    parser.add_argument('--model_dir', type=str, required=True, help="model directory")
    parser.add_argument('--pool_doc_output_embedding', type=str, required=True, help="pool_doc_output_embedding")
    parser.add_argument('--num_passage', type=int, required=True, help="num_passage")
    parser.add_argument('--pool_doc_max_token_length', type=int, required=True, help="item_size")
    parser.add_argument('--batch_size', type=int, required=True, help="batch_size")
    args = parser.parse_args()
    main(args)
