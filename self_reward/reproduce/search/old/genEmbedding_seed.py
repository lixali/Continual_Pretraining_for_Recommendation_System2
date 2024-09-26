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


class SeedDocDataset(Dataset):
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
        template1 = "Here is the document: "
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
        sequence_text = []
        sequence_key = []
        batch_target = []
        for example in data:
            seq_text = example["text"]
            seq = self.tokenizer.encode(
                seq_text, add_special_tokens=False, truncation=False
            )
            # Segment the user sequence text first, then add prompt to the first subsequence
            seq = list_split(seq, self.args.split_num)

            # TODO: maybe need this line when num_passage=1
            # seq = seq[:self.args.num_passage]
            seq[0] = self.template[0] + seq[0] + self.template[1]
            seq = seq[:self.args.num_passage]

            s_ids = []
            s_masks = []
            for s in seq:

                outputs = self.tokenizer.encode_plus(
                    s,
                    max_length=self.args.seed_doc_max_token,
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
            if cur_item < self.args.num_passage:
                b = self.args.num_passage - cur_item
                l = s_ids.size(1)
                pad = torch.zeros([b, l], dtype=s_ids.dtype)
                s_ids = torch.cat((s_ids, pad), dim=0)
                s_masks = torch.cat((s_masks, pad), dim=0)
            sequence_ids.append(s_ids[None])
            sequence_masks.append(s_masks[None])
            sequence_text.append(example["text"])
            sequence_key.append(example["id"])
        sequence_ids = torch.cat(sequence_ids, dim=0)
        sequence_masks = torch.cat(sequence_masks, dim=0)

        return {
            "seed_doc_ids": sequence_ids,
            "seed_doc_masks": sequence_masks,
            "text": sequence_text,
            "id": sequence_key,
        }



def list_split(array, n):
    split_list = []
    s1 = array[:n]
    s2 = array[n:]
    split_list.append(s1)
    if len(s2) != 0:
        split_list.append(s2)
    return split_list


def genEmbedding(model, seed_documents_dataloader, device, logging):
    logging.info("***** Running testing *****")
    model.eval()
    model = model.module if hasattr(model, "module") else model
    seed_doc_emb_list = []
    target_item_list = []
    seed_document_text_list = []
    seed_document_key_list = []
    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(seed_documents_dataloader), total=len(seed_documents_dataloader)
        ):
            seed_doc_inputs = batch["seed_doc_ids"].to(device)
            seed_doc_masks = batch["seed_doc_masks"].to(device)
            _, seed_doc_emb = model(seed_doc_inputs, seed_doc_masks)
            seed_doc_emb_list.append(seed_doc_emb.cpu().numpy())
            seed_document_text_list.append(batch["text"])
            seed_document_key_list.append(batch["id"])

        seed_doc_emb_list = np.concatenate(seed_doc_emb_list, 0)
        seed_document_text_list = np.concatenate(seed_document_text_list, 0)
        seed_document_key_list = np.concatenate(seed_document_key_list, 0)
        seed_documents_text_emebdding = [{"id": seed_document_key_list[i], "text": seed_document_text_list[i], "embedding": seed_doc_emb_list[i].tolist()} for i in range(len(seed_document_text_list))]

    count = 0
    with open(f'{args.seed_doc_output_embedding}', 'w') as f:
        for entry in seed_documents_text_emebdding:
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


    seed_doc = load_dataset('json', data_files=args.seed_doc, split='train')
    seed_doc_list = [{"text":seed["1_text"], "id":seed["1_id"]} for seed in seed_doc]
    seed_doc_dataset = SeedDocDataset(seed_doc_list, tokenizer=tokenizer, args=args)

    test_seed_doc_sampler = SequentialSampler(seed_doc_dataset)

    seed_documents_dataloader = DataLoader(
        seed_doc_dataset,
        sampler=test_seed_doc_sampler,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=seed_doc_dataset.collect_fn,
    )

    logger.info(f"############ Finish loading dataset into dataloader; start generating Embeddings ###########")
    
    genEmbedding(model, seed_documents_dataloader, device, logging)

    logger.info(f"############ Finish generating the embedding jsonl files ###########")
    end_time = time.time()
    end_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print(f"End time: {end_time_readable}")
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken: {int(minutes)} minutes and {seconds:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_doc', type=str, required=True, help="Path to the input file")
    parser.add_argument('--tokenizer', type=str, required=True, help="tokenizer")
    parser.add_argument('--model_dir', type=str, required=True, help="model directory")
    parser.add_argument('--seed_doc_output_embedding', type=str, required=True, help="seed_doc_output_embedding")
    parser.add_argument('--split_num', type=int, required=True, help="split_num")
    parser.add_argument('--num_passage', type=int, required=True, help="num_passage")
    parser.add_argument('--seed_doc_max_token', type=int, required=True, help="seq_size")
    parser.add_argument('--batch_size', type=int, required=True, help="batch_size")
    args = parser.parse_args()
    main(args)
