import argparse
import logging
import sys

import pandas as pd
from datasets import Dataset, Features, load_dataset, Value
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.setLevel(logging.INFO)
import csv

# import wandb
import os

import random

from datasets import Dataset

random.seed(42)
import random

import jsonlines


def log_time_samples():
    total_samples_trained = (
        trainer.state.global_step * training_args.per_device_train_batch_size
    )
    elapsed_time = time.time() - trainer.state.start_time
    logger.info(
        f"Elapsed time: {elapsed_time:.2f} seconds | Samples trained: {total_samples_trained}"
    )


import pandas as pd
from datasets import concatenate_datasets, Dataset, load_dataset, load_from_disk


def main(args):
    # Load the dataset
    os.environ["WANDB_DISABLED"] = "true"
    logger.info(f"############ Beginning ###########")

    train_dataset_tokenized = load_from_disk(
        "/mnt/mffuse/pretrain_recommendation/bertclassifier/train_dataset_tokenized3"
    )
    valid_dataset_tokenized = load_from_disk(
        "/mnt/mffuse/pretrain_recommendation/bertclassifier//valid_dataset_tokenized3"
    )
    model = BertForSequenceClassification.from_pretrained(
        "/mnt/mffuse/pretrain_recommendation/bertclassifier/bert-base-uncased",
        local_files_only=True,
        num_labels=2,
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=30,
        weight_decay=0.01,
        save_steps=100,
        eval_steps=100,
        logging_steps=100,
        report_to=["tensorboard"],  # Use TensorBoard for logging
        logging_dir=args.tensorboard_dir,  # Directory for TensorBoard logs
        learning_rate=1e-5,
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=valid_dataset_tokenized,
    )

    logger.info(f"############ start training ###########")
    checkpoint = None
    if os.path.exists(training_args.output_dir) and os.listdir(
        training_args.output_dir
    ):
        checkpoints = os.listdir(training_args.output_dir)
        if checkpoints[-1] == "best_dev":  # ignore any temporary evaluation result
            checkpoints = checkpoints[:-1]
        checkpoint = os.path.join(training_args.output_dir, checkpoints[-1])
        logger.info(
            f"---- try resuming from checkpoint: {training_args.resume_from_checkpoint}... ----"
        )

    logger.info(f"---- checkpoint: {checkpoint} ----")

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(args.output_dir)

    logger.info(f"############ finish ###########")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output results",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        required=True,
        help="Directory to save the tensorboard log results",
    )
    args = parser.parse_args()
    main(args)
