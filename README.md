# Continual Pretraining for Recommendation System


## Quick Links

- [1 Environment](#1-environment)
- [2 Dataset](#2-dataset)
- [3 Experiments](#3-experiments)
  - [3.1 Self Reward](#31-Self-Reward)
  - [3.2 Synthetic In-doamain Data Generation](#32-Synthetic-In-doamain-Data-Generation)
  - [3.3 BERT classififer](#33-BERT-classififer)
- [4 Citation](#4-citation)

## 1 Environment

**Python version**

The code is tested on Python 3.8.19

**Install basic dependencies**

```bash
conda env create -f environment.yml 
```

## 2 Dataset
The out-of-domain dataset we used is sample-350BT from fineweb dataset. The in-domain dataset we used in AmazonReivew beauty, toys and sports

## 3 Experiments


### 3.1 Self Reward

- We provide scripts to generate contrastive pairs for T5 based recommendation embedding model training

```bash
cd self_reward/reproduce
bash search.sh

cd self_reward/build_train
query_pos_neg_tokens_id_gen.py
```


### 3.2 Synthetic In-doamain Data Generation

- We provide scritps to reporduce the synthetic in-domain data (generate various output for different sequence length) that we used

```bash
cd synthetic_amazonreview/reproduce/search
bash search.sh

cd synthetic_amazonreview/reproduce/build_train
bash build_train_t5.sh

```

- We provide scritps to reporduce the synthetic in-domain data (only generate the last output given the sequence) that we used

```bash
cd synthetic_amazonreview/reproduce/search_1
bash search.sh

cd synthetic_amazonreview/reproduce/build_train_1
bash build_train_t5.sh
```


### 3.3 BERT classififer

- We provide a training script to reproduce the BERT classifider model

```bash
cd bert/train
bash launch_train_torchx.sh

```

## 4 Citation

