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
The dataset we used is sample-350BT from fineweb dataset. 

## 3 Experiments


### 3.1 Self Reward

1️⃣ We provide a training script to reproduce the BERT classifider model

```bash
cd self_reward/reproduce
bash search.sh

cd 
```



```bash

```

### 3.2 Synthetic In-doamain Data Generation


1️⃣ Get oracle data influence:

```bash
```


2️⃣ Train data influence model:

```bash

```


### 3.3 BERT classififer

1️⃣ We provide a training script to reproduce the BERT classifider model

```bash

```

2️⃣ We provide an inference script to reproduce the data filtered by BERT classifier
```bash

```

## 4 Citation

