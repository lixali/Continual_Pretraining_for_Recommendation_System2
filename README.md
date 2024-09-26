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

Our pretraining is run stage by stage to facilitate the model-aware data selection. Each stage consists of 10k steps. For instance, in the initial warmup 10k steps, you can run:

```bash

```

- `ckpt=0` denotes we are training from scratch.

To resume the pretraining from previous steps (e.g., 10k), you can run:

```bash

```

- `ckpt=40000` denotes our gradient accumulation step is 4.
- `method=random` is the random data selection. You can replace it with `mates` for MATES after the first 10k steps.

### 3.2 Synthetic In-doamain Data Generation

After the first 10k steps, we can start the MATES data selection process every 10k steps. One data selection process consists of four steps:

1️⃣ Get oracle data influence:

```bash
```

- For the 10k checkpoint, `method=random`, but for the following, `method=mates`.

2️⃣ Train data influence model:

```bash

```

- The selected data will be saved in `data/c4/pythia-410m/mates/40000`.

### 3.3 BERT classififer

1️⃣ It is advised to run the evaluation after the decay stage for intermediate checkpoints for better stability.

```bash
model_name=pythia-410m \
method=mates \
ckpt=80000 \
decay=true \
bash scripts/pretrain.sh
```

2️⃣ We provide a simple evaluation example here and you can modify the parameters based on your needs.

```bash
model_name=pythia-410m \
method=mates \
ckpt=80800 \
bash scripts/eval.sh
```

- After running the evaluation script, you can find the results in the `results/c4/$model/$method/iter-$ckpt-ckpt/results.json`.

## 4 Citation

Please cite our work
