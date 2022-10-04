# Controllable Text Generation with Neurally-Decomposed Oracle
This repository contains the source code to reproduce the experiments in NeurIPS 2022 paper
[Controllable Text Generation with Neurally-Decomposed Oracle](https://arxiv.org/abs/2205.14219) by [Tao Meng](https://mtsomethree.github.io/), [Sidi Lu](http://sidilu.cn/), [Nanyun Peng](https://vnpeng.net/) and [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/).

- ### Abstract
We propose a general and efficient framework to control auto-regressive generation models with NeurAlly-Decomposed Oracle (NADO). Given a pre-trained base language model and a sequence-level boolean oracle function, we propose to decompose the oracle function into token-level guidance to steer the base model in text generation. Specifically, the token-level guidance is approximated by a neural model trained with examples sampled from the base model, demanding no additional auxiliary labeled data. We present the closed-form optimal solution to incorporate the token-level guidance into the base model for controllable generation. We further provide a theoretical analysis of how the approximation quality of NADO affects the controllable generation results. Experiments conducted on two applications: (1) text generation with lexical constraints and (2) machine translation with formality control demonstrate that our framework efficiently guides the base model towards the given oracle while maintaining high generation quality.

- ### Experiments
This repository will contain both experiments described in this paper. So far the LCG part is still under construction and expected to come out later October.

- ### Data

The machine translation formality change experiments leverage the [CALLHOME Spanish-English Speech Translation Corpus](https://aclanthology.org/2013.iwslt-papers.14/) as source data, and evaluate the BLUE score with the [fluent references](https://aclanthology.org/N19-1285/). Note that LDC access is required for the first dataset.

- ### Running experiments

**Requirements**

```bash
pip install -r requirements.txt
```

**Running**

```bash
python train_MT.py
```

The code will automatically download [MarianMT model](https://huggingface.co/docs/transformers/model_doc/marian) and sample translated texts from source texts from Fisher-and-Callhome Corpus. The sampled data will be dumped in ./dump/MT directory. The sampled data is labeled by an formality oracle trained in [FUDGE paper](https://arxiv.org/abs/2104.05218). A NADO model will be trained by those labeled sampled data. The translated results will be evaluated based on oracle scores and the BLEU scores compared to fluent references.

```bash
Alternative arguments:
  --sample_batch_size  the batch size in sampling. Must be integer times of 8.
  --batch_size  the batch size in training. Must be a divider of sample_batch_size
  --regularization  the strength of regularization
  --max_length  the maximum length accepted in training or evaluation
```
