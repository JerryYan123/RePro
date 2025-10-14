# RePro: Training Language Models to Faithfully Recycle the Web for Pretraining

<p align="center"><a href='https://arxiv.org/abs/2510.10681'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://huggingface.co/cx-cmu/repro-rephraser-4B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Rephraser-blue'></p>

Recycled data coming soon...

## Quick Links

- [Prerequisites](#prerequisites)
- [Rephraser Training](#rephraser-training)
- [Rephraser Inference](#rephraser-inference)
- [Quality Filtering](#quality-filtering)
- [Tokenization](#tokenization)
- [Pretraining](#pretraining)
- [Evaluation](#evaluation)

Our RL code is based on Open R1, and our pretraining code is based on DCLM. Please refer to their repos for more details.

## Prerequisites

The code is tested on Python 3.10.16. Install basic dependencies:

```bash
pip install -r requirements.txt
```

Run setup.py to download necessary files:

```bash
cd pretrain
python setup.py install
```

## Rephraser Training

Run `scripts/rl.sh`.

- We provide a 1000-example subset of the training data (`rl/1000_sample_low_score.jsonl`) for testing purposes.

## Rephraser Inference

```bash
bash scripts/infer.sh 0 7
```

- 0 and 7 are the start and end index of the data shards you want to process, you can change them based on your need.
- We processed 600 shards for 72B tokens.

## Quality Filtering

Run `scripts/filter.sh`:

- `source_ref_paths`: data pool path
- `output_dir`: filtered data dir

## Tokenization

Please install rust in your conda environment.

Run `scripts/tokenize.sh`:

- `input`: the original text data dir
- `output`: the tokenized data dir

## Pretraining

Run `scripts/pretrain.sh`:

- `scale`: DCLM running scale, please find the supported ones in `training/configs`
- `data-config`: specify the run name ("name") and the tokenized data location ("manifest_url"), create one when you have a new dataset
- `logs`: where to store the checkpoint
- `multiple-data-passes`: used to allow multiple epochs

## Evaluation

Run `scripts/eval.sh`:

- `method`: the generated checkpoint dir name
- `checkpoint`: the specific epoch you want to evaluate
- `model`: model scale config in `training/open_lm_configs`
- `output-file`: where to store the evaluation result