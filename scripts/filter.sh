#!/bin/bash

cd pretrain

mkdir -p $SPILL_LOCATION
ray start --head --temp-dir=$SPILL_LOCATION --system-config='{"object_spilling_config":"{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"/tmp/ray\"}}"}'

TMPDIR=/tmp PYTHONPATH=$(pwd) python ray_processing/process.py \
    --source_ref_paths exp_data/datasets/raw_sources/refinedweb_01_0_Qwen.json \
    --readable_name fasttext_01_0 \
    --output_dir /tmp/data/Qwen3-4B-grpo-1980/fasttext_7.2B \
    --config_path baselines/baselines_configs/fasttext_filter.yaml \
    --source_name cc

rm -rf exp_data/datasets/untokenized/fasttext_01_0.json

ray stop
rm -rf $SPILL_LOCATION