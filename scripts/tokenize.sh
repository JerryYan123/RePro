#!/bin/bash

cd pretrain/rust_processing/tokshuf-rs

cargo run --release -- \
    --input /tmp/data/Qwen3-4B-grpo-1980/fasttext_7.2B/fasttext_filter/processed_data \
    --local-cell-dir /tmp \
    --output /tmp/data/Qwen3-4B-grpo-1980/fasttext_7.2B/tokenized \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --use-tiktoken \
    --seqlen 2049 \
    --wds-chunk-size 8192 \
    --num-local-cells 512