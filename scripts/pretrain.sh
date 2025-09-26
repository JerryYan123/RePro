#!/bin/bash

cd pretrain

torchrun --nproc-per-node 8 -m training.train -- \
    --scale 411m_1x \
    --data-config exp_data/datasets/tokenized/baseline_01_0_fasttext_repro.json \
    --logs /tmp/dclm_logs \
    --multiple-data-passes \
    --report-to-wandb