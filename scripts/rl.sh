#!/bin/bash

cd rl

CUDA_VISIBLE_DEVICES=0 vllm serve RuPeng/DataMan-1.5B-EN --port 8000 > logs/dataman_llm.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-4B --port 8001 > logs/structure_llm.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 PYTHONPATH=$PWD/src ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/grpo_synthetic.py --config recipes/Qwen3/grpo/config_4B.yaml