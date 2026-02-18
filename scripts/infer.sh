#!/bin/bash

cd rl

begin=$1
end=$2
mkdir -p /tmp/logs/${begin}_${end}
gpu_index=0
for ((s=begin; s<=end; s++)); do
    echo $s
    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=$gpu_index python src/infer/run_infer.py $s \
    --model /tmp/synthetic_data_generator_Qwen3-4B_grpo/checkpoint-1980 \
    --write_template "data/refinedweb_01_0/Qwen3-4B-grpo-1980/textfiles/shard_{}_processed.jsonl" \
    > /tmp/logs/${begin}_${end}/log_job_s${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done