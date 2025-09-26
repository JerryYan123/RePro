#!/bin/bash

cd pretrain

method="baseline_01_0_fasttext-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000"
for SLURM_ARRAY_TASK_ID in 6; do
    export c=$SLURM_ARRAY_TASK_ID
    TMPDIR=/tmp PYTHONUNBUFFERED=1 torchrun --nproc_per_node 8 --master_port 47763 eval/eval_openlm_ckpt.py \
        --donot-compute-perplexity \
        --checkpoint /tmp/dclm_logs/${method}/checkpoints/epoch_$c.pt \
        --model ../training/open_lm_configs/open_lm_1b_swiglutorch.json \
        --config /tmp/dclm_logs/${method}/params.txt \
        --eval-yaml eval/mmlu_and_lowvar.yaml \
        --output-file results/${method}/epoch_$c/metrics_mmlu_and_lowvar.json \
        --use-temp-working-dir
    rm -rf eval_openlm_ckpt_temp_dirs_${SLURM_ARRAY_TASK_ID}
done