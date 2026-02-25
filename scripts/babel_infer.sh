#!/bin/bash
#SBATCH --job-name=repro-infer
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100_80GB:8
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/user_data/jerryy2/logs/repro-infer-%j.out
#SBATCH --error=/data/user_data/jerryy2/logs/repro-infer-%j.err

source ~/.bashrc
conda activate repro
export HF_HOME=/data/user_data/jerryy2/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

cd /home/jerryy2/RePro/rl

CHECKPOINT_DIR=/data/user_data/jerryy2/repro_checkpoints/Qwen3-4B_grpo
CHECKPOINT=${1:-"checkpoint-1980"}
OUTPUT_BASE=/data/user_data/jerryy2/repro_output/infer
mkdir -p /data/user_data/jerryy2/logs
mkdir -p $OUTPUT_BASE

BEGIN=${2:-0}
END=${3:-7}

echo "[$(date)] Starting inference with checkpoint: $CHECKPOINT_DIR/$CHECKPOINT"
echo "[$(date)] Processing shards $BEGIN to $END"

gpu_index=0
for ((s=BEGIN; s<=END; s++)); do
    echo "[$(date)] Launching shard $s on GPU $gpu_index"
    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=$gpu_index python src/infer/run_infer.py $s \
        --model $CHECKPOINT_DIR/$CHECKPOINT \
        --write_template "$OUTPUT_BASE/shard_{}_processed.jsonl" \
        > /data/user_data/jerryy2/logs/infer_shard_${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done

echo "[$(date)] All inference jobs launched, waiting..."
wait
echo "[$(date)] Inference complete!"
