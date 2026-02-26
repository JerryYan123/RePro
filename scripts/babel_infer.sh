#!/bin/bash
#SBATCH --job-name=repro-infer
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:8
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jerryy2/repro_logs/repro-infer-%j.out
#SBATCH --error=/home/jerryy2/repro_logs/repro-infer-%j.err

source ~/.bashrc
conda activate repro
export HF_HOME=/data/user_data/jerryy2/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

cd /home/jerryy2/RePro/rl

CHECKPOINT_DIR=/data/user_data/jerryy2/repro_checkpoints/Qwen3-4B_grpo
CHECKPOINT=${1:-"checkpoint-1980"}
INPUT_DIR=${2:-"/data/user_data/jerryy2/repro_data/dclm_refinedweb"}
OUTPUT_DIR=/data/user_data/jerryy2/repro_output/infer
mkdir -p $OUTPUT_DIR

BEGIN=${3:-0}
END=${4:-7}

echo "[$(date)] Starting inference"
echo "[$(date)] Model: $CHECKPOINT_DIR/$CHECKPOINT"
echo "[$(date)] Input: $INPUT_DIR"
echo "[$(date)] Output: $OUTPUT_DIR"
echo "[$(date)] Shards: $BEGIN to $END"

gpu_index=0
for ((s=BEGIN; s<=END; s++)); do
    echo "[$(date)] Launching shard $s on GPU $gpu_index"
    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=$gpu_index python src/infer/run_infer.py $s \
        --model $CHECKPOINT_DIR/$CHECKPOINT \
        --read_dir $INPUT_DIR \
        --write_dir $OUTPUT_DIR \
        > /home/jerryy2/repro_logs/infer_shard_${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done

echo "[$(date)] All inference jobs launched, waiting..."
wait
echo "[$(date)] Inference complete!"
