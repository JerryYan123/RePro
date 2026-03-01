#!/bin/bash
#SBATCH --job-name=repro-rl-0.6B
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --exclude=babel-s5-24
#SBATCH --exclude=babel-s5-24
#SBATCH --output=/home/jerryy2/repro_logs/repro-rl-0.6B-%j.out
#SBATCH --error=/home/jerryy2/repro_logs/repro-rl-0.6B-%j.err

set -euo pipefail
mkdir -p ~/repro_logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate repro

export HF_HOME=/data/user_data/jerryy2/hf_cache
export HF_HUB_CACHE=/data/user_data/jerryy2/hf_cache/hub
export TRANSFORMERS_CACHE=/data/user_data/jerryy2/hf_cache
export HF_DATASETS_CACHE=/data/user_data/jerryy2/hf_cache/datasets
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

export TMPDIR=/scratch/${SLURM_JOB_ID}
export VLLM_CACHE_ROOT=$TMPDIR/vllm
export TRITON_CACHE_DIR=$TMPDIR/triton
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"
cleanup() { kill $DATAMAN_PID $STRUCTURE_PID 2>/dev/null; rm -rf "$TMPDIR"; }
trap cleanup EXIT
cd ~/RePro/rl

DATAMAN_PORT=38000
STRUCTURE_PORT=38001

echo "[$(date)] Starting DataMan server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model RuPeng/DataMan-1.5B-EN --port $DATAMAN_PORT --max-model-len 2048 &
DATAMAN_PID=$!

echo "[$(date)] Starting Structure server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B --port $STRUCTURE_PORT --max-model-len 2048 &
STRUCTURE_PID=$!

echo "[$(date)] Waiting for servers..."
for port in $DATAMAN_PORT $STRUCTURE_PORT; do
    for i in $(seq 1 60); do
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            echo "[$(date)] Server on port $port is ready!"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "[$(date)] Server on port $port failed to start"
            exit 1
        fi
        sleep 5
    done
done

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=600000

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
echo "[$(date)] Starting GRPO training..."
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 PYTHONPATH=$PWD/src ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2_6gpu.yaml \
    src/open_r1/grpo_synthetic.py \
    --config recipes/Qwen3/grpo/config_0.6B.yaml

echo "[$(date)] Training complete!"
