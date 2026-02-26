#!/bin/bash
#SBATCH --job-name=repro-rl
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:8
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jerryy2/repro_logs/repro-rl-%j.out
#SBATCH --error=/home/jerryy2/repro_logs/repro-rl-%j.err

source ~/.bashrc
conda activate repro
export HF_HOME=/data/user_data/jerryy2/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

cd /home/jerryy2/RePro/rl

OUTPUT_DIR=/data/user_data/jerryy2/repro_checkpoints/Qwen3-4B_grpo
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "[$(date)] Starting DataMan server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 vllm serve RuPeng/DataMan-1.5B-EN --port 8000 > logs/dataman_llm.log 2>&1 &
DATAMAN_PID=$!

echo "[$(date)] Starting Structure server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-4B --port 8001 > logs/structure_llm.log 2>&1 &
STRUCTURE_PID=$!

echo "[$(date)] Waiting for servers..."
for port in 8000 8001; do
    for i in $(seq 1 60); do
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            echo "[$(date)] Server on port $port is ready!"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "[$(date)] ERROR: Server on port $port failed to start!"
            kill $DATAMAN_PID $STRUCTURE_PID 2>/dev/null
            exit 1
        fi
        sleep 5
    done
done

echo "[$(date)] Starting GRPO training..."
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 PYTHONPATH=$PWD/src ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/grpo_synthetic.py \
    --config recipes/Qwen3/grpo/config_4B.yaml

echo "[$(date)] Training complete!"
kill $DATAMAN_PID $STRUCTURE_PID 2>/dev/null
