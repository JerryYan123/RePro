#!/bin/bash
#SBATCH --job-name=repro-pretrain
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:8
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jerryy2/repro_logs/repro-pretrain-%j.out
#SBATCH --error=/home/jerryy2/repro_logs/repro-pretrain-%j.err

source ~/.bashrc
conda activate repro
export HF_HOME=/data/user_data/jerryy2/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

cd /home/jerryy2/RePro/pretrain

LOG_DIR=/data/user_data/jerryy2/repro_output/pretrain_logs
DATA_CONFIG=${1:-"exp_data/datasets/tokenized/baseline_01_0_fasttext_repro.json"}
mkdir -p $LOG_DIR

echo "[$(date)] Starting pretraining..."
echo "[$(date)] Data config: $DATA_CONFIG"
echo "[$(date)] Logs dir: $LOG_DIR"

torchrun --nproc-per-node 8 -m training.train -- \
    --scale 411m_1x \
    --data-config $DATA_CONFIG \
    --logs $LOG_DIR \
    --multiple-data-passes \
    --report-to-wandb

echo "[$(date)] Pretraining complete!"
