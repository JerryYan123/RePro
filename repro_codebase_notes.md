# RePro Notes

Repo: [cxcscmu/RePro](https://github.com/cxcscmu/RePro) | Fork: [JerryYan123/RePro](https://github.com/JerryYan123/RePro)

---

## Quick Start: Run RL Training on Babel

```bash
# 1. LOCAL: push code changes
cd ~/Desktop/00research-2026/directed_research/RePro
git add -A && git commit -m "msg" && git push origin main

# 2. BABEL: pull and submit (from login node, no interactive node needed)
ssh jerryy2@login.babel.cs.cmu.edu
cd ~/RePro && git pull origin main
mkdir -p ~/repro_logs
sbatch ~/RePro/scripts/babel_rl_train.sh

# 3. Monitor
squeue -u jerryy2                                           # job status
tail -f ~/repro_logs/repro-rl-<JOBID>.out                   # training log
```

First time on Babel: `cd ~ && git clone https://github.com/JerryYan123/RePro.git`

---

## Babel Paths

| Item | Location |
|---|---|
| Code | `~/RePro` |
| Conda env | `repro` (Python 3.11) |
| Model cache | `/data/user_data/jerryy2/.cache/huggingface` |
| Checkpoints | `/data/user_data/jerryy2/repro_checkpoints/` |
| Logs (Slurm) | `~/repro_logs/` (home, accessible everywhere) |

## GPU Selection (2025-02-25)

| GPU | VRAM | Nodes | Compatible | Notes |
|---|---|---|---|---|
| **A100 80GB** | 80 GB | 3 (24 GPUs) | **Yes** | **Best. Rarely queued for 8-GPU.** |
| L40S | 48 GB | ~50 (~400 GPUs) | Yes | Backup. More nodes, easier to schedule. |
| A6000 | 48 GB | ~28 (~180 GPUs) | Yes | Slower. |
| RTX PRO 6000 | 96 GB | 3 (24 GPUs) | **NO** | Blackwell sm_120, torch 2.6 incompatible. |

Scripts default to `A100_80GB`. To switch to L40S, change `--gres=gpu:A100_80GB:8` to `--gres=gpu:L40S:8` in the sbatch file.

### Check Availability

```bash
sinfo -p general -o "%G %t %D" --noheader | sort   # GPU summary (mix=has free, alloc=full)
squeue -p general -o "%b %t" --noheader | sort | uniq -c | sort -rn   # queue by GPU type
squeue -u jerryy2                                   # your jobs
```

---

## Full Pipeline

### Step 1: RL Train (sbatch, 8 GPUs)

```bash
sbatch ~/RePro/scripts/babel_rl_train.sh
```

Launches DataMan vLLM (GPU 0, port 8000) + Structure vLLM (GPU 1, port 8001), then GRPO training on GPUs 2-7. Saves checkpoints every 60 steps to `/data/user_data/jerryy2/repro_checkpoints/Qwen3-4B_grpo/`.

### Step 2: Infer (sbatch, 8 GPUs)

Before running: download DCLM-RefinedWeb shards to `/data/user_data/jerryy2/repro_data/dclm_refinedweb/`.

```bash
sbatch ~/RePro/scripts/babel_infer.sh                                    # default: checkpoint-1980, shards 0-7
sbatch ~/RePro/scripts/babel_infer.sh checkpoint-1200 /path/to/input 0 15  # custom
```

Input files should be named `shard_00000000_processed.jsonl.zstd` (or `.jsonl`). Output goes to `/data/user_data/jerryy2/repro_output/infer/`.

### Step 3: Filter (interactive, optional)

```bash
srun --partition=general --qos=normal --gres=gpu:A100_80GB:1 --cpus-per-task=16 --time=4:00:00 --pty bash
conda activate repro && cd ~/RePro/pretrain
export SPILL_LOCATION=/data/user_data/jerryy2/tmp/ray_spill && mkdir -p $SPILL_LOCATION
ray start --head --temp-dir=$SPILL_LOCATION
TMPDIR=/tmp PYTHONPATH=$(pwd) python ray_processing/process.py \
    --source_ref_paths exp_data/datasets/raw_sources/<YOUR_SOURCE>.json \
    --readable_name fasttext_repro \
    --output_dir /data/user_data/jerryy2/repro_output/filtered \
    --config_path baselines/baselines_configs/fasttext_filter.yaml --source_name cc
ray stop && rm -rf $SPILL_LOCATION
```

### Step 4: Pretrain (sbatch, 8 GPUs)

```bash
sbatch ~/RePro/scripts/babel_pretrain.sh
```

### Step 5: Eval (interactive, 8 GPUs)

```bash
srun --partition=general --qos=normal --gres=gpu:A100_80GB:8 --cpus-per-task=64 --time=4:00:00 --pty bash
conda activate repro && cd ~/RePro/pretrain
torchrun --nproc_per_node 8 --master_port 47763 eval/eval_openlm_ckpt.py \
    --donot-compute-perplexity \
    --checkpoint /data/user_data/jerryy2/repro_output/pretrain_logs/<METHOD>/checkpoints/epoch_<N>.pt \
    --model training/open_lm_configs/open_lm_1b_swiglutorch.json \
    --config /data/user_data/jerryy2/repro_output/pretrain_logs/<METHOD>/params.txt \
    --eval-yaml eval/mmlu_and_lowvar.yaml \
    --output-file results/<METHOD>/epoch_<N>/metrics.json --use-temp-working-dir
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA error: no kernel image` | Wrong GPU — use A100/L40S, NOT RTX PRO 6000 |
| Job stuck pending | `sinfo -p general` check; try L40S if A100 full |
| vLLM won't start | Check `~/RePro/rl/logs/*.log` |
| wandb errors | `wandb login` or remove from config yaml `report_to` |
| Models not found | `export HF_HOME=/data/user_data/jerryy2/.cache/huggingface` |

---

## Architecture

```
rl/src/open_r1/grpo_synthetic.py  # Training entry → loads 1000_sample_low_score.jsonl
rl/src/open_r1/rewards.py         # 5 rewards: DataMan(3), Format(1), BERTScore(1), Structure(1), Length(1)
rl/src/infer/run_infer.py         # Large-scale rephrasing inference
rl/recipes/Qwen3/grpo/config_4B.yaml  # Main training config
scripts/babel_*.sh                # Babel sbatch scripts
```

8 GPUs: GPU 0 = DataMan vLLM (port 8000), GPU 1 = Structure vLLM (port 8001), GPUs 2-7 = GRPO training (ZeRO-3).

Key config: lr=1e-6, beta=0.005, 2000 steps, 8 gens/prompt, batch 8, grad_accum 4, save every 60 steps.

---

## Improvement Ideas

1. **Keep/Rephrase/Discard Routing** — Skip good docs, drop unsalvageable ones. Modify prompt + add routing reward.
2. **Continuous Faithfulness Rewards** — Replace binary thresholds with smooth ramps. Replace 4B structure judge with small classifier.
3. **Auto-Generated Rewards** — AutoRule / RLCF / Auto-Rubric to discover criteria automatically.
4. **Deletion + Rephrasing** — ProX/RefineX deletion first, then RePro rephrasing.
5. **Reward Hacking Prevention** — Sweep beta, add retention constraint, enable repetition_penalty.
6. **Iterative Training** — Multiple RL rounds, curriculum learning, self-play.

## Key Files

| What | File |
|---|---|
| Rewards | `rl/src/open_r1/rewards.py` |
| Prompts / data | `rl/src/open_r1/grpo_synthetic.py` |
| Config | `rl/recipes/Qwen3/grpo/config_4B.yaml` |
| Babel scripts | `scripts/babel_rl_train.sh`, `babel_infer.sh`, `babel_pretrain.sh` |
