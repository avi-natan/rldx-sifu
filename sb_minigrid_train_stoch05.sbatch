#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name mg05
#SBATCH --array 0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=8G
#SBATCH --time 06:00:00
#SBATCH --output experiment_logs/minigrid_train/stoch05-%A_%a.out

# A middle stochasticity variant: train the obs-input policy UNDER noise_prob=0.5
# ("0.5 success" -> 50% intended executes, ~33% genuinely-different). Intended as a
# diagnosable-but-more-diverse operating point between 0.7 (diagnosable) and 0.3 (near-random).
# Two seeds, GIFs.

module load anaconda
source activate rldx_py311

SEED=$SLURM_ARRAY_TASK_ID
mkdir -p experiment_logs/minigrid_train

echo "stoch05 task $SLURM_ARRAY_TASK_ID: noise=0.5 seed=$SEED"
python experiments_scripts/train_minigrid_ppo.py \
    --domain MiniGrid-Empty-16x16-v0 \
    --timesteps 6000000 \
    --n_envs 8 \
    --use_subproc \
    --n_steps 512 \
    --noise_prob 0.5 \
    --seed ${SEED} \
    --render_gif \
    --out_dir runs/minigrid_stoch05
