#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name mg03
#SBATCH --array 0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=8G
#SBATCH --time 06:00:00
#SBATCH --output experiment_logs/minigrid_train/stoch03-%A_%a.out

# A MUCH more stochastic variant: train the obs-input policy UNDER noise_prob=0.3
# ("0.3 success" -> ~53% intended / ~47% different). Heavier noise = far more trajectory
# diversity + harder diagnosis, since start/goal/policy are otherwise fixed. Two seeds, GIFs.
# Expect greedy success < 1.0 here (that's fine: trajectories stay diagnosable).

module load anaconda
source activate rldx_py311

SEED=$SLURM_ARRAY_TASK_ID
mkdir -p experiment_logs/minigrid_train

echo "stoch03 task $SLURM_ARRAY_TASK_ID: noise=0.3 seed=$SEED"
python experiments_scripts/train_minigrid_ppo.py \
    --domain MiniGrid-Empty-16x16-v0 \
    --timesteps 6000000 \
    --n_envs 8 \
    --use_subproc \
    --n_steps 512 \
    --noise_prob 0.3 \
    --seed ${SEED} \
    --render_gif \
    --out_dir runs/minigrid_stoch03
