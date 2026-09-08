#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name mgtrain
#SBATCH --array 0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=8G
#SBATCH --time 10:00:00
#SBATCH --output experiment_logs/minigrid_train/task-%A_%a.out

# Train an OBSERVATION-input PPO policy for MiniGrid-Empty-16x16 (item 6).
# Sweep hedges over dynamics noise x seed so at least one good policy is ready by morning:
#   task 0: noise 1.0 (deterministic) seed 0     task 2: noise 0.7 seed 0
#   task 1: noise 1.0 (deterministic) seed 1     task 3: noise 0.7 seed 1
# PPO with vectorized envs IS parallel work, so 8 cores/task is legitimate (unlike the
# single-threaded diagnoser). SubprocVecEnv uses the 8 cores.

module load anaconda
source activate rldx_py311

NOISES=(1.0 1.0 0.7 0.7)
SEEDS=(0 1 0 1)
NOISE=${NOISES[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

mkdir -p experiment_logs/minigrid_train

echo "task ${SLURM_ARRAY_TASK_ID}: noise=${NOISE} seed=${SEED}"
python experiments_scripts/train_minigrid_ppo.py \
    --domain MiniGrid-Empty-16x16-v0 \
    --timesteps 6000000 \
    --n_envs 8 \
    --use_subproc \
    --n_steps 512 \
    --noise_prob ${NOISE} \
    --seed ${SEED} \
    --out_dir runs/minigrid_train
