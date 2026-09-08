#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name mgxtrain
#SBATCH --array 0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=8G
#SBATCH --time 12:00:00
#SBATCH --output experiment_logs/minigrid_train/crossing-%A_%a.out

# Train a GENERALIZING obs-input PPO policy for MiniGrid-SimpleCrossing-S11N2 (11x11, 2 wall
# barriers whose gaps move every seed). Policy input = the OBSERVATION (flatten(image)/255 ++
# one_hot(direction) = 151-dim, size-independent), NOT the true state. Each training episode resets
# to a fresh random layout, so the policy must LEARN to find gaps from the egocentric view.
# Sweep: noise {0.7, 0.5} x seed {0, 1}. 10M steps, 8 cores/task (SubprocVecEnv), GIF.

module load anaconda
source activate rldx_py311

NOISES=(0.7 0.7 0.5 0.5)
SEEDS=(0 1 0 1)
i=$SLURM_ARRAY_TASK_ID
NOISE=${NOISES[$i]}; SEED=${SEEDS[$i]}
mkdir -p experiment_logs/minigrid_train

echo "crossing task $i: domain=SimpleCrossing-S11N2 noise=$NOISE seed=$SEED"
python experiments_scripts/train_minigrid_ppo.py \
    --domain MiniGrid-SimpleCrossing-S11N2-v0 \
    --timesteps 10000000 \
    --n_envs 8 \
    --use_subproc \
    --n_steps 512 \
    --noise_prob ${NOISE} \
    --seed ${SEED} \
    --render_gif \
    --out_dir runs/crossing_s11n2_train
