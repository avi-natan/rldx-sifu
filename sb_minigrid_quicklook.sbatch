#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name mgquick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=8G
#SBATCH --time 00:25:00
#SBATCH --output experiment_logs/minigrid_train/quicklook-%j.out

# A fast (~7 min) noise-0.7 policy so we can SEE it execute (renders a greedy-episode GIF)
# without waiting for the full 6M-step sweep. Not the final policy; just a quick look.

module load anaconda
source activate rldx_py311

mkdir -p experiment_logs/minigrid_train

python experiments_scripts/train_minigrid_ppo.py \
    --domain MiniGrid-Empty-16x16-v0 \
    --timesteps 800000 \
    --n_envs 8 \
    --use_subproc \
    --n_steps 512 \
    --noise_prob 0.7 \
    --seed 0 \
    --render_gif \
    --out_dir runs/minigrid_quicklook
