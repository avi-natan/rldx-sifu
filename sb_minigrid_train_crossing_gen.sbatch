#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name mgxgen
#SBATCH --array 0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=0
#SBATCH --mem=12G
#SBATCH --time 20:00:00
#SBATCH --output experiment_logs/minigrid_train/gen-%A_%a.out

# GENERALIZATION-FOCUSED sweep for the obs-policy on MiniGrid-SimpleCrossing-S11N2 (per-seed layouts).
# Complements sb_minigrid_train_crossing_sweep.sbatch by pushing the levers that most drive
# generalization across the random layouts (the earlier sweep under-used these):
#   - DIVERSITY per gradient update: many parallel envs (16-32) => each update sees many DISTINCT
#     layouts, so the net can't memorize one grid.
#   - REGULARIZATION: a SMALL net (32) + higher entropy (0.03) resist memorizing specific layouts.
# All 20M steps so every run reliably FINISHES and saves (SB3 saves only at the end). Eval scores 200
# HELD-OUT layouts, so success_rate is the generalization metric to rank on.
#
# 3 configs x 2 noises = 6 tasks (index = cfg*2 + noise_idx). Own out_dir per config (no collisions).

module load anaconda
source activate rldx_py311
cd ~/rldx_repo/rldx-sifu
export OMP_NUM_THREADS=1
mkdir -p experiment_logs/minigrid_train

# config ->            manyenv  smallnet  bigdiv
CFG_NAMES=(manyenv smallnet bigdiv)
LRS=(     2.5e-4   2.5e-4    2.5e-4 )
ENTS=(    0.02     0.03      0.025  )
NETS=(    64       32        64     )
NSTEPS=(  256      512       512    )
NENVS=(   32       16        24     )
NOISES=(0.5 0.7)

i=$SLURM_ARRAY_TASK_ID
cfg=$(( i / 2 ))
nz=$(( i % 2 ))
NOISE=${NOISES[$nz]}
NAME=${CFG_NAMES[$cfg]}

echo "gen task $i: cfg=$NAME noise=$NOISE lr=${LRS[$cfg]} ent=${ENTS[$cfg]} net=${NETS[$cfg]} n_steps=${NSTEPS[$cfg]} n_envs=${NENVS[$cfg]}"
python experiments_scripts/train_minigrid_ppo.py \
    --domain MiniGrid-SimpleCrossing-S11N2-v0 \
    --timesteps 20000000 \
    --n_envs ${NENVS[$cfg]} --use_subproc \
    --n_steps ${NSTEPS[$cfg]} \
    --ent_coef ${ENTS[$cfg]} \
    --learning_rate ${LRS[$cfg]} \
    --net_width ${NETS[$cfg]} \
    --noise_prob ${NOISE} \
    --seed 0 \
    --device cpu \
    --render_gif \
    --out_dir runs/crossing_sweep/${NAME}
