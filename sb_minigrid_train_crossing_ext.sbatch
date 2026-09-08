#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name mgxext
#SBATCH --array 0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=0
#SBATCH --mem=12G
#SBATCH --time 24:00:00
#SBATCH --output experiment_logs/minigrid_train/ext-%A_%a.out

# EXTENDED sweep: covers the training dimensions the first two sweeps did NOT touch, all aimed at a
# better-generalizing obs-policy on MiniGrid-SimpleCrossing-S11N2:
#   - n_epochs (PPO update epochs): FEWER (4) = less overfitting to each rollout batch.
#   - longer training (40M): see MORE distinct layouts; made safe with periodic checkpoints so a
#     run that doesn't reach the end still leaves a usable model (SB3 otherwise saves only at end).
#   - gamma 0.95: shorter effective horizon for the sparse terminal (goal) reward.
#   - linear lr decay: often a better final policy than a constant lr.
#   - ep4small: stack the two anti-overfit levers (n_epochs 4 + a small 32-wide net + entropy 0.03).
# 5 configs x 2 noises = 10 tasks (index = cfg*2 + noise_idx). Own out_dir per config. Ranks on
# held-out success_rate. Complements the sweep + gen sbatches.

module load anaconda
source activate rldx_py311
cd ~/rldx_repo/rldx-sifu
export OMP_NUM_THREADS=1
mkdir -p experiment_logs/minigrid_train

# config ->             ep4        long40     gamma95    lrdecay    ep4small
CFG_NAMES=(ep4 long40 gamma95 lrdecay ep4small)
NEPOCHS=(   4          10         10         10         4        )
TIMESTEPS=( 20000000   40000000   20000000   20000000   20000000 )
GAMMAS=(    0.99       0.99       0.95       0.99       0.99     )
LRSCHED=(   const      const      const      linear     const    )
LRS=(       2.5e-4     2.5e-4     2.5e-4     3e-4       2.5e-4   )
ENTS=(      0.02       0.02       0.02       0.02       0.03     )
NETS=(      64         64         64         64         32       )
NENVS=(     16         16         16         16         16       )
CKPT=(      0          2000000    0          0          0        )
NOISES=(0.5 0.7)

i=$SLURM_ARRAY_TASK_ID
cfg=$(( i / 2 ))
nz=$(( i % 2 ))
NOISE=${NOISES[$nz]}
NAME=${CFG_NAMES[$cfg]}

echo "ext task $i: cfg=$NAME noise=$NOISE steps=${TIMESTEPS[$cfg]} n_epochs=${NEPOCHS[$cfg]} gamma=${GAMMAS[$cfg]} lr=${LRS[$cfg]} lrsched=${LRSCHED[$cfg]} ent=${ENTS[$cfg]} net=${NETS[$cfg]} ckpt=${CKPT[$cfg]}"
python experiments_scripts/train_minigrid_ppo.py \
    --domain MiniGrid-SimpleCrossing-S11N2-v0 \
    --timesteps ${TIMESTEPS[$cfg]} \
    --n_envs ${NENVS[$cfg]} --use_subproc \
    --n_steps 512 \
    --n_epochs ${NEPOCHS[$cfg]} \
    --gamma ${GAMMAS[$cfg]} \
    --lr_schedule ${LRSCHED[$cfg]} \
    --ent_coef ${ENTS[$cfg]} \
    --learning_rate ${LRS[$cfg]} \
    --net_width ${NETS[$cfg]} \
    --checkpoint_freq ${CKPT[$cfg]} \
    --noise_prob ${NOISE} \
    --seed 0 \
    --device cpu \
    --render_gif \
    --out_dir runs/crossing_sweep/${NAME}
