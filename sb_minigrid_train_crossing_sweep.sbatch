#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name mgxsweep
#SBATCH --array 0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=0
#SBATCH --mem=10G
#SBATCH --time 20:00:00
#SBATCH --output experiment_logs/minigrid_train/sweep-%A_%a.out

# Overnight HYPERPARAMETER SWEEP for a GENERALIZING obs-input PPO on
# MiniGrid-SimpleCrossing-S11N2 (11x11, 2 wall barriers whose gaps move every seed). Goal: the best
# policy we can get at BOTH noise 0.5 and 0.7. Policy input = the OBSERVATION (flatten(image)/255 ++
# one_hot(direction) = 151-dim); training auto-resets to a FRESH random layout each episode and eval
# scores 200 HELD-OUT layouts (seeds 10000+), so success_rate is a true generalization metric.
#
# 5 configs x 2 noises = 10 tasks (index = cfg*2 + noise_idx). Each config writes to its OWN out_dir
# so the two noises (distinct tags ..._noise0.5 / ..._noise0.7) never collide. 20M steps each.
# SB3 forces MlpPolicy to CPU (a GPU A/B proved requested=cuda -> resolved=cpu), so this is CPU-only;
# the lever is cores/envs. Complements the already-running default job 21143052 (10M, seeds 0/1).

module load anaconda
source activate rldx_py311
cd ~/rldx_repo/rldx-sifu
export OMP_NUM_THREADS=1
mkdir -p experiment_logs/minigrid_train

# config index ->              base    ent03   net128  long    hi_lr
CFG_NAMES=(base ent03 net128 long hi_lr)
LRS=(     2.5e-4  2.5e-4  2.5e-4  3e-4    5e-4  )
ENTS=(    0.01    0.03    0.01    0.01    0.02  )
NETS=(    64      64      128     128     128   )
NSTEPS=(  512     512     512     1024    512   )
NENVS=(   8       8       8       16      8     )
NOISES=(0.5 0.7)

i=$SLURM_ARRAY_TASK_ID
cfg=$(( i / 2 ))
nz=$(( i % 2 ))
NOISE=${NOISES[$nz]}
NAME=${CFG_NAMES[$cfg]}

echo "sweep task $i: cfg=$NAME noise=$NOISE lr=${LRS[$cfg]} ent=${ENTS[$cfg]} net=${NETS[$cfg]} n_steps=${NSTEPS[$cfg]} n_envs=${NENVS[$cfg]}"
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
