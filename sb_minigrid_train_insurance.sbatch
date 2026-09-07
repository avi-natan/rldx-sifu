#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name mgins
#SBATCH --array 0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=8G
#SBATCH --time 06:00:00
#SBATCH --output experiment_logs/minigrid_train/ins-%A_%a.out

# Overnight INSURANCE sweep for the noise-0.7 (stochastic) policy so a strong candidate is
# guaranteed by morning and no retraining is needed tomorrow. All noise 0.7, 6M steps, GIF.
# Varies seed + two robustness knobs (wider net, higher exploration):
#   0-3: seeds 2,3,4,5   baseline (width 64,  ent 0.01)
#   4-5: seeds 2,3       WIDE net (width 256, ent 0.01)
#   6-7: seeds 2,3       HIGH exploration (width 64, ent 0.03)

module load anaconda
source activate rldx_py311

SEEDS=(2 3 4 5 2 3 2 3)
WIDTHS=(64 64 64 64 256 256 64 64)
ENTS=(0.01 0.01 0.01 0.01 0.01 0.01 0.03 0.03)
i=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$i]}; WIDTH=${WIDTHS[$i]}; ENT=${ENTS[$i]}

mkdir -p experiment_logs/minigrid_train

echo "insurance task $i: noise=0.7 seed=$SEED width=$WIDTH ent=$ENT"
python experiments_scripts/train_minigrid_ppo.py \
    --domain MiniGrid-Empty-16x16-v0 \
    --timesteps 6000000 \
    --n_envs 8 \
    --use_subproc \
    --n_steps 512 \
    --noise_prob 0.7 \
    --seed ${SEED} \
    --ent_coef ${ENT} \
    --net_width ${WIDTH} \
    --render_gif \
    --out_dir runs/minigrid_insurance/task${i}_seed${SEED}_w${WIDTH}_ent${ENT}
