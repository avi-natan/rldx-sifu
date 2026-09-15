#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name taxi_racing
#SBATCH --array 0-99
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem=4G
#SBATCH --time 01:00:00
#SBATCH --output ufr_experiments/Taxi_v4/taxi_racing_eps004/logs/task-%A_%a.out

# Taxi-v4 HARD class-2, UNKNOWN fault rate, CONFIDENCE-BOUNDED RACING diagnoser -- PARALLEL.
# 500 work-units (100 class-2 seeds x 5 visibilities) split into 100 array tasks (5 units each).
# eps 0.04, matches the committed full-ufr baseline. Results -> ufr_experiments/Taxi_v4/.
module load anaconda
source activate rldx_py311
cd ~/rldx_repo/rldx-sifu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p ufr_experiments/Taxi_v4/taxi_racing_eps004/logs

python main.py --racing --epsilon 0.04 \
    --mg_group ${SLURM_ARRAY_TASK_ID} --mg_num_groups 100 \
    -o taxi_racing_eps004
