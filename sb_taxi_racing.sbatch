#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name taxi_ufr
#SBATCH --array 0-99
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem=4G
#SBATCH --time 02:00:00
#SBATCH --output experiment_logs/taxi_ufr/task-%A_%a.out

# Taxi-v4 HARD class-2, UNKNOWN fault rate, one of the experimental ufr diagnosers -- PARALLEL.
# 500 work-units (100 class-2 seeds x 5 visibilities) split into 100 array tasks. eps 0.04, matches
# the committed full-ufr baseline. MG_METHOD in {racing, v1, v1_freeze}. Results -> ufr_experiments/.
module load anaconda
source activate rldx_py311
cd ~/rldx_repo/rldx-sifu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
MG_METHOD=${MG_METHOD:-racing}
# optional free-form suffix (e.g. "_eps") so a variant of the same method lands in its own folder
RF="taxi_${MG_METHOD}_eps004${MG_OUT_SUFFIX:-}"
mkdir -p ufr_experiments/Taxi_v4/${RF}/logs

python main.py --mg_method ${MG_METHOD} --epsilon 0.04 \
    --mg_group ${SLURM_ARRAY_TASK_ID} --mg_num_groups 100 \
    -o ${RF}
