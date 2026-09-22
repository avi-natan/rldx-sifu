#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name taxi_ufr
#SBATCH --array 0-499
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
MG_EPS=${MG_EPS:-0.04}                    # epsilon knob (was hardcoded 0.04)
EPS_TAG=$(echo "$MG_EPS" | tr '.' '_')
# brute-force UNKNOWN study: MG_UFR=1 adds -ufr; MG_RESULTS_ROOT routes to a study folder
# (driver then writes xlsx to <root>/Taxi_v4/<RF>/xlsx/). MG_FIXED_TRIES is read straight from the
# job env by the diagnoser (fixed sims-per-estimate) -- no flag needed here, just pass via --export.
UFR_FLAG=""; if [ "${MG_UFR:-0}" = "1" ]; then UFR_FLAG="-ufr"; fi
ROOT_FLAG=""; if [ -n "${MG_RESULTS_ROOT:-}" ]; then ROOT_FLAG="--results_root ${MG_RESULTS_ROOT}"; fi
# optional free-form suffix (e.g. "_eps") so a variant of the same method lands in its own folder
RF="${MG_RUN_FOLDER:-taxi_${MG_METHOD}_eps${EPS_TAG}${MG_OUT_SUFFIX:-}}"
mkdir -p experiment_logs/taxi_ufr

python main.py --mg_method ${MG_METHOD} --epsilon ${MG_EPS} \
    ${UFR_FLAG} ${ROOT_FLAG} \
    --mg_group ${SLURM_ARRAY_TASK_ID} --mg_num_groups 500 \
    -o ${RF}
