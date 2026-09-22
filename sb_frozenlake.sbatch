#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name fl_ufr
#SBATCH --array 0-499
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem=4G
#SBATCH --time 04:00:00
#SBATCH --output experiment_logs/frozenlake_PO/task-%A_%a.out

# FrozenLake WAY-2 fault benchmark, UNKNOWN fault rate, one of the experimental ufr diagnosers --
# FULL PARALLELISM: 100 maps x 5 visibilities x 1 fault rate = 500 work-units split into 500 array
# tasks (--fl_num_groups 500 -> ONE diagnosis per task). eps 0.04. MG_METHOD in {racing,v1,v1_freeze,v2}.
# Results -> ufr_experiments/FrozenLake_v1/<RF>/ . One fault rate per run (FL_FR, default 0.5).
module load anaconda
source activate rldx_py311
cd ~/rldx_repo/rldx-sifu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
MG_METHOD=${MG_METHOD:-v2}
FL_FR=${FL_FR:-0.5}
MG_EPS=${MG_EPS:-0.04}                    # epsilon knob (was hardcoded 0.04)
EPS_TAG=$(echo "$MG_EPS" | tr '.' '_')
# brute-force UNKNOWN study: MG_UFR=1 adds -ufr; MG_RESULTS_ROOT routes to a study folder; the driver
# then puts xlsx in <root>/FrozenLake_v1/<RF>/xlsx/ and we point logs at <root>/FrozenLake_v1/<RF>/logs/.
UFR_FLAG=""; if [ "${MG_UFR:-0}" = "1" ]; then UFR_FLAG="-ufr"; fi
ROOT_FLAG=""; if [ -n "${MG_RESULTS_ROOT:-}" ]; then ROOT_FLAG="--results_root ${MG_RESULTS_ROOT}"; fi
RF="${MG_RUN_FOLDER:-frozenlake_${MG_METHOD}_eps${EPS_TAG}${MG_OUT_SUFFIX:-}}"
mkdir -p experiment_logs/frozenlake_PO

python main.py --frozenlake --mg_method ${MG_METHOD} --epsilon ${MG_EPS} \
    ${UFR_FLAG} ${ROOT_FLAG} \
    --fl_group ${SLURM_ARRAY_TASK_ID} --fl_num_groups 500 \
    --fl_fault_rates ${FL_FR} \
    -o ${RF}
