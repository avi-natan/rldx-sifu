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
RF="frozenlake_${MG_METHOD}_eps004${MG_OUT_SUFFIX:-}"
mkdir -p experiment_logs/frozenlake_PO ufr_experiments/FrozenLake_v1/${RF}

python main.py --frozenlake --mg_method ${MG_METHOD} --epsilon 0.04 \
    --fl_group ${SLURM_ARRAY_TASK_ID} --fl_num_groups 500 \
    --fl_fault_rates ${FL_FR} \
    -o ${RF}
