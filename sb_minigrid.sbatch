#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name mgbench
#SBATCH --array 0-389
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem=4G
#SBATCH --time 00:40:00
#SBATCH --output experiment_logs/minigrid_PO/task-%A_%a.out

# MiniGrid PO fault benchmark (item 5). ONE submission = ONE (noise, fault_rate) run over the
# FIXED benchmark = 26 faults x 3 seeds x 5 visibilities = 390 work-units. MAX PARALLELISM: split
# into 390 array tasks (--mg_num_groups 390) -> exactly ONE diagnosis per task. NOISE selects env
# noise AND the matching trained policy together; FAULT_RATE is this run's fault-firing probability.
#
# Launch the grid by exporting MG_NOISE / MG_FR per submission; put xlsx AND logs in one per-noise
# folder by also overriding --output, e.g. for a full 0.7 noise level (all 3 fault rates):
#   R="experimental results/MiniGrid_Empty_16x16_v0/minigrid_bench_noise0_7"; mkdir -p "$R/logs"
#   for FR in 0.3 0.5 0.8; do
#     sbatch --export=ALL,MG_NOISE=0.7,MG_FR=$FR --output="$R/logs/fr${FR}-%A_%a.out" sb_minigrid.sbatch
#   done
# (xlsx land in $R via -o below; all 3 fault rates share the one 0.7 folder, distinct filenames.)
#
# CPU note: the diagnoser is single-threaded Monte-Carlo -> exactly ONE core per task.

module load anaconda
source activate rldx_py311

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MG_NOISE=${MG_NOISE:-0.7}
MG_FR=${MG_FR:-0.5}
NOISE_TAG=$(echo "$MG_NOISE" | tr '.' '_')
FR_TAG=$(echo "$MG_FR" | tr '.' '_')

mkdir -p experiment_logs/minigrid_PO

python main.py \
    --minigrid \
    --mg_group ${SLURM_ARRAY_TASK_ID} \
    --mg_num_groups 390 \
    --mg_noise ${MG_NOISE} \
    --mg_fault_rate ${MG_FR} \
    --epsilon 0.04 \
    -o minigrid_bench_noise${NOISE_TAG}
