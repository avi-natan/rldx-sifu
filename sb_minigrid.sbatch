#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name mgB
#SBATCH --array 0-499
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem=4G
#SBATCH --time 00:30:00
#SBATCH --output experiment_logs/minigrid_PO/task-%A_%a.out

# MiniGrid partial-observability fault benchmark. Work = 100 instances x 5
# visibilities x 3 fault rates = 1500 units (one diagnosis each, ~1 min). Split into 500
# groups -> 3 units (~3 min) per task. One xlsx per task; merge afterwards.
# (Check the cluster's MaxArraySize; if < 500, raise --mg_num_groups is not needed, just
#  lower the array size and units-per-task rise accordingly.)
#
# CPU note: the diagnoser is SERIAL (single-threaded Monte-Carlo). The cluster flagged our
# earlier 2-core jobs as ~50% utilised, so we request exactly ONE core per task here and pin
# the math libs to 1 thread. This queues faster and packs more array tasks.

module load anaconda
source activate rldx_py311

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p experiment_logs/minigrid_PO

python main.py \
    --minigrid \
    --mg_group ${SLURM_ARRAY_TASK_ID} \
    --mg_num_groups 500 \
    --epsilon 0.04 \
    -o minigrid_PO_INJFR_03-05-08_eps004
