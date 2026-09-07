#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name mgB
#SBATCH --array 0-99
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem=4G
#SBATCH --time 04:00:00
#SBATCH --output experiment_logs/minigrid_B/task-%A_%a.out

# MiniGrid partial-observability (approach B) fault benchmark, split into 100 instance
# groups (1 instance each). One xlsx per task; merge afterwards.
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

mkdir -p experiment_logs/minigrid_B

python main.py \
    --minigrid \
    --mg_group ${SLURM_ARRAY_TASK_ID} \
    --mg_num_groups 100 \
    --epsilon 0.04 \
    -o minigrid_B_INJFR_03-05-08_eps004
