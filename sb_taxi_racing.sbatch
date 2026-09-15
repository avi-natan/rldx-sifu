#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name taxi_racing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem=4G
#SBATCH --time 12:00:00
#SBATCH --output ufr_experiments/Taxi_v4/taxi_racing_eps004/logs/run-%j.out

# Taxi-v4 HARD class-2, UNKNOWN fault rate, solved by the CONFIDENCE-BOUNDED RACING diagnoser.
# Taxi is the stress test: ~92% of instances are near-ties, so racing's decisions rarely fire ->
# this measures how the ~10x-speed / small-rank-cost tradeoff behaves in the hardest regime.
# One task = all 100 class-2 seeds x 5 visibilities at eps 0.04 (matches the committed full-ufr
# baseline in experimental results/Taxi_v4/UNknown_fr_experiments_eps_sweep). Results -> ufr_experiments/.
module load anaconda
source activate rldx_py311
cd ~/rldx_repo/rldx-sifu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p ufr_experiments/Taxi_v4/taxi_racing_eps004/logs

python main.py --racing --epsilon 0.04 -o taxi_racing_eps004
