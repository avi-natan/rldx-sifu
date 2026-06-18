#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-06:00:00
#SBATCH --job-name rldx_taxi_train
#SBATCH --output rldx_taxi_train-%J.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --tmp=10G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahmade@post.bgu.ac.il

# Taxi-v4 PPO training search on the rainy env (CPU job, rldx_py311).
# Trains several recipes, evaluates each with eval_taxi_policy.py (success@20),
# writes models to /scratch, copies a results summary + best models back to
# ~/rldx_taxi_results/<jobid> (OUTSIDE the repo, so the git checkout stays clean).
# Does NOT touch environments/Taxi_v4/models/PPO/Taxi_v4__PPO.zip (gated promotion).

echo "JOBID=$SLURM_JOBID NODE=$SLURM_JOB_NODELIST START=$(date)"
module load anaconda
source activate rldx_py311
cd ~/rldx_repo/rldx-sifu || exit 1
git log --oneline -1

SCRATCH="${SLURM_SCRATCH_DIR:-/scratch/$USER/$SLURM_JOBID}"
RUNS="$SCRATCH/runs"
mkdir -p "$RUNS"
RESULTS="$SCRATCH/results.csv"
echo "config,model,success_line" > "$RESULTS"

run_and_eval () {
  name="$1"; shift
  echo "==================== TRAIN: $name ===================="
  python -u train_taxi_v4_ppo.py --save_dir "$RUNS" --tag "$name" --n_envs 8 "$@"
  for m in best_model final_model; do
    mp="$RUNS/$name/$m.zip"
    if [ -f "$mp" ]; then
      echo "-------- EVAL: $name/$m --------"
      line=$(python -u eval_taxi_policy.py --model "$mp" --n_seeds 20 | grep "success@")
      echo "RESULT  $name  $m  ->  $line"
      echo "$name,$m,$line" >> "$RESULTS"
    fi
  done
}

# --- Phase 1: deterministic base (proves recipe end-to-end; curriculum seed) ---
run_and_eval det --rainy_probability 1.0 --timesteps 400000 --ent_coef 0.01

# --- Phase 2A: rainy from scratch, moderate exploration ---
run_and_eval rainy_scratch --rainy_probability 0.7 --timesteps 1000000 --ent_coef 0.01

# --- Phase 2A': rainy from scratch, higher exploration ---
run_and_eval rainy_highent --rainy_probability 0.7 --timesteps 1000000 --ent_coef 0.05

# --- Phase 2B: curriculum — fine-tune the deterministic base on rain ---
if [ -f "$RUNS/det/best_model.zip" ]; then
  run_and_eval curriculum --init_from "$RUNS/det/best_model.zip" \
      --rainy_probability 0.7 --timesteps 600000 --ent_coef 0.02
else
  echo "SKIP curriculum: no det/best_model.zip"
fi

# --- copy results + best/final models back (outside the repo) ---
DEST="$HOME/rldx_taxi_results/$SLURM_JOBID"
mkdir -p "$DEST"
cp "$RESULTS" "$DEST/" 2>/dev/null
for d in "$RUNS"/*/; do
  n=$(basename "$d")
  [ -f "$d/best_model.zip" ]  && cp "$d/best_model.zip"  "$DEST/${n}_best.zip"
  [ -f "$d/final_model.zip" ] && cp "$d/final_model.zip" "$DEST/${n}_final.zip"
done

echo "==================== SUMMARY ===================="
cat "$RESULTS"
echo "Models + results copied to: $DEST"
echo "DONE=$(date)"
