#!/usr/bin/env bash
# GPU training launcher for DispatchPPO (the paper's training runs).
#
# Usage (from the repository root):
#   bash scripts/train_gpu.sh SETTING [OUT_DIR]
#   SETTING in g5n1 g5n2 g5n5 g10n2 g10n5 g10n10 g30n5 g30n10 g30n30
#
# Environment overrides (defaults reproduce the paper's training command):
#   CU        uncertainty increment C^U            (default: per-setting table below)
#   SHIFTS    training mean shifts                 (default 1,2,3)
#   UPDATES   PPO updates                          (default 10000)
#   VALEVERY  validation period in updates          (default 100)
#   SEED      training seed                        (default 101)
#   FRAC      per-process CUDA memory fraction     (default 0.2)
#   DEVICE    torch device                         (default cuda)
#   SWMODE    swanlab mode: disabled|cloud|local   (default disabled; cloud needs --swanlab-project in EXTRA)
#   EXTRA     extra trainer flags appended verbatim (e.g. "--resume-actor-optimizer --resume-critic")
#
# The run writes args.txt, manifest.json, history.jsonl and the checkpoints
# (initial.pt, latest.pt, candidate_XXXX.pt, best_XXXX.pt / step_XXXX.pt,
# best_train_hit.pt) into OUT_DIR.
set -euo pipefail

SETTING=${1:?usage: train_gpu.sh SETTING [OUT_DIR]}
cd "$(dirname "$0")/.."

# Per-setting table: G N A B H_hat CU_default
#   A      affected-set size of the validation protocol (training uses |B| = N)
#   B      parallel environments
#   H_hat  pseudo-threshold of the periodic validation used to select checkpoints
setting_row() {
  case "$1" in
    g5n1)   echo "5 1 2 512 5.556 0.01" ;;
    g5n2)   echo "5 2 2 512 7.752 0.01" ;;
    g5n5)   echo "5 5 2 512 14.9768 0.01" ;;
    g10n2)  echo "10 2 5 512 10.8856 0.01" ;;
    g10n5)  echo "10 5 5 512 16.6968 0.01" ;;
    g10n10) echo "10 10 5 512 25.704 0.01" ;;
    g30n5)  echo "30 5 10 256 13.9776 0.001" ;;
    g30n10) echo "30 10 10 256 24.2136 0.001" ;;
    g30n30) echo "30 30 10 256 54.9592 0.001" ;;
    *) echo "unknown setting $1" >&2; return 1 ;;
  esac
}
read G N A B H CU_DEFAULT <<< "$(setting_row "$SETTING")"

CU=${CU:-$CU_DEFAULT}
SHIFTS=${SHIFTS:-1,2,3}
UPDATES=${UPDATES:-10000}
VALEVERY=${VALEVERY:-100}
SEED=${SEED:-101}
FRAC=${FRAC:-0.2}
DEVICE=${DEVICE:-cuda}
SWMODE=${SWMODE:-disabled}

TAG=${SETTING}_DPPO_cu${CU}_u${UPDATES}_s${SEED}
OUT=${2:-results/$TAG}
mkdir -p "$OUT"

export PYTHONPATH=${PYTHONPATH:-$PWD}
python -m training.make_scratch_checkpoint --grid-size "$G" --num-uavs "$N" --output "$OUT/scratch_init.pt" --seed "$SEED" > /dev/null

ARGS="--grid-size $G --num-uavs $N --num-anomalies $N --num-envs $B --uncertainty-increment $CU \
  --init-checkpoint $OUT/scratch_init.pt --ppo-updates $UPDATES --learning-rate 3e-4 --lr-final 3e-5 --validate-every $VALEVERY \
  --ppo-epochs 2 --num-mini-batches 16 --clip-param 0.2 --gamma 0.99 --entropy-coef 0.001 --target-kl 0.02 \
  --episode-steps 200 --change-low 5 --change-high 50 --training-shifts $SHIFTS --label-free --decode-order sorted \
  --validate-fixed-threshold $H --validate-replications 1000 --validate-chunk-size 250 --checkpoint-interval 20 \
  --seed $SEED --validate-num-anomalies $A --value-loss-form huber --huber-delta 10 --cuda-memory-fraction $FRAC ${EXTRA:-}"
echo "$ARGS" > "$OUT/args.txt"
echo "launching $TAG (G=$G N=$N, train |B|=$N, val |B|=$A, H_hat=$H, C^U=$CU) -> $OUT"
exec python -m training.train_dispatch_gpu $ARGS --output-dir "$OUT" --device "$DEVICE" --swanlab-mode "$SWMODE"
