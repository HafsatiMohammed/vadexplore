#!/bin/bash
# Overnight batch: train ALL FOUR models from scratch, then evaluate everything.
# Clean models use configs/train_clean.yaml (augment off).
# Augmented models use configs/train_aug.yaml (augment on).
# No config toggling: each run points at the right explicit config.
# Logs to logs/. Continues past failures (no set -e).

set -u
mkdir -p logs runs/robustness explore_out/figures

TS=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="logs/overnight_${TS}.log"
CLEAN_CFG="configs/train_clean.yaml"
AUG_CFG="configs/train_aug.yaml"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"; }

DEVICE="${1:-mps}"
log "=== Overnight run starting on device=$DEVICE ==="
log "clean config: $(grep -m1 '^  enabled:' $CLEAN_CFG)"
log "aug config:   $(grep -m1 '^  enabled:' $AUG_CFG)"

# ---------- CLEAN MODELS (train_clean.yaml, augment off) ----------
log ">>> [1/4] CLEAN BiGRU (bigru_bridged)"
python vadexplore/train.py --config "$CLEAN_CFG" \
  --name bigru_bridged --core bigru --label bridged \
  --epochs 30 --device "$DEVICE" --no-progress \
  >> "logs/train_bigru_bridged_${TS}.log" 2>&1
log "    bigru_bridged done (exit $?)"

log ">>> [2/4] CLEAN causal (attn_pw_1s)"
python vadexplore/train.py --config "$CLEAN_CFG" \
  --name attn_pw_1s --core causal_attn --label bridged \
  --past-window-frames 50 --lookahead-frames 5 \
  --epochs 30 --device "$DEVICE" --no-progress \
  >> "logs/train_attn_pw_1s_${TS}.log" 2>&1
log "    attn_pw_1s done (exit $?)"

# ---------- AUGMENTED MODELS (train_aug.yaml, augment on) ----------
log ">>> [3/4] AUGMENTED BiGRU (bigru_augmented)"
python vadexplore/train.py --config "$AUG_CFG" \
  --name bigru_augmented --core bigru --label bridged \
  --epochs 30 --device "$DEVICE" --no-progress \
  >> "logs/train_bigru_augmented_${TS}.log" 2>&1
log "    bigru_augmented done (exit $?)"

log ">>> [4/4] AUGMENTED causal (attn_augmented)"
python vadexplore/train.py --config "$AUG_CFG" \
  --name attn_augmented --core causal_attn --label bridged \
  --past-window-frames 50 --lookahead-frames 5 \
  --epochs 30 --device "$DEVICE" --no-progress \
  >> "logs/train_attn_augmented_${TS}.log" 2>&1
log "    attn_augmented done (exit $?)"

# ---------- CLEAN-TEST EVAL (all four) ----------
for run in bigru_bridged attn_pw_1s bigru_augmented attn_augmented; do
  log ">>> Evaluating $run on clean test"
  python vadexplore/evaluate.py --run "runs/$run" --split test --device "$DEVICE" \
    >> "logs/eval_${TS}.log" 2>&1
  log "    eval $run done (exit $?)"
done

# ---------- ROBUSTNESS MATRICES ----------
log ">>> Robustness matrix: BiGRU clean vs augmented"
python scripts/robustness_eval.py \
  --clean runs/bigru_bridged --augmented runs/bigru_augmented \
  --split test --device "$DEVICE" \
  --out runs/robustness/matrix_bigru.json \
  --figure explore_out/figures/robustness_bigru.png \
  >> "logs/robustness_${TS}.log" 2>&1
log "    bigru matrix done (exit $?)"

log ">>> Robustness matrix: causal clean vs augmented"
python scripts/robustness_eval.py \
  --clean runs/attn_pw_1s --augmented runs/attn_augmented \
  --split test --device "$DEVICE" \
  --out runs/robustness/matrix_causal.json \
  --figure explore_out/figures/robustness_causal.png \
  >> "logs/robustness_${TS}.log" 2>&1
log "    causal matrix done (exit $?)"

log "=== COMPLETE. Check logs/ and runs/robustness/ ==="
