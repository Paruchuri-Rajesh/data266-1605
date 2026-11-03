#!/usr/bin/env bash
# Part C (OPTIONAL per the HW4.5 clarifications) on a two-GPU pairing, e.g. Kaggle "GPU T4 x2". Appends to RUN_LOG.txt.
# Usage: ./run_ddp.sh <pair-label>      e.g. kaggle_t4x2, lab_4090x2
# Skipping it is fine: make_report.py then writes "Part C skipped (optional)" in METRICS.md.
set -euo pipefail
PAIR=${1:?pair label}
SLICE=${SLICE:-605}   # same data as HW4.a
SEED=${SEED:-1605}
ARGS="--card $PAIR --slice-start $SLICE --seed $SEED --data ${DATA:-data/Tweets.csv} ${BATCHES:+--batches $BATCHES}"
{
  echo "=== [$(date '+%F %T')] [$PAIR] Part C"
  ./capture_env.sh "$PAIR"
  torchrun --standalone --nproc_per_node 1 ddp_bert.py $ARGS
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH torchrun --standalone --nproc_per_node 2 ddp_bert.py $ARGS
} 2>&1 | tee -a RUN_LOG.txt
