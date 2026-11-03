#!/usr/bin/env bash
# Every single-GPU measurement for one card (HW4.a, Parts A, B, D). Appends everything to RUN_LOG.txt.
# Usage: ./run_card.sh <card-label> [part ...]     parts: env hw4a serve partA partB partD
# One GPU is enough (HW4.5 clarifications); Part C is optional and lives in run_ddp.sh.
set -euo pipefail
CARD=${1:?card label, e.g. rtx4090, colab_t4, kaggle_t4}; shift
PARTS=${*:-env hw4a serve partA partB partD}
# Same data as HW4.a: SID4=1605 -> SLICE=605, SEED=1605. Override only if HW4.a used something else.
SLICE=${SLICE:-605}
SEED=${SEED:-1605}
DATA=${DATA:-data/Tweets.csv}
# Part D KV test: the 7B model needs a 24 GB+ card; on a 16 GB T4 use a model that fits
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
if [ "${VRAM_MB:-0}" -ge 20000 ]; then DEFAULT_KV=Qwen/Qwen2.5-7B-Instruct; else DEFAULT_KV=Qwen/Qwen2.5-1.5B-Instruct; fi
KV_MODEL=${KV_MODEL:-$DEFAULT_KV}
COMMON="--slice-start $SLICE --seed $SEED --data $DATA"
log() { echo "=== [$(date '+%F %T')] [$CARD] $*"; }
{
log "SLICE=$SLICE SEED=$SEED DATA=$DATA KV_MODEL=$KV_MODEL"
for p in $PARTS; do
  log "start $p"
  case $p in
    env)   ./capture_env.sh "$CARD" ;;
    hw4a)  python train_bert.py $COMMON --tag hw4a --write-lock true --save-dir models/bert_hw4a_$CARD --out results/$CARD/hw4a.json ;;
    serve) python serve_bench.py demo --card "$CARD" --prompt "${PROMPT:-Explain what a KV cache is to a first-year student.}"
           python serve_bench.py hf   --card "$CARD"
           python serve_bench.py vllm --card "$CARD" --batches 1 8 32 ;;
    partA) python sweep.py partA --card "$CARD" $COMMON ;;
    partB) python sweep.py partB --card "$CARD" $COMMON ;;
    partD) python serve_bench.py hf   --card "$CARD"
           python serve_bench.py vllm --card "$CARD" --batches 1 8 32
           python serve_bench.py kv   --card "$CARD" --model "$KV_MODEL" --kv-batch 8 ;;
    *) echo "unknown part $p"; exit 1 ;;
  esac
  log "end $p"
done
} 2>&1 | tee -a RUN_LOG.txt
