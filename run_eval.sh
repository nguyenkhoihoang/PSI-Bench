#!/usr/bin/env bash
# run_eval.sh — Full PSI-bench evaluation pipeline
#
# Pipeline:
#   Step 1 (LLM)  : Classify valence   (runs in background)
#   Step 1 (LLM)  : Classify PTC codes (runs in background)
#   Step 2 (wait) : JS divergence for valence & PTC (needs Step 1 outputs)
#   Step 3        : Depressive linguistic markers
#   Step 3        : Message lengths
#   Step 3        : Lexical diversity (MTLD)
#   Step 4        : Aggregate all metrics
#
# Data source: HuggingFace dataset specified in eval.hf_dataset of CONFIG.
# The LLM classifier model/API are also set in CONFIG.
#
# Usage:
#   bash run_eval.sh [OPTIONS]
#
# Options:
#   --config PATH          Path to YAML config (default: configs/default.yaml)
#   --output-dir DIR       Root output directory  (default: output/eval_run)
#   --batch-size N         LLM classification batch size (default: 384)
#   --turn-threshold N     Max turns to analyse (default: 16)
#   --help                 Show this help and exit

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
CONFIG="configs/default.yaml"
OUTPUT_DIR="output/eval_run"
BATCH_SIZE=384
TURN_THRESHOLD=16

# ── argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)        CONFIG="$2";         shift 2 ;;
    --output-dir)    OUTPUT_DIR="$2";     shift 2 ;;
    --batch-size)    BATCH_SIZE="$2";     shift 2 ;;
    --turn-threshold) TURN_THRESHOLD="$2"; shift 2 ;;
    --help)
      sed -n '/^# Usage/,/^[^#]/p' "$0" | head -n -1
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── derived output paths ─────────────────────────────────────────────────────
VALENCE_DIR="${OUTPUT_DIR}/valence_analysis"
PTC_DIR="${OUTPUT_DIR}/ptc_analysis"
DEP_DIR="${OUTPUT_DIR}/depressive_markers"
LEN_DIR="${OUTPUT_DIR}/length_comparison"
LEX_DIR="${OUTPUT_DIR}/lexical_diversity"
AGG_DIR="${OUTPUT_DIR}/aggregate"

mkdir -p "$VALENCE_DIR" "$PTC_DIR" "$DEP_DIR" "$LEN_DIR" "$LEX_DIR" "$AGG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ────────────────────────────────────────────────────────────────────────────
# STEP 1 — LLM classification (valence + PTC) — run in parallel
# ────────────────────────────────────────────────────────────────────────────
log "Step 1: Starting LLM classifiers in background..."

python -m psibench.eval.valence_classification \
  --hf \
  --batch-size "$BATCH_SIZE" \
  --turn-threshold "$TURN_THRESHOLD" \
  --output-dir "$VALENCE_DIR" \
  --config "$CONFIG" \
  > "${OUTPUT_DIR}/valence_classification.out" 2>&1 &
VALENCE_PID=$!
log "  Valence classifier PID=$VALENCE_PID  (log: ${OUTPUT_DIR}/valence_classification.out)"

python -m psibench.eval.ptc.ptc_classification \
  --hf \
  --batch-size "$BATCH_SIZE" \
  --single-turn \
  --no-filler \
  --turn-threshold "$TURN_THRESHOLD" \
  --output-dir "$PTC_DIR" \
  --config "$CONFIG" \
  > "${OUTPUT_DIR}/ptc_classification.out" 2>&1 &
PTC_PID=$!
log "  PTC classifier      PID=$PTC_PID  (log: ${OUTPUT_DIR}/ptc_classification.out)"

# ── wait for both classifiers ────────────────────────────────────────────────
log "  Waiting for classifiers to finish..."
wait "$VALENCE_PID" || { log "ERROR: valence classification failed — check ${OUTPUT_DIR}/valence_classification.out"; exit 1; }
log "  Valence classification done."
wait "$PTC_PID" || { log "ERROR: PTC classification failed — check ${OUTPUT_DIR}/ptc_classification.out"; exit 1; }
log "  PTC classification done."

# ────────────────────────────────────────────────────────────────────────────
# STEP 2 — JS divergence (uses classification outputs)
# ────────────────────────────────────────────────────────────────────────────
log "Step 2: Computing JS divergence..."

python psibench/eval/js_divergence.py \
  --csv-file "${VALENCE_DIR}/valence_percentages_by_turn_t16_no_neutral.csv" \
  --turn-threshold "$TURN_THRESHOLD" \
  --output-dir "$VALENCE_DIR" \
  --label-column valence \
  --label-type valence
log "  Valence JS divergence done."

python psibench/eval/js_divergence.py \
  --csv-file "${PTC_DIR}/ptc_percentages_by_turn_t16_no_filler.csv" \
  --turn-threshold "$TURN_THRESHOLD" \
  --output-dir "$PTC_DIR" \
  --label-column category \
  --label-type ptc
log "  PTC JS divergence done."

# ────────────────────────────────────────────────────────────────────────────
# STEP 3 — Non-LLM metrics (can all run in parallel)
# ────────────────────────────────────────────────────────────────────────────
log "Step 3: Running non-LLM metrics in parallel..."

python psibench/eval/depressive_linguistic_markers.py \
  --output-dir "$DEP_DIR" \
  --metrics all \
  --token-scale 1000 \
  --combined \
  > "${OUTPUT_DIR}/depressive_markers.out" 2>&1 &
DEP_PID=$!

python -m psibench.eval.message_lengths \
  --config "$CONFIG" \
  --output-dir "$LEN_DIR" \
  > "${OUTPUT_DIR}/message_lengths.out" 2>&1 &
LEN_PID=$!

python psibench/eval/lexical_diversity.py \
  --output-dir "$LEX_DIR" \
  --min-tokens 100 \
  --sort-method base-model \
  > "${OUTPUT_DIR}/lexical_diversity.out" 2>&1 &
LEX_PID=$!

wait "$DEP_PID" || { log "ERROR: depressive markers failed — check ${OUTPUT_DIR}/depressive_markers.out"; exit 1; }
log "  Depressive linguistic markers done."
wait "$LEN_PID" || { log "ERROR: message lengths failed — check ${OUTPUT_DIR}/message_lengths.out"; exit 1; }
log "  Message lengths done."
wait "$LEX_PID" || { log "ERROR: lexical diversity failed — check ${OUTPUT_DIR}/lexical_diversity.out"; exit 1; }
log "  Lexical diversity done."

python psibench/eval/lexical_diversity.py \
  --from-csv "${LEX_DIR}/lexical_diversity_all_pairs.csv" \
  --output-dir "$LEX_DIR" \
  --min-tokens 100 \
  --sort-method base-model \
  > "${OUTPUT_DIR}/lexical_diversity_normalized.out" 2>&1
log "  Lexical diversity (normalized) done."

# ────────────────────────────────────────────────────────────────────────────
# STEP 4 — Aggregate
# ────────────────────────────────────────────────────────────────────────────
log "Step 4: Aggregating all metrics..."

python psibench/eval/aggregate.py \
  --mtld    "${LEX_DIR}/wasserstein_distances.csv" \
  --valence "${VALENCE_DIR}/valence_js_divergence_average.csv" \
  --ptc     "${PTC_DIR}/ptc_js_divergence_average.csv" \
  --verbosity "${LEN_DIR}/hf/comprehensive_metrics.csv" \
  --depressive "${DEP_DIR}/depressive_distance.csv" \
  --output-dir "$AGG_DIR"

log "Done! Aggregate results in: ${AGG_DIR}"
