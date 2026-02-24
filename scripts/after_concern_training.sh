#!/bin/bash
# Run this AFTER concern training finishes.
# 1. Set CONCERN_RUN to the new run dir, e.g. results/runs/distilroberta_base_lora_concern_20260223_165030
# 2. This script updates pipeline.yaml and runs evaluation + comparison.

set -e
cd "$(dirname "$0")/.."

# Find the most recent concern run if not set
CONCERN_RUN="${CONCERN_RUN:-$(ls -td results/runs/distilroberta_base_lora_concern_* 2>/dev/null | head -1)}"
if [ -z "$CONCERN_RUN" ] || [ ! -d "$CONCERN_RUN/checkpoint" ]; then
  echo "Usage: Set CONCERN_RUN to the new concern run path, e.g.:"
  echo "  CONCERN_RUN=results/runs/distilroberta_base_lora_concern_20260223_165030 ./scripts/after_concern_training.sh"
  echo "Or run from repo root after training: ls -td results/runs/distilroberta_base_lora_concern_* | head -1"
  exit 1
fi

CONCERN_CHECKPOINT="${CONCERN_RUN}/checkpoint"
echo "Using concern checkpoint: $CONCERN_CHECKPOINT"

# Update pipeline.yaml (replace concern_checkpoint line)
if [ -f configs/pipeline.yaml ]; then
  if grep -q "concern_checkpoint:" configs/pipeline.yaml; then
    sed -i.bak "s|concern_checkpoint:.*|concern_checkpoint: ${CONCERN_CHECKPOINT}|" configs/pipeline.yaml
    echo "Updated configs/pipeline.yaml (backup: configs/pipeline.yaml.bak)"
  fi
fi

# Run evaluation and save after-retrain metrics
echo "Running evaluation on test set..."
PYTHONPATH=. python scripts/evaluate_classifiers.py \
  --pipeline-config configs/pipeline.yaml \
  --data-config configs/data.yaml \
  --split test \
  --output results/classifier_metrics_after_retrain.json

# Compare baseline vs after
echo ""
echo "Comparison (baseline vs after concern retrain):"
python scripts/compare_classifier_metrics.py \
  results/classifier_metrics_baseline.json \
  results/classifier_metrics_after_retrain.json \
  --labels

echo ""
echo "Done."
