#!/usr/bin/env bash
#
# RunPod setup + run script for subterranean-agents.
#
# Usage (inside a RunPod pod, referenced by the pod spec's dockerArgs):
#   bash setup.sh generate   # synth data generation (CPU/API-bound; needs ANTHROPIC_API_KEY)
#   bash setup.sh train      # full fine-tuning (GPU)
#   bash setup.sh evaluate   # evaluation harness (CPU/API-bound; needs ANTHROPIC_API_KEY)
#   bash setup.sh serve      # OpenAI-compatible vLLM endpoint (GPU)
#
# Environment variables (set via the pod spec "env" block or the RunPod console):
#   SUBTERRANEAN_EXAMPLE     example name / build subdir (default: travel)
#   SUBTERRANEAN_BUILD_DIR   build dir (default: /workspace/build/$SUBTERRANEAN_EXAMPLE)
#   SUBTERRANEAN_BASE_MODEL  HF base model id (train)
#   SUBTERRANEAN_EPOCHS      training epochs (train)
#   SUBTERRANEAN_N           conversations (generate) / scenarios (evaluate)
#   SUBTERRANEAN_BUDGET      USD hard cap for generate/evaluate
#   SUBTERRANEAN_PORT        serve port (default: 8000)
#   ANTHROPIC_API_KEY        required for generate/evaluate
set -euo pipefail

STAGE="${1:-train}"
EXAMPLE="${SUBTERRANEAN_EXAMPLE:-travel}"
BUILD_DIR="${SUBTERRANEAN_BUILD_DIR:-/workspace/build/${EXAMPLE}}"
PORT="${SUBTERRANEAN_PORT:-8000}"

# Install the package with the extras the stage needs.
case "$STAGE" in
  train)    EXTRAS="train" ;;
  serve)    EXTRAS="serve" ;;
  generate) EXTRAS="report" ;;
  evaluate) EXTRAS="report" ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac

echo ">> Installing subterranean-agents[${EXTRAS}]"
pip install --no-cache-dir "subterranean-agents[${EXTRAS}]"

case "$STAGE" in
  generate)
    : "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set for generation}"
    subterranean generate "${BUILD_DIR}" \
      --n "${SUBTERRANEAN_N:-2000}" \
      --budget "${SUBTERRANEAN_BUDGET:-60}"
    ;;
  train)
    subterranean train "${BUILD_DIR}" \
      --base "${SUBTERRANEAN_BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
      --epochs "${SUBTERRANEAN_EPOCHS:-20}"
    ;;
  evaluate)
    : "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set for evaluation}"
    subterranean eval "${BUILD_DIR}" \
      --baselines in_context \
      --n "${SUBTERRANEAN_N:-200}" \
      --budget "${SUBTERRANEAN_BUDGET:-60}"
    ;;
  serve)
    subterranean serve "${BUILD_DIR}" --port "${PORT}" --host 0.0.0.0
    ;;
esac
