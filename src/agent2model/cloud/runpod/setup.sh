#!/usr/bin/env bash
#
# RunPod setup + run script for agent2model.
#
# Usage (inside a RunPod pod, referenced by the pod spec's dockerArgs):
#   bash setup.sh generate   # synth data generation (CPU/API-bound; needs ANTHROPIC_API_KEY)
#   bash setup.sh train      # full fine-tuning (GPU)
#   bash setup.sh evaluate   # evaluation harness (CPU/API-bound; needs ANTHROPIC_API_KEY)
#   bash setup.sh serve      # OpenAI-compatible vLLM endpoint (GPU)
#
# Environment variables (set via the pod spec "env" block or the RunPod console):
#   AGENT2MODEL_EXAMPLE     example name / build subdir (default: travel)
#   AGENT2MODEL_BUILD_DIR   build dir (default: /workspace/build/$AGENT2MODEL_EXAMPLE)
#   AGENT2MODEL_BASE_MODEL  HF base model id (train)
#   AGENT2MODEL_SIZE        training preset: 3b (single GPU) or 8b (ZeRO-3); default 3b
#   AGENT2MODEL_EPOCHS      training epochs (train)
#   AGENT2MODEL_N           conversations (generate) / scenarios (evaluate)
#   AGENT2MODEL_BUDGET      USD hard cap for generate/evaluate
#   AGENT2MODEL_PORT        serve port (default: 8000)
#   ANTHROPIC_API_KEY        required for generate/evaluate
set -euo pipefail

STAGE="${1:-train}"
EXAMPLE="${AGENT2MODEL_EXAMPLE:-travel}"
BUILD_DIR="${AGENT2MODEL_BUILD_DIR:-/workspace/build/${EXAMPLE}}"
PORT="${AGENT2MODEL_PORT:-8000}"

# Install the package with the extras the stage needs.
case "$STAGE" in
  train)    EXTRAS="train" ;;
  serve)    EXTRAS="serve" ;;
  generate) EXTRAS="report" ;;
  evaluate) EXTRAS="report" ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac

echo ">> Installing agent2model[${EXTRAS}]"
pip install --no-cache-dir "agent2model[${EXTRAS}]"

case "$STAGE" in
  generate)
    : "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set for generation}"
    agent2model generate "${BUILD_DIR}" \
      --n "${AGENT2MODEL_N:-2000}" \
      --budget "${AGENT2MODEL_BUDGET:-60}"
    ;;
  train)
    # --size selects the recipe: 3b (single GPU) or 8b (DeepSpeed ZeRO-3, which
    # the CLI launches via `accelerate launch` across the pod's GPUs). Without it
    # an 8B model would train on the single-GPU 3B preset and OOM.
    agent2model train "${BUILD_DIR}" \
      --base "${AGENT2MODEL_BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
      --size "${AGENT2MODEL_SIZE:-3b}" \
      --epochs "${AGENT2MODEL_EPOCHS:-20}"
    ;;
  evaluate)
    : "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set for evaluation}"
    agent2model eval "${BUILD_DIR}" \
      --baselines in_context \
      --n "${AGENT2MODEL_N:-200}" \
      --budget "${AGENT2MODEL_BUDGET:-60}"
    ;;
  serve)
    agent2model serve "${BUILD_DIR}" --port "${PORT}" --host 0.0.0.0
    ;;
esac
