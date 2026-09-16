#!/usr/bin/env bash
set -euo pipefail

# Stage 1: build a variable-length feature cache on one 8-GPU node.
# Override any value before invoking this script, for example:
# DATASET_METADATA_PATH=/path/to/metadata_32.csv bash script/run_wan22_s2v_build_cache.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPO_ROOT}"

export ACCELERATE_BIN="${ACCELERATE_BIN:-/data-training/miniconda/bin/accelerate}"
export MODEL_BASE_PATH="${MODEL_BASE_PATH:-/data-training/models}"
export DATASET_BASE_PATH="${DATASET_BASE_PATH:-/data-training/train_data_5s}"
export DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/metadata.csv}"
export DATA_FEATURE_CACHE_PATH="${DATA_FEATURE_CACHE_PATH:-${DATASET_BASE_PATH}/cache_s2v_f81-113_16n1_fps16_v2}"
export DATA_FILE_KEYS="${DATA_FILE_KEYS:-video,input_audio}"
export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"

export NUM_FRAMES="${NUM_FRAMES:-113}"
export MIN_NUM_FRAMES="${MIN_NUM_FRAMES:-81}"
export FRAME_RATE="${FRAME_RATE:-16}"
export FIX_FRAME_RATE="${FIX_FRAME_RATE:-1}"
export FRAME_COUNT_STRIDE="${FRAME_COUNT_STRIDE:-16}"
export FRAME_COUNT_REMAINDER="${FRAME_COUNT_REMAINDER:-1}"
export FRAME_COUNT_ROUNDING="${FRAME_COUNT_ROUNDING:-nearest}"
export MAX_FRAME_PADDING="${MAX_FRAME_PADDING:-8}"

export AUDIO_SAMPLE_RATE="${AUDIO_SAMPLE_RATE:-16000}"
export AUDIO_DURATION_POLICY="${AUDIO_DURATION_POLICY:-trim_pad}"
export AUDIO_DURATION_TOLERANCE_SECONDS="${AUDIO_DURATION_TOLERANCE_SECONDS:-0.05}"
export MAX_AUDIO_PADDING_SECONDS="${MAX_AUDIO_PADDING_SECONDS:-0.5}"
export DATA_PROCESSING_LOG_SAMPLES="${DATA_PROCESSING_LOG_SAMPLES:-8}"

export TILED="${TILED:-1}"
export TILE_SIZE="${TILE_SIZE:-30,52}"
export TILE_STRIDE="${TILE_STRIDE:-15,26}"
export USE_GRADIENT_CHECKPOINTING_OFFLOAD="${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-1}"
export S2V_REF_ROPE_MODE="${S2V_REF_ROPE_MODE:-source_id_local}"
export RESUME_FEATURE_CACHE="${RESUME_FEATURE_CACHE:-0}"

echo "[Example] Building Wan2.2-S2V variable-length feature cache"
echo "  metadata: ${DATASET_METADATA_PATH}"
echo "  output: ${DATA_FEATURE_CACHE_PATH}"
echo "  resume: ${RESUME_FEATURE_CACHE}"

exec bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-cache-run.sh
