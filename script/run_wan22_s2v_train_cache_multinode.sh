#!/usr/bin/env bash
set -euo pipefail

# Stage 2: train on node2/node3 from a completed feature cache.
# Run this script once on node2; it starts node3 over passwordless SSH.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPO_ROOT}"

export NODES="${NODES:-node2 node3}"
export MASTER_NODE="${MASTER_NODE:-node2}"
export MASTER_ADDR="${MASTER_ADDR:-${MASTER_NODE}}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export LOCAL_NODE="${LOCAL_NODE:-node2}"
export TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode-cache-run.sh}"
export TRAIN_LOG="${TRAIN_LOG:-train-cache.log}"

export ACCELERATE_BIN="${ACCELERATE_BIN:-/data-training/miniconda/bin/accelerate}"
export MODEL_BASE_PATH="${MODEL_BASE_PATH:-/data-training/models}"
export COMM_IFNAME="${COMM_IFNAME:-bond0}"
export DATASET_BASE_PATH="${DATASET_BASE_PATH:-/data-training/train_data_5s}"
export DATA_FEATURE_CACHE_PATH="${DATA_FEATURE_CACHE_PATH:-${DATASET_BASE_PATH}/cache_s2v_f81-113_16n1_fps16_v2}"
export DATASET_REPEAT="${DATASET_REPEAT:-1}"
export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"

export NUM_FRAMES="${NUM_FRAMES:-113}"
export MIN_NUM_FRAMES="${MIN_NUM_FRAMES:-81}"
export FRAME_RATE="${FRAME_RATE:-16}"
export FRAME_COUNT_STRIDE="${FRAME_COUNT_STRIDE:-16}"
export FRAME_COUNT_REMAINDER="${FRAME_COUNT_REMAINDER:-1}"
export FRAME_COUNT_ROUNDING="${FRAME_COUNT_ROUNDING:-nearest}"
export MAX_FRAME_PADDING="${MAX_FRAME_PADDING:-8}"
export AUDIO_SAMPLE_RATE="${AUDIO_SAMPLE_RATE:-16000}"

export OUTPUT_PATH="${OUTPUT_PATH:-./models/train/Wan2.2-S2V-14B_variable_length}"
export LEARNING_RATE="${LEARNING_RATE:-1e-7}"
export NUM_EPOCHS="${NUM_EPOCHS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export ENABLE_TENSORBOARD_LOG="${ENABLE_TENSORBOARD_LOG:-1}"
export USE_GRADIENT_CHECKPOINTING_OFFLOAD="${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-1}"
export S2V_REF_ROPE_MODE="${S2V_REF_ROPE_MODE:-source_id_local}"

echo "[Example] Training Wan2.2-S2V from feature cache"
echo "  nodes: ${NODES}"
echo "  cache: ${DATA_FEATURE_CACHE_PATH}"
echo "  checkpoint: ${RESUME_FROM_CHECKPOINT:-none}"
echo "  output: ${OUTPUT_PATH}"

exec bash script/launch_wan22_s2v_multinode.sh "$@"
