#!/usr/bin/env bash
set -euo pipefail

# Train directly from raw videos on node2/node3.
# The default ZeRO-3 configuration must use a fixed frame count. For mixed
# 81/97/113-frame training, use build_cache followed by train_cache_multinode.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPO_ROOT}"

export NODES="${NODES:-node2 node3}"
export MASTER_NODE="${MASTER_NODE:-node2}"
export MASTER_ADDR="${MASTER_ADDR:-${MASTER_NODE}}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export LOCAL_NODE="${LOCAL_NODE:-node2}"
export TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode-run.sh}"
export TRAIN_LOG="${TRAIN_LOG:-train-raw.log}"

export ACCELERATE_BIN="${ACCELERATE_BIN:-/data-training/miniconda/bin/accelerate}"
export MODEL_BASE_PATH="${MODEL_BASE_PATH:-/data-training/models}"
export COMM_IFNAME="${COMM_IFNAME:-bond0}"
export DATASET_BASE_PATH="${DATASET_BASE_PATH:-/data-training/train_data_5s}"
export DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/metadata.csv}"
export DATA_FILE_KEYS="${DATA_FILE_KEYS:-video,input_audio}"
export DATASET_REPEAT="${DATASET_REPEAT:-1}"
export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"

export NUM_FRAMES="${NUM_FRAMES:-81}"
export MIN_NUM_FRAMES="${MIN_NUM_FRAMES:-${NUM_FRAMES}}"
export FRAME_RATE="${FRAME_RATE:-16}"
export FIX_FRAME_RATE="${FIX_FRAME_RATE:-True}"
export FRAME_COUNT_STRIDE="${FRAME_COUNT_STRIDE:-16}"
export FRAME_COUNT_REMAINDER="${FRAME_COUNT_REMAINDER:-1}"
export FRAME_COUNT_ROUNDING="${FRAME_COUNT_ROUNDING:-nearest}"
export MAX_FRAME_PADDING="${MAX_FRAME_PADDING:-8}"
export AUDIO_SAMPLE_RATE="${AUDIO_SAMPLE_RATE:-16000}"
export AUDIO_DURATION_POLICY="${AUDIO_DURATION_POLICY:-trim_pad}"
export MAX_AUDIO_PADDING_SECONDS="${MAX_AUDIO_PADDING_SECONDS:-0.5}"
export TILED="${TILED:-1}"

export OUTPUT_PATH="${OUTPUT_PATH:-./models/train/Wan2.2-S2V-14B_raw_f81}"
export LEARNING_RATE="${LEARNING_RATE:-1e-7}"
export NUM_EPOCHS="${NUM_EPOCHS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export ENABLE_TENSORBOARD_LOG="${ENABLE_TENSORBOARD_LOG:-1}"
export USE_GRADIENT_CHECKPOINTING_OFFLOAD="${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-1}"
export S2V_REF_ROPE_MODE="${S2V_REF_ROPE_MODE:-source_id_local}"

echo "[Example] Training Wan2.2-S2V directly from raw data"
echo "  nodes: ${NODES}"
echo "  metadata: ${DATASET_METADATA_PATH}"
echo "  frames: ${MIN_NUM_FRAMES}-${NUM_FRAMES}"
echo "  checkpoint: ${RESUME_FROM_CHECKPOINT:-none}"
echo "  output: ${OUTPUT_PATH}"

exec bash script/launch_wan22_s2v_multinode.sh "$@"
