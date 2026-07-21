#!/usr/bin/env bash
set -euo pipefail

# Run this script on every node. By default it expects node1 and node2.
# Override NODE_RANK when the hostname does not exactly match one of NODES.

NODES="${NODES:-node1 node2}"
MASTER_ADDR="${MASTER_ADDR:-node1}"
MASTER_PORT="${MASTER_PORT:-29500}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/app/miniconda3/bin/accelerate}"
CONFIG_FILE="${CONFIG_FILE:-examples/wanvideo/model_training/full/accelerate_config_zero3.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-examples/wanvideo/model_training/train.py}"
COMM_IFNAME="${COMM_IFNAME:-enp94s0f0np0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

DATASET_BASE_PATH="${DATASET_BASE_PATH:-data/test_data}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/metadata.csv}"
DATA_FILE_KEYS="${DATA_FILE_KEYS:-video,input_audio}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"

HEIGHT="${HEIGHT:-}"
WIDTH="${WIDTH:-}"
MAX_PIXELS="${MAX_PIXELS:-589824}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FRAME_RATE="${FRAME_RATE:-16}"
FIX_FRAME_RATE="${FIX_FRAME_RATE:-True}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
SAVE_STEPS="${SAVE_STEPS:-200}"
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/Wan2.2-S2V-14B_full}"

# 0/false: normal training from raw data.
# 1/true: pre-extract data features first, then train from cached features.
# only: only run feature pre-extraction and skip training.
# cache: only train from an existing feature cache.
PRE_EXTRACT_DATA_FEATURES="${PRE_EXTRACT_DATA_FEATURES:-0}"
if [ -n "${HEIGHT}" ] && [ -n "${WIDTH}" ]; then
  DATA_FEATURE_SIZE_TAG="${DATA_FEATURE_SIZE_TAG:-${HEIGHT}x${WIDTH}x${NUM_FRAMES}}"
else
  DATA_FEATURE_SIZE_TAG="${DATA_FEATURE_SIZE_TAG:-max_pixels_${MAX_PIXELS}_frames_${NUM_FRAMES}}"
fi
DATA_FEATURE_CACHE_PATH="${DATA_FEATURE_CACHE_PATH:-./models/cache/Wan2.2-S2V-14B_full_${DATA_FEATURE_SIZE_TAG}_features}"

MODEL_ID_WITH_ORIGIN_PATHS="${MODEL_ID_WITH_ORIGIN_PATHS:-Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth}"
AUDIO_PROCESSOR_PATH="${AUDIO_PROCESSOR_PATH:-Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/}"
TRAINABLE_MODELS="${TRAINABLE_MODELS:-dit}"
REMOVE_PREFIX_IN_CKPT="${REMOVE_PREFIX_IN_CKPT:-pipe.dit.}"
EXTRA_INPUTS="${EXTRA_INPUTS:-input_image,input_audio}"
USE_GRADIENT_CHECKPOINTING_OFFLOAD="${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-1}"

if [[ "${ACCELERATE_BIN}" == */* ]]; then
  ACCELERATE_DIR="$(cd "$(dirname "${ACCELERATE_BIN}")" && pwd)"
  export PATH="${ACCELERATE_DIR}:${PATH}"
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${COMM_IFNAME}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${COMM_IFNAME}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"

read -r -a NODE_LIST <<< "${NODES}"
NUM_MACHINES="${NUM_MACHINES:-${#NODE_LIST[@]}}"

if [ -z "${GPUS_PER_NODE:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS_PER_NODE="$(nvidia-smi -L | wc -l)"
  else
    GPUS_PER_NODE="8"
  fi
fi

NUM_PROCESSES="${NUM_PROCESSES:-$((GPUS_PER_NODE * NUM_MACHINES))}"

if [ -z "${NODE_RANK:-}" ]; then
  SHORT_HOST="$(hostname -s)"
  FULL_HOST="$(hostname -f 2>/dev/null || hostname)"
  NODE_RANK=""
  for i in "${!NODE_LIST[@]}"; do
    if [ "${SHORT_HOST}" = "${NODE_LIST[$i]}" ] || [ "${FULL_HOST}" = "${NODE_LIST[$i]}" ]; then
      NODE_RANK="${i}"
      break
    fi
  done
fi

if [ -z "${NODE_RANK:-}" ]; then
  echo "Cannot infer NODE_RANK from hostname. Set NODE_RANK=0 on node1 and NODE_RANK=1 on node2." >&2
  exit 1
fi

if [ "${NODE_RANK}" -lt 0 ] || [ "${NODE_RANK}" -ge "${NUM_MACHINES}" ]; then
  echo "NODE_RANK=${NODE_RANK} is out of range for NUM_MACHINES=${NUM_MACHINES}." >&2
  exit 1
fi

if [ ! -x "${ACCELERATE_BIN}" ]; then
  echo "ACCELERATE_BIN=${ACCELERATE_BIN} is not executable. Override ACCELERATE_BIN if needed." >&2
  exit 1
fi

if ! command -v ninja >/dev/null 2>&1; then
  echo "ninja is not found in PATH. Install ninja or prepend its bin directory to PATH." >&2
  exit 1
fi

LAUNCH_ARGS=(
  --config_file "${CONFIG_FILE}"
  --deepspeed_multinode_launcher standard
  --num_machines "${NUM_MACHINES}"
  --num_processes "${NUM_PROCESSES}"
  --machine_rank "${NODE_RANK}"
  --main_process_ip "${MASTER_ADDR}"
  --main_process_port "${MASTER_PORT}"
  --same_network
)

SIZE_ARGS=(
  --max_pixels "${MAX_PIXELS}"
  --num_frames "${NUM_FRAMES}"
  --frame_rate "${FRAME_RATE}"
)
if [ -n "${HEIGHT}" ]; then
  SIZE_ARGS+=(--height "${HEIGHT}")
fi
if [ -n "${WIDTH}" ]; then
  SIZE_ARGS+=(--width "${WIDTH}")
fi
if [ "${FIX_FRAME_RATE}" = "1" ] || [ "${FIX_FRAME_RATE}" = "true" ] || [ "${FIX_FRAME_RATE}" = "True" ]; then
  SIZE_ARGS+=(--fix_frame_rate True)
fi

MODEL_ARGS=(
  --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN_PATHS}"
  --audio_processor_path "${AUDIO_PROCESSOR_PATH}"
  --trainable_models "${TRAINABLE_MODELS}"
  --remove_prefix_in_ckpt "${REMOVE_PREFIX_IN_CKPT}"
  --extra_inputs "${EXTRA_INPUTS}"
)
if [ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD}" = "1" ] || [ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD}" = "true" ] || [ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD}" = "True" ]; then
  MODEL_ARGS+=(--use_gradient_checkpointing_offload)
else
  MODEL_ARGS+=(--use_gradient_checkpointing)
fi

RAW_DATA_ARGS=(
  --dataset_base_path "${DATASET_BASE_PATH}"
  --dataset_metadata_path "${DATASET_METADATA_PATH}"
  --data_file_keys "${DATA_FILE_KEYS}"
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  "${SIZE_ARGS[@]}"
)

echo "Launching Wan2.2-S2V-14B full training:"
echo "  mode: ${PRE_EXTRACT_DATA_FEATURES}"
echo "  nodes: ${NODES}"
echo "  node_rank: ${NODE_RANK}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  config_file: ${CONFIG_FILE}"
echo "  gpus_per_node: ${GPUS_PER_NODE}"
echo "  total_processes: ${NUM_PROCESSES}"
echo "  dataset_base_path: ${DATASET_BASE_PATH}"
echo "  dataset_metadata_path: ${DATASET_METADATA_PATH}"
echo "  feature_cache_path: ${DATA_FEATURE_CACHE_PATH}"
echo "  max_pixels: ${MAX_PIXELS}"
echo "  size: ${HEIGHT:-auto}x${WIDTH:-auto}x${NUM_FRAMES}"
echo "  frame_rate: ${FRAME_RATE}"
echo "  fix_frame_rate: ${FIX_FRAME_RATE}"
echo "  nccl_socket_ifname: ${NCCL_SOCKET_IFNAME}"
echo "  gloo_socket_ifname: ${GLOO_SOCKET_IFNAME}"
echo "  nccl_net: ${NCCL_NET}"
echo "  nccl_ib_disable: ${NCCL_IB_DISABLE}"
echo "  nccl_channels: ${NCCL_MIN_NCHANNELS}-${NCCL_MAX_NCHANNELS}"
echo "  ninja: $(command -v ninja)"

run_pre_extract() {
  echo "[Wan2.2-S2V] Pre-extracting data features to ${DATA_FEATURE_CACHE_PATH}"
  local cmd=(
    "${ACCELERATE_BIN}" launch "${LAUNCH_ARGS[@]}" "${TRAIN_SCRIPT}"
    "${RAW_DATA_ARGS[@]}"
    --dataset_repeat 1
    "${MODEL_ARGS[@]}"
    --task sft:data_process
    --output_path "${DATA_FEATURE_CACHE_PATH}"
  )
  "${cmd[@]}"
}

run_train_from_raw() {
  echo "[Wan2.2-S2V] Training from raw dataset ${DATASET_BASE_PATH}"
  local cmd=(
    "${ACCELERATE_BIN}" launch "${LAUNCH_ARGS[@]}" "${TRAIN_SCRIPT}"
    "${RAW_DATA_ARGS[@]}"
    --dataset_repeat "${DATASET_REPEAT}"
    "${MODEL_ARGS[@]}"
    --learning_rate "${LEARNING_RATE}"
    --num_epochs "${NUM_EPOCHS}"
    --save_steps "${SAVE_STEPS}"
    --task sft
    --output_path "${OUTPUT_PATH}"
  )
  "${cmd[@]}"
}

run_train_from_cache() {
  echo "[Wan2.2-S2V] Training from pre-extracted data features ${DATA_FEATURE_CACHE_PATH}"
  local cmd=(
    "${ACCELERATE_BIN}" launch "${LAUNCH_ARGS[@]}" "${TRAIN_SCRIPT}"
    --dataset_base_path "${DATA_FEATURE_CACHE_PATH}"
    --dataset_repeat "${DATASET_REPEAT}"
    --dataset_num_workers "${DATASET_NUM_WORKERS}"
    "${SIZE_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    --learning_rate "${LEARNING_RATE}"
    --num_epochs "${NUM_EPOCHS}"
    --save_steps "${SAVE_STEPS}"
    --task sft:train
    --output_path "${OUTPUT_PATH}"
  )
  "${cmd[@]}"
}

case "${PRE_EXTRACT_DATA_FEATURES}" in
  1|true|True|yes|YES)
    run_pre_extract
    run_train_from_cache
    ;;
  only|ONLY)
    run_pre_extract
    ;;
  cache|CACHE|cached|CACHED|train_from_cache)
    run_train_from_cache
    ;;
  0|false|False|no|NO)
    run_train_from_raw
    ;;
  *)
    echo "Unsupported PRE_EXTRACT_DATA_FEATURES=${PRE_EXTRACT_DATA_FEATURES}. Use 0, 1, only, or cache." >&2
    exit 1
    ;;
esac
