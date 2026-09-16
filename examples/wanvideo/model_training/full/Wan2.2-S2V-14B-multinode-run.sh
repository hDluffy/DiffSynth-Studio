#!/usr/bin/env bash
set -euo pipefail

# Run this script on every node. For one-click SSH orchestration use
# script/launch_wan22_s2v_multinode.sh.
#
# With the default ZeRO-3 config, raw-data training must use a fixed frame
# count. Variable-length raw inputs can make frozen VAE collectives diverge;
# use Wan2.2-S2V-14B-cache-run.sh followed by
# Wan2.2-S2V-14B-multinode-cache-run.sh for variable-length training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

NODES="${NODES:-node1 node2}"
MASTER_ADDR="${MASTER_ADDR:-node1}"
MASTER_PORT="${MASTER_PORT:-29500}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
CONFIG_FILE="${CONFIG_FILE:-examples/wanvideo/model_training/full/accelerate_config_zero3.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-examples/wanvideo/model_training/train.py}"
MODEL_BASE_PATH=${MODEL_BASE_PATH:-${DIFFSYNTH_MODEL_BASE_PATH:-./models}}
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE_PATH}"
COMM_IFNAME="${COMM_IFNAME:-enp94s0f0np0}"

DATASET_BASE_PATH="${DATASET_BASE_PATH:-data/test_data}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/metadata.csv}"
DATA_FILE_KEYS="${DATA_FILE_KEYS:-video,input_audio}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"

HEIGHT="${HEIGHT:-}"
WIDTH="${WIDTH:-}"
MAX_PIXELS="${MAX_PIXELS:-589824}"
NUM_FRAMES="${NUM_FRAMES:-81}"
MIN_NUM_FRAMES="${MIN_NUM_FRAMES:-${NUM_FRAMES}}"
FRAME_RATE="${FRAME_RATE:-16}"
FIX_FRAME_RATE="${FIX_FRAME_RATE:-True}"
FRAME_COUNT_STRIDE="${FRAME_COUNT_STRIDE:-16}"
FRAME_COUNT_REMAINDER="${FRAME_COUNT_REMAINDER:-1}"
FRAME_COUNT_ROUNDING="${FRAME_COUNT_ROUNDING:-nearest}"
MAX_FRAME_PADDING="${MAX_FRAME_PADDING:-8}"
AUDIO_SAMPLE_RATE="${AUDIO_SAMPLE_RATE:-16000}"
AUDIO_DURATION_POLICY="${AUDIO_DURATION_POLICY:-trim_pad}"
AUDIO_DURATION_TOLERANCE_SECONDS="${AUDIO_DURATION_TOLERANCE_SECONDS:-0.05}"
MAX_AUDIO_PADDING_SECONDS="${MAX_AUDIO_PADDING_SECONDS:-0.5}"
MAX_AUDIO_TRIMMING_SECONDS="${MAX_AUDIO_TRIMMING_SECONDS:-}"
DATA_PROCESSING_LOG_SAMPLES="${DATA_PROCESSING_LOG_SAMPLES:-8}"
TILED="${TILED:-1}"
TILE_SIZE="${TILE_SIZE:-30,52}"
TILE_STRIDE="${TILE_STRIDE:-15,26}"

MODEL_ID_WITH_ORIGIN_PATHS="${MODEL_ID_WITH_ORIGIN_PATHS:-Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth}"
AUDIO_PROCESSOR_PATH="${AUDIO_PROCESSOR_PATH:-Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/}"
TRAINABLE_MODELS="${TRAINABLE_MODELS:-dit}"
REMOVE_PREFIX_IN_CKPT="${REMOVE_PREFIX_IN_CKPT:-pipe.dit.}"
EXTRA_INPUTS="${EXTRA_INPUTS:-input_image,input_audio}"
OFFLOAD_MODELS="${OFFLOAD_MODELS:-}"
FP8_MODELS="${FP8_MODELS:-}"

LEARNING_RATE="${LEARNING_RATE:-1e-7}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
SAVE_STEPS="${SAVE_STEPS:-100}"
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/Wan2.2-S2V-14B_full}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
ENABLE_TENSORBOARD_LOG="${ENABLE_TENSORBOARD_LOG:-1}"
USE_GRADIENT_CHECKPOINTING_OFFLOAD="${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-1}"

#legacy_time_offset or source_id_local
S2V_REF_ROPE_MODE="${S2V_REF_ROPE_MODE:-source_id_local}"
S2V_REF_SOURCE_ID="${S2V_REF_SOURCE_ID:-1.0}"
S2V_REF_ROPE_THETA="${S2V_REF_ROPE_THETA:-10000.0}"
S2V_REF_TIME_BASE="${S2V_REF_TIME_BASE:-30}"
S2V_REF_TIME_MARGIN="${S2V_REF_TIME_MARGIN:-9}"

if [[ "${ACCELERATE_BIN}" == */* ]]; then
  ACCELERATE_DIR="$(cd "$(dirname "${ACCELERATE_BIN}")" && pwd)"
  export PATH="${ACCELERATE_DIR}:${PATH}"
fi

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${COMM_IFNAME}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${COMM_IFNAME}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

if [[ "${ACCELERATE_BIN}" == */* ]]; then
  if [ ! -x "${ACCELERATE_BIN}" ]; then
    echo "ACCELERATE_BIN=${ACCELERATE_BIN} is not executable. Override ACCELERATE_BIN if needed." >&2
    exit 1
  fi
elif ! command -v "${ACCELERATE_BIN}" >/dev/null 2>&1; then
  echo "ACCELERATE_BIN=${ACCELERATE_BIN} is not available in PATH." >&2
  exit 1
fi

if ! command -v ninja >/dev/null 2>&1; then
  echo "ninja is not found in PATH. Install ninja or prepend its bin directory to PATH." >&2
  exit 1
fi

if [ "${MIN_NUM_FRAMES}" -lt "${NUM_FRAMES}" ]; then
  echo "Variable-length raw-data training requested (${MIN_NUM_FRAMES}-${NUM_FRAMES} frames)." >&2
  echo "The default ZeRO-3 config does not support this safely; precompute features first." >&2
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

DATA_ARGS=(
  --dataset_base_path "${DATASET_BASE_PATH}"
  --dataset_metadata_path "${DATASET_METADATA_PATH}"
  --data_file_keys "${DATA_FILE_KEYS}"
  --dataset_repeat "${DATASET_REPEAT}"
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --max_pixels "${MAX_PIXELS}"
  --num_frames "${NUM_FRAMES}"
  --min_num_frames "${MIN_NUM_FRAMES}"
  --frame_rate "${FRAME_RATE}"
  --frame_count_stride "${FRAME_COUNT_STRIDE}"
  --frame_count_remainder "${FRAME_COUNT_REMAINDER}"
  --frame_count_rounding "${FRAME_COUNT_ROUNDING}"
  --max_frame_padding "${MAX_FRAME_PADDING}"
  --audio_sample_rate "${AUDIO_SAMPLE_RATE}"
  --audio_duration_policy "${AUDIO_DURATION_POLICY}"
  --audio_duration_tolerance_seconds "${AUDIO_DURATION_TOLERANCE_SECONDS}"
  --max_audio_padding_seconds "${MAX_AUDIO_PADDING_SECONDS}"
  --data_processing_log_samples "${DATA_PROCESSING_LOG_SAMPLES}"
)
if [ -n "${HEIGHT}" ]; then
  DATA_ARGS+=(--height "${HEIGHT}")
fi
if [ -n "${WIDTH}" ]; then
  DATA_ARGS+=(--width "${WIDTH}")
fi
if [ "${FIX_FRAME_RATE}" = "1" ] || [ "${FIX_FRAME_RATE}" = "true" ] || [ "${FIX_FRAME_RATE}" = "True" ]; then
  DATA_ARGS+=(--fix_frame_rate True)
fi
if [ -n "${MAX_AUDIO_TRIMMING_SECONDS}" ]; then
  DATA_ARGS+=(--max_audio_trimming_seconds "${MAX_AUDIO_TRIMMING_SECONDS}")
fi
if [ "${TILED}" = "1" ] || [ "${TILED}" = "true" ] || [ "${TILED}" = "True" ]; then
  DATA_ARGS+=(--tiled --tile_size "${TILE_SIZE}" --tile_stride "${TILE_STRIDE}")
fi

MODEL_ARGS=(
  --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN_PATHS}"
  --audio_processor_path "${AUDIO_PROCESSOR_PATH}"
  --trainable_models "${TRAINABLE_MODELS}"
  --remove_prefix_in_ckpt "${REMOVE_PREFIX_IN_CKPT}"
  --extra_inputs "${EXTRA_INPUTS}"
  --s2v_ref_rope_mode "${S2V_REF_ROPE_MODE}"
  --s2v_ref_source_id "${S2V_REF_SOURCE_ID}"
  --s2v_ref_rope_theta "${S2V_REF_ROPE_THETA}"
  --s2v_ref_time_base "${S2V_REF_TIME_BASE}"
  --s2v_ref_time_margin "${S2V_REF_TIME_MARGIN}"
)
if [ -n "${OFFLOAD_MODELS}" ]; then
  MODEL_ARGS+=(--offload_models "${OFFLOAD_MODELS}")
fi
if [ -n "${FP8_MODELS}" ]; then
  MODEL_ARGS+=(--fp8_models "${FP8_MODELS}")
fi
if [ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD}" = "1" ] || [ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD}" = "true" ] || [ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD}" = "True" ]; then
  MODEL_ARGS+=(--use_gradient_checkpointing_offload)
else
  MODEL_ARGS+=(--use_gradient_checkpointing)
fi

OPTIONAL_TRAIN_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  OPTIONAL_TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [ "${ENABLE_TENSORBOARD_LOG}" = "1" ] || [ "${ENABLE_TENSORBOARD_LOG}" = "true" ] || [ "${ENABLE_TENSORBOARD_LOG}" = "True" ]; then
  OPTIONAL_TRAIN_ARGS+=(--enable_tensorboard_log)
fi

echo "Launching Wan2.2-S2V-14B raw-data training:"
echo "  nodes: ${NODES}"
echo "  node_rank: ${NODE_RANK}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  config_file: ${CONFIG_FILE}"
echo "  gpus_per_node: ${GPUS_PER_NODE}"
echo "  total_processes: ${NUM_PROCESSES}"
echo "  dataset_base_path: ${DATASET_BASE_PATH}"
echo "  dataset_metadata_path: ${DATASET_METADATA_PATH}"
echo "  size: ${HEIGHT:-auto}x${WIDTH:-auto}, frames=${MIN_NUM_FRAMES}-${NUM_FRAMES}"
echo "  frame_count: ${FRAME_COUNT_STRIDE}n+${FRAME_COUNT_REMAINDER}, rounding=${FRAME_COUNT_ROUNDING}, max_padding=${MAX_FRAME_PADDING}"
echo "  frame_rate: ${FRAME_RATE}"
echo "  audio: sample_rate=${AUDIO_SAMPLE_RATE}, policy=${AUDIO_DURATION_POLICY}"
echo "  tiled: ${TILED}"
echo "  nccl_socket_ifname: ${NCCL_SOCKET_IFNAME}"
echo "  gloo_socket_ifname: ${GLOO_SOCKET_IFNAME}"
echo "  nccl_net: ${NCCL_NET}"
echo "  nccl_ib_disable: ${NCCL_IB_DISABLE}"
echo "  nccl_channels: ${NCCL_MIN_NCHANNELS}-${NCCL_MAX_NCHANNELS}"
echo "  s2v_ref_rope_mode: ${S2V_REF_ROPE_MODE}"
echo "  resume_from_checkpoint: ${RESUME_FROM_CHECKPOINT:-none}"
echo "  gradient_checkpointing_offload: ${USE_GRADIENT_CHECKPOINTING_OFFLOAD}"
echo "  output_path: ${OUTPUT_PATH}"
echo "  ninja: $(command -v ninja)"

cmd=(
  "${ACCELERATE_BIN}" launch "${LAUNCH_ARGS[@]}" "${TRAIN_SCRIPT}"
  "${DATA_ARGS[@]}"
  "${MODEL_ARGS[@]}"
  --learning_rate "${LEARNING_RATE}"
  --num_epochs "${NUM_EPOCHS}"
  --save_steps "${SAVE_STEPS}"
  "${OPTIONAL_TRAIN_ARGS[@]}"
  --task sft
  --output_path "${OUTPUT_PATH}"
)
"${cmd[@]}"
