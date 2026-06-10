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

echo "Launching Wan2.2-S2V-14B full training:"
echo "  nodes: ${NODES}"
echo "  node_rank: ${NODE_RANK}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  config_file: ${CONFIG_FILE}"
echo "  gpus_per_node: ${GPUS_PER_NODE}"
echo "  total_processes: ${NUM_PROCESSES}"
echo "  nccl_socket_ifname: ${NCCL_SOCKET_IFNAME}"
echo "  gloo_socket_ifname: ${GLOO_SOCKET_IFNAME}"
echo "  nccl_net: ${NCCL_NET}"
echo "  nccl_ib_disable: ${NCCL_IB_DISABLE}"
echo "  nccl_channels: ${NCCL_MIN_NCHANNELS}-${NCCL_MAX_NCHANNELS}"
echo "  ninja: $(command -v ninja)"

"${ACCELERATE_BIN}" launch \
  --config_file "${CONFIG_FILE}" \
  --deepspeed_multinode_launcher standard \
  --num_machines "${NUM_MACHINES}" \
  --num_processes "${NUM_PROCESSES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --same_network \
  "${TRAIN_SCRIPT}" \
  --dataset_base_path data/test_data \
  --dataset_metadata_path data/test_data/metadata.csv \
  --data_file_keys "video,input_audio" \
  --height 576 \
  --width 768 \
  --num_frames 241 \
  --dataset_repeat 5 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \
  --audio_processor_path "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --trainable_models "dit" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-S2V-14B_full" \
  --extra_inputs "input_image,input_audio" \
  --use_gradient_checkpointing_offload
