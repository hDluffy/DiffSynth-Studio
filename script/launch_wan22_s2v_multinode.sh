#!/usr/bin/env bash
set -euo pipefail

# One-click launcher for Wan2.2-S2V-14B distributed training.
# Run this script on the master node. It starts the same training command on
# every host in NODES and assigns NODE_RANK from the host's position in NODES.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

NODES_VALUE="${NODES:-node1 node2}"
MASTER_NODE_VALUE="${MASTER_NODE:-${MASTER_ADDR:-}}"
LOCAL_NODE_VALUE="${LOCAL_NODE:-}"
REMOTE_DIR_VALUE="${REMOTE_DIR:-${REPO_ROOT}}"
TRAIN_LAUNCHER_VALUE="${TRAIN_LAUNCHER:-examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode-run.sh}"
LOG_FILE_VALUE="${TRAIN_LOG:-train.log}"
SSH_USER_VALUE="${SSH_USER:-}"
SSH_PORT_VALUE="${SSH_PORT:-}"
SSH_OPTIONS_VALUE="${SSH_OPTIONS:-}"
DRY_RUN=0
FOLLOW_LOG=0

usage() {
  cat <<'USAGE'
Usage:
  bash script/launch_wan22_s2v_multinode.sh [OPTIONS]

Run on the master node to start Wan2.2-S2V-14B full training on all nodes.
The position in NODES determines NODE_RANK (node1=0, node2=1, ...).
Passwordless SSH and the same repository path on every node are expected.

Options:
      --nodes LIST          Space-separated hosts; default: "node1 node2".
      --master-node HOST    Master host/address; default: first host in NODES.
      --local-node HOST     Host on which this script is running. Normally
                            detected from hostname; set this for SSH aliases.
      --remote-dir PATH     Repository path on remote hosts; default: current repo path.
      --launcher PATH       Training launcher relative to remote dir (or absolute path).
      --log-file PATH       Per-node log path; default: train.log.
      --ssh-user USER       SSH user for remote hosts.
      --ssh-port PORT       SSH port.
      --ssh-options OPTIONS Extra options passed to ssh (shell-word split).
      --follow              Tail the local log after all nodes are started.
      --dry-run             Print commands without starting training.
  -h, --help                Show this help.

Environment variables:
  NODES, MASTER_NODE/MASTER_ADDR, LOCAL_NODE, REMOTE_DIR, TRAIN_LAUNCHER,
  TRAIN_LOG, SSH_USER, SSH_PORT, SSH_OPTIONS, MASTER_PORT, GPUS_PER_NODE,
  NUM_MACHINES, RESUME_FROM_CHECKPOINT and other variables consumed by the
  training launcher.

Examples:
  # 1. Precompute variable-length features on one 8-GPU node.
  bash script/run_wan22_s2v_build_cache.sh

  # Resume an interrupted cache build with exactly the same configuration.
  RESUME_FEATURE_CACHE=1 bash script/run_wan22_s2v_build_cache.sh

  # 2. Train from a complete feature cache on node2/node3 (16 GPUs).
  RESUME_FROM_CHECKPOINT=/path/to/checkpoint.safetensors \
    bash script/run_wan22_s2v_train_cache_multinode.sh --follow

  # 3. Train directly from raw data with a fixed frame count on 16 GPUs.
  DATASET_METADATA_PATH=/data-training/train_data_5s/metadata_32.csv \
    bash script/run_wan22_s2v_train_raw_multinode.sh --follow
USAGE
}

die() {
  echo "Error: $*" >&2
  exit 2
}

shell_quote() {
  local value="$1"
  printf "'%s'" "${value//\'/\'\\\'\'}"
}

is_positive_integer() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && (( 10#${1} > 0 ))
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --nodes)
      [[ $# -ge 2 ]] || die "--nodes requires a space-separated host list."
      NODES_VALUE="$2"
      shift 2
      ;;
    --master-node)
      [[ $# -ge 2 ]] || die "--master-node requires a host."
      MASTER_NODE_VALUE="$2"
      shift 2
      ;;
    --local-node)
      [[ $# -ge 2 ]] || die "--local-node requires a host."
      LOCAL_NODE_VALUE="$2"
      shift 2
      ;;
    --remote-dir)
      [[ $# -ge 2 ]] || die "--remote-dir requires a path."
      REMOTE_DIR_VALUE="$2"
      shift 2
      ;;
    --launcher)
      [[ $# -ge 2 ]] || die "--launcher requires a path."
      TRAIN_LAUNCHER_VALUE="$2"
      shift 2
      ;;
    --log-file)
      [[ $# -ge 2 ]] || die "--log-file requires a path."
      LOG_FILE_VALUE="$2"
      shift 2
      ;;
    --ssh-user)
      [[ $# -ge 2 ]] || die "--ssh-user requires a user."
      SSH_USER_VALUE="$2"
      shift 2
      ;;
    --ssh-port)
      [[ $# -ge 2 ]] || die "--ssh-port requires a port."
      SSH_PORT_VALUE="$2"
      shift 2
      ;;
    --ssh-options)
      [[ $# -ge 2 ]] || die "--ssh-options requires a value."
      SSH_OPTIONS_VALUE="$2"
      shift 2
      ;;
    --follow)
      FOLLOW_LOG=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      [[ $# -eq 0 ]] || die "unexpected positional arguments: $*"
      ;;
    -* )
      die "unknown option: $1"
      ;;
    *)
      die "unexpected positional argument: $1"
      ;;
  esac
done

read -r -a node_list <<< "${NODES_VALUE}"
(( ${#node_list[@]} > 0 )) || die "NODES must contain at least one host."

if [[ -z "${MASTER_NODE_VALUE}" ]]; then
  MASTER_NODE_VALUE="${node_list[0]}"
fi

if [[ -z "${LOCAL_NODE_VALUE}" ]]; then
  short_host="$(hostname -s)"
  full_host="$(hostname -f 2>/dev/null || hostname)"
  for host in "${node_list[@]}"; do
    compare_host="${host#*@}"
    compare_host="${compare_host#[}"
    compare_host="${compare_host%]}"
    if [[ "${compare_host}" == "${short_host}" || "${compare_host}" == "${full_host}" ]]; then
      LOCAL_NODE_VALUE="${host}"
      break
    fi
  done
fi

# The command is expected to run on MASTER_NODE. If hostnames use aliases and
# cannot be detected, assume the current machine is the configured master.
if [[ -z "${LOCAL_NODE_VALUE}" ]]; then
  LOCAL_NODE_VALUE="${MASTER_NODE_VALUE}"
  echo "Warning: current hostname is not in NODES; treating this machine as ${LOCAL_NODE_VALUE}." >&2
fi

num_machines="${NUM_MACHINES:-${#node_list[@]}}"
is_positive_integer "${num_machines}" || die "NUM_MACHINES must be a positive integer."
if [[ -n "${SSH_PORT_VALUE}" ]] && ! is_positive_integer "${SSH_PORT_VALUE}"; then
  die "SSH_PORT must be a positive integer."
fi

master_index=""
local_index=""
for i in "${!node_list[@]}"; do
  [[ -n "${node_list[$i]}" ]] || die "NODES contains an empty host."
  if [[ "${node_list[$i]}" == "${MASTER_NODE_VALUE}" ]]; then
    master_index="${i}"
  fi
  if [[ "${node_list[$i]}" == "${LOCAL_NODE_VALUE}" ]]; then
    local_index="${i}"
  fi
done
[[ -n "${master_index}" ]] || die "MASTER_NODE='${MASTER_NODE_VALUE}' is not present in NODES='${NODES_VALUE}'."
[[ -n "${local_index}" ]] || die "LOCAL_NODE='${LOCAL_NODE_VALUE}' is not present in NODES='${NODES_VALUE}'."
(( master_index == local_index )) || die "Run this launcher on MASTER_NODE=${MASTER_NODE_VALUE}; LOCAL_NODE=${LOCAL_NODE_VALUE}."
(( num_machines >= ${#node_list[@]} )) || die "NUM_MACHINES is smaller than the number of hosts in NODES."

if [[ "${TRAIN_LAUNCHER_VALUE}" = /* ]]; then
  launcher_path="${TRAIN_LAUNCHER_VALUE}"
else
  launcher_path="${REMOTE_DIR_VALUE%/}/${TRAIN_LAUNCHER_VALUE}"
fi

ssh_args=()
if [[ -n "${SSH_PORT_VALUE}" ]]; then
  ssh_args+=(-p "${SSH_PORT_VALUE}")
fi
if [[ -n "${SSH_OPTIONS_VALUE}" ]]; then
  read -r -a extra_ssh_args <<< "${SSH_OPTIONS_VALUE}"
  ssh_args+=("${extra_ssh_args[@]}")
fi

remote_host() {
  local host="$1"
  if [[ -n "${SSH_USER_VALUE}" && "${host}" != *@* ]]; then
    printf '%s@%s' "${SSH_USER_VALUE}" "${host}"
  else
    printf '%s' "${host}"
  fi
}

# Keep this list centralized so local and SSH-launched nodes receive exactly
# the same optional training configuration.
FORWARDED_VARIABLES=(
  GPUS_PER_NODE NUM_PROCESSES ACCELERATE_BIN CONFIG_FILE DATA_PROCESS_CONFIG_FILE
  TRAIN_SCRIPT MODEL_BASE_PATH
  COMM_IFNAME NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_DISABLE NCCL_IB_HCA NCCL_NET
  NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS PYTORCH_CUDA_ALLOC_CONF
  DATASET_BASE_PATH DATASET_METADATA_PATH DATA_FEATURE_CACHE_PATH DATA_FEATURE_SIZE_TAG
  DATA_FILE_KEYS DATASET_REPEAT DATASET_NUM_WORKERS
  HEIGHT WIDTH MAX_PIXELS NUM_FRAMES MIN_NUM_FRAMES FRAME_RATE FIX_FRAME_RATE
  FRAME_COUNT_STRIDE FRAME_COUNT_REMAINDER FRAME_COUNT_ROUNDING MAX_FRAME_PADDING
  AUDIO_SAMPLE_RATE AUDIO_DURATION_POLICY AUDIO_DURATION_TOLERANCE_SECONDS
  MAX_AUDIO_PADDING_SECONDS MAX_AUDIO_TRIMMING_SECONDS DATA_PROCESSING_LOG_SAMPLES
  TILED TILE_SIZE TILE_STRIDE
  DIT_MODEL_ID_WITH_ORIGIN_PATH MODEL_ID_WITH_ORIGIN_PATHS AUDIO_PROCESSOR_PATH
  TRAINABLE_MODELS REMOVE_PREFIX_IN_CKPT EXTRA_INPUTS OFFLOAD_MODELS FP8_MODELS
  LEARNING_RATE NUM_EPOCHS SAVE_STEPS OUTPUT_PATH RESUME_FROM_CHECKPOINT
  RESUME_FEATURE_CACHE ENABLE_TENSORBOARD_LOG USE_GRADIENT_CHECKPOINTING_OFFLOAD
  S2V_REF_ROPE_MODE S2V_REF_SOURCE_ID S2V_REF_ROPE_THETA S2V_REF_TIME_BASE
  S2V_REF_TIME_MARGIN
)

# Include distributed settings and common launcher overrides when they are
# present on the master.
remote_env=(
  "NODES=$(shell_quote "${NODES_VALUE}")"
  "NUM_MACHINES=$(shell_quote "${num_machines}")"
  "MASTER_ADDR=$(shell_quote "${MASTER_NODE_VALUE}")"
  "MASTER_PORT=$(shell_quote "${MASTER_PORT:-29500}")"
)
for variable in "${FORWARDED_VARIABLES[@]}"; do
  if [[ -n "${!variable+x}" ]]; then
    remote_env+=("${variable}=$(shell_quote "${!variable}")")
  fi
done

local_log_path="${LOG_FILE_VALUE}"
if [[ "${local_log_path}" != /* ]]; then
  local_log_path="${REPO_ROOT}/${local_log_path}"
fi

start_node() {
  local host="$1" rank="$2" remote="$3" command_string node_log
  local env_string=""
  local env_args=("NODES=${NODES_VALUE}" "NUM_MACHINES=${num_machines}"
    "MASTER_ADDR=${MASTER_NODE_VALUE}" "MASTER_PORT=${MASTER_PORT:-29500}" "NODE_RANK=${rank}")
  node_log="${LOG_FILE_VALUE}"
  if [[ "${host}" != "${LOCAL_NODE_VALUE}" ]]; then
    # Repository paths are often shared across nodes. Distinct files prevent
    # concurrent shell redirections from corrupting the log with sparse/NUL
    # regions while preserving the requested name for the local master log.
    node_log="${LOG_FILE_VALUE}.rank${rank}"
  fi
  for item in "${remote_env[@]}"; do
    env_string+="${item} "
  done
  env_string+="NODE_RANK=$(shell_quote "${rank}")"
  for variable in "${FORWARDED_VARIABLES[@]}"; do
    if [[ -n "${!variable+x}" ]]; then
      env_args+=("${variable}=${!variable}")
    fi
  done

  command_string="cd $(shell_quote "${REMOTE_DIR_VALUE}") && mkdir -p $(shell_quote "$(dirname -- "${node_log}")") && ${env_string} nohup bash $(shell_quote "${launcher_path}") > $(shell_quote "${node_log}") 2>&1 < /dev/null &"

  echo "[${host}] starting rank ${rank}; log=${node_log}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    if [[ "${host}" == "${LOCAL_NODE_VALUE}" ]]; then
      echo "  local: ${command_string}"
    else
      echo "  ssh ${remote}: ${command_string}"
    fi
    return 0
  fi

  if [[ "${host}" == "${LOCAL_NODE_VALUE}" ]]; then
    (
      cd "${REMOTE_DIR_VALUE}"
      mkdir -p "$(dirname -- "${node_log}")"
      env "${env_args[@]}" nohup bash "${launcher_path}" > "${node_log}" 2>&1 < /dev/null &
      train_pid=$!
      echo "TRAIN_PID=${train_pid}"
    )
  else
    # -n closes stdin and -f backgrounds the SSH client after authentication,
    # so a long-running remote training process cannot hold up the master.
    ssh "${ssh_args[@]}" -n -f "${remote}" "${command_string}"
    echo "[${host}] remote launch submitted"
  fi
}

if [[ "${DRY_RUN}" -eq 0 ]] && ! command -v ssh >/dev/null 2>&1; then
  die "ssh is required when not using --dry-run."
fi

master_port_is_listening() {
  command -v ss >/dev/null 2>&1 &&
    [[ -n "$(ss -H -ltn "sport = :${MASTER_PORT:-29500}" 2>/dev/null)" ]]
}

wait_for_master_port() {
  if ! command -v ss >/dev/null 2>&1; then
    sleep 2
    return
  fi
  local attempt
  for attempt in {1..30}; do
    if master_port_is_listening; then
      echo "Master rendezvous is listening on port ${MASTER_PORT:-29500}."
      return
    fi
    sleep 1
  done
  die "Master rendezvous did not listen on port ${MASTER_PORT:-29500} within 30 seconds. Check ${LOG_FILE_VALUE}."
}

echo "Wan2.2-S2V-14B multinode launch"
echo "  nodes: ${NODES_VALUE}"
echo "  master: ${MASTER_NODE_VALUE} (rank ${master_index})"
echo "  local: ${LOCAL_NODE_VALUE} (rank ${local_index})"
echo "  launcher: ${launcher_path}"
echo "  log: ${LOG_FILE_VALUE}"

# Refuse to connect a new run to a stale torch-elastic rendezvous.
if [[ "${DRY_RUN}" -eq 0 ]] && master_port_is_listening; then
  die "MASTER_PORT=${MASTER_PORT:-29500} is already in use. Stop the previous run or choose another port."
fi

# Start the local master first. Once its rendezvous socket is ready, start all
# remote workers so they cannot accidentally connect to a stale/previous run.
start_node "${node_list[$local_index]}" "${local_index}" "$(remote_host "${node_list[$local_index]}")"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  wait_for_master_port
fi
for i in "${!node_list[@]}"; do
  [[ "${i}" -eq "${local_index}" ]] && continue
  start_node "${node_list[$i]}" "${i}" "$(remote_host "${node_list[$i]}")"
done

echo "All node launch commands completed. The master log is '${LOG_FILE_VALUE}'; remote logs use '${LOG_FILE_VALUE}.rank<N>'."
if [[ "${FOLLOW_LOG}" -eq 1 && "${DRY_RUN}" -eq 0 ]]; then
  exec tail -F "${local_log_path}"
fi
