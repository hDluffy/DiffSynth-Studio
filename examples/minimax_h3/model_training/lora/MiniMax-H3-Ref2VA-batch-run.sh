#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA-batch-run.sh DATA_ROOT

DATA_ROOT must contain one directory per dataset. Only direct children that
contain metadata.json are processed:

  DATA_ROOT/
    dataset_a/metadata.json
    dataset_b/metadata.json

The script runs stage 1 (cache) and stage 2 (LoRA training) sequentially for
each dataset. The original MiniMax-H3-Ref2VA-run.sh is not invoked or modified.

Optional environment variables:
  STAGE=both|cache|train       Stages to run (default: both)
  MAX_PIXELS=1048576          Dynamic-resolution maximum pixel area
  HEIGHT=... WIDTH=...        Set both to use fixed resolution instead
  NUM_FRAMES=362              Must satisfy NUM_FRAMES % 17 == 5
  DATASET_REPEAT=5            Stage-2 dataset repeat (cache always uses 1)
  LEARNING_RATE=1e-4          LoRA learning rate
  NUM_EPOCHS=5                Stage-2 epochs
  LORA_RANK=32                LoRA rank
  CACHE_ROOT=...              Cache parent directory
  OUTPUT_ROOT=...             LoRA output parent directory
  ZERO3_CONFIG=...            Stage-2 Accelerate/DeepSpeed config
  SKIP_COMPLETED=1            Skip completed cache/training outputs
  CONTINUE_ON_ERROR=1         Continue with the next dataset after a failure
  DRY_RUN=1                   Print commands without executing them
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd -P)
DATA_ROOT_INPUT=$1

if [[ ! -d $DATA_ROOT_INPUT ]]; then
  echo "Error: dataset root does not exist or is not a directory: $DATA_ROOT_INPUT" >&2
  exit 2
fi

DATA_ROOT=$(cd -- "$DATA_ROOT_INPUT" && pwd -P)
cd -- "$PROJECT_ROOT"

STAGE=${STAGE:-both}
HEIGHT=${HEIGHT:-}
WIDTH=${WIDTH:-}
MAX_PIXELS=${MAX_PIXELS:-1048576}
NUM_FRAMES=${NUM_FRAMES:-362}
DATASET_REPEAT=${DATASET_REPEAT:-5}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
NUM_EPOCHS=${NUM_EPOCHS:-5}
LORA_RANK=${LORA_RANK:-32}
CACHE_ROOT=${CACHE_ROOT:-$PROJECT_ROOT/models/train/MiniMax-H3-Ref2VA-batch-cache}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_ROOT/models/train/MiniMax-H3-Ref2VA}
ZERO3_CONFIG=${ZERO3_CONFIG:-$PROJECT_ROOT/examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml}
SKIP_COMPLETED=${SKIP_COMPLETED:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}
DRY_RUN=${DRY_RUN:-0}

case "$STAGE" in
  both|cache|train) ;;
  *)
    echo "Error: STAGE must be one of: both, cache, train; got: $STAGE" >&2
    exit 2
    ;;
esac

if (( NUM_FRAMES % 17 != 5 )); then
  echo "Error: NUM_FRAMES must satisfy NUM_FRAMES % 17 == 5; got: $NUM_FRAMES" >&2
  exit 2
fi

if [[ -n $HEIGHT && -z $WIDTH ]] || [[ -z $HEIGHT && -n $WIDTH ]]; then
  echo "Error: HEIGHT and WIDTH must either both be set (fixed resolution) or both be empty (dynamic resolution)." >&2
  exit 2
fi

if [[ ! $MAX_PIXELS =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: MAX_PIXELS must be a positive integer; got: $MAX_PIXELS" >&2
  exit 2
fi

if [[ -n $HEIGHT ]]; then
  if [[ ! $HEIGHT =~ ^[1-9][0-9]*$ || ! $WIDTH =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: HEIGHT and WIDTH must be positive integers." >&2
    exit 2
  fi
  RESOLUTION_MODE="fixed ${WIDTH}x${HEIGHT}"
  RESOLUTION_ARGS=(--height "$HEIGHT" --width "$WIDTH" --max_pixels "$MAX_PIXELS")
else
  RESOLUTION_MODE="dynamic (max_pixels=$MAX_PIXELS)"
  RESOLUTION_ARGS=(--max_pixels "$MAX_PIXELS")
fi

if [[ ! -f $ZERO3_CONFIG ]]; then
  echo "Error: ZeRO-3 config does not exist: $ZERO3_CONFIG" >&2
  exit 2
fi

shopt -s nullglob
metadata_files=("$DATA_ROOT"/*/metadata.json)
shopt -u nullglob

if (( ${#metadata_files[@]} == 0 )); then
  echo "Error: no direct child containing metadata.json was found under: $DATA_ROOT" >&2
  exit 2
fi

mkdir -p -- "$CACHE_ROOT" "$OUTPUT_ROOT"

TRAIN_SCRIPT="$PROJECT_ROOT/examples/minimax_h3/model_training/train.py"
DATA_PROCESS_MODELS="MiniMax/MiniMax-H3:Ref2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:Ref2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:Ref2VA/audio_vae/model.safetensors"
DIT_MODEL="MiniMax/MiniMax-H3:Ref2VA/transformer/model*.safetensors"
PROCESSOR="MiniMax/MiniMax-H3:Ref2VA/processor/"
LORA_TARGETS="attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2"

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_logged() {
  local log_file=$1
  shift

  print_command "$@"
  if [[ $DRY_RUN == 1 ]]; then
    return 0
  fi

  "$@" 2>&1 | tee "$log_file"
}

run_cache_stage() {
  local dataset_dir=$1
  local metadata_file=$2
  local cache_dir=$3
  local complete_marker="$cache_dir/.cache-complete"

  if [[ $SKIP_COMPLETED == 1 && -f $complete_marker ]]; then
    echo "[stage 1] Cache already complete; skipping: $cache_dir"
    return 0
  fi

  mkdir -p -- "$cache_dir"
  echo "[stage 1] Building cache"
  run_logged "$cache_dir/stage-1.log" \
    accelerate launch "$TRAIN_SCRIPT" \
      --dataset_base_path "$dataset_dir" \
      --dataset_metadata_path "$metadata_file" \
      --data_file_keys "video,input_audio,references" \
      --extra_inputs "input_audio,references" \
      "${RESOLUTION_ARGS[@]}" \
      --num_frames "$NUM_FRAMES" \
      --dataset_repeat 1 \
      --model_id_with_origin_paths "$DATA_PROCESS_MODELS" \
      --processor_path "$PROCESSOR" \
      --learning_rate "$LEARNING_RATE" \
      --num_epochs 1 \
      --remove_prefix_in_ckpt "pipe.dit." \
      --output_path "$cache_dir" \
      --lora_base_model "dit" \
      --lora_target_modules "$LORA_TARGETS" \
      --lora_rank "$LORA_RANK" \
      --use_gradient_checkpointing_offload \
      --task "sft:data_process" || return $?

  if [[ $DRY_RUN != 1 ]]; then
    touch "$complete_marker"
  fi
}

run_train_stage() {
  local cache_dir=$1
  local output_dir=$2

  if [[ ! -d $cache_dir ]]; then
    echo "Error: cache directory does not exist: $cache_dir" >&2
    return 1
  fi
  if [[ $DRY_RUN != 1 ]] && ! find "$cache_dir" -type f -name '*.pth' -print -quit | grep -q .; then
    echo "Error: no .pth cache files found under: $cache_dir" >&2
    return 1
  fi
  if [[ $SKIP_COMPLETED == 1 && -f $output_dir/.train-complete ]]; then
    echo "[stage 2] Training already complete; skipping: $output_dir"
    return 0
  fi

  mkdir -p -- "$output_dir"
  echo "[stage 2] Training LoRA with ZeRO-3"
  run_logged "$output_dir/stage-2.log" \
    accelerate launch \
      --config_file "$ZERO3_CONFIG" \
      "$TRAIN_SCRIPT" \
      --dataset_base_path "$cache_dir" \
      --data_file_keys "video,input_audio,references" \
      --extra_inputs "input_audio,references" \
      "${RESOLUTION_ARGS[@]}" \
      --num_frames "$NUM_FRAMES" \
      --dataset_repeat "$DATASET_REPEAT" \
      --model_id_with_origin_paths "$DIT_MODEL" \
      --processor_path "$PROCESSOR" \
      --learning_rate "$LEARNING_RATE" \
      --num_epochs "$NUM_EPOCHS" \
      --remove_prefix_in_ckpt "pipe.dit." \
      --output_path "$output_dir" \
      --lora_base_model "dit" \
      --lora_target_modules "$LORA_TARGETS" \
      --lora_rank "$LORA_RANK" \
      --use_gradient_checkpointing_offload \
      --find_unused_parameters \
      --task "sft:train" || return $?

  if [[ $DRY_RUN != 1 ]]; then
    touch "$output_dir/.train-complete"
  fi
}

run_dataset() {
  local metadata_file=$1
  local dataset_dir
  local dataset_name
  local cache_dir
  local output_dir

  dataset_dir=$(dirname -- "$metadata_file")
  dataset_name=$(basename -- "$dataset_dir")
  cache_dir="$CACHE_ROOT/$dataset_name"
  output_dir="$OUTPUT_ROOT/$dataset_name"

  echo
  echo "============================================================"
  echo "Dataset: $dataset_name"
  echo "Source:  $dataset_dir"
  echo "Cache:   $cache_dir"
  echo "Output:  $output_dir"
  echo "============================================================"

  if [[ $STAGE == both || $STAGE == cache ]]; then
    run_cache_stage "$dataset_dir" "$metadata_file" "$cache_dir" || return $?
  fi
  if [[ $STAGE == both || $STAGE == train ]]; then
    run_train_stage "$cache_dir" "$output_dir" || return $?
  fi
}

echo "Found ${#metadata_files[@]} dataset(s) under $DATA_ROOT"
echo "Stage selection: $STAGE"
echo "Resolution: $RESOLUTION_MODE"

failed_datasets=()
for metadata_file in "${metadata_files[@]}"; do
  dataset_name=$(basename -- "$(dirname -- "$metadata_file")")
  if ! run_dataset "$metadata_file"; then
    failed_datasets+=("$dataset_name")
    echo "Error: dataset failed: $dataset_name" >&2
    if [[ $CONTINUE_ON_ERROR != 1 ]]; then
      exit 1
    fi
  fi
done

if (( ${#failed_datasets[@]} > 0 )); then
  echo
  echo "Batch finished with failures: ${failed_datasets[*]}" >&2
  exit 1
fi

echo
echo "Batch finished successfully."
