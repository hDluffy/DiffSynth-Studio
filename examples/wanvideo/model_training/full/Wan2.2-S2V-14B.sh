#!/usr/bin/env bash
set -euo pipefail

ACCELERATE_BIN=${ACCELERATE_BIN:-/app/miniconda3/bin/accelerate}
CONFIG_FILE=${CONFIG_FILE:-examples/wanvideo/model_training/full/accelerate_config_14B.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-examples/wanvideo/model_training/train.py}

DATASET_BASE_PATH=${DATASET_BASE_PATH:-./data/test_data}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/metadata.csv}
DATA_FILE_KEYS=${DATA_FILE_KEYS:-video,input_audio}
DATASET_REPEAT=${DATASET_REPEAT:-5}
DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS:-0}

HEIGHT=${HEIGHT:-}
WIDTH=${WIDTH:-}
MAX_PIXELS=${MAX_PIXELS:-1048576}
NUM_FRAMES=${NUM_FRAMES:-81}
FRAME_RATE=${FRAME_RATE:-24}
FIX_FRAME_RATE=${FIX_FRAME_RATE:-0}

LEARNING_RATE=${LEARNING_RATE:-1e-5}
NUM_EPOCHS=${NUM_EPOCHS:-5}
OUTPUT_PATH=${OUTPUT_PATH:-./models/train/Wan2.2-S2V-14B_full}

# 0/false: normal training from raw data.
# 1/true: pre-extract data features first, then train from cached features.
# only: only run feature pre-extraction and skip training.
# cache: only train from an existing feature cache.
PRE_EXTRACT_DATA_FEATURES=${PRE_EXTRACT_DATA_FEATURES:-0}
if [ -n "${HEIGHT}" ] && [ -n "${WIDTH}" ]; then
  DATA_FEATURE_SIZE_TAG=${DATA_FEATURE_SIZE_TAG:-${HEIGHT}x${WIDTH}x${NUM_FRAMES}}
else
  DATA_FEATURE_SIZE_TAG=${DATA_FEATURE_SIZE_TAG:-max_pixels_${MAX_PIXELS}_frames_${NUM_FRAMES}}
fi
DATA_FEATURE_CACHE_PATH=${DATA_FEATURE_CACHE_PATH:-./models/cache/Wan2.2-S2V-14B_full_${DATA_FEATURE_SIZE_TAG}_features}

MODEL_ID_WITH_ORIGIN_PATHS=${MODEL_ID_WITH_ORIGIN_PATHS:-Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth}
AUDIO_PROCESSOR_PATH=${AUDIO_PROCESSOR_PATH:-Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/}
TRAINABLE_MODELS=${TRAINABLE_MODELS:-dit}
REMOVE_PREFIX_IN_CKPT=${REMOVE_PREFIX_IN_CKPT:-pipe.dit.}
EXTRA_INPUTS=${EXTRA_INPUTS:-input_image,input_audio}
USE_GRADIENT_CHECKPOINTING_OFFLOAD=${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-1}
NUM_PROCESSES=${NUM_PROCESSES:-}

export PYTHONPATH="${PYTHONPATH:-.}"

LAUNCH_ARGS=(--config_file "${CONFIG_FILE}")
if [ -n "${NUM_PROCESSES}" ]; then
  LAUNCH_ARGS+=(--num_processes "${NUM_PROCESSES}")
fi

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
