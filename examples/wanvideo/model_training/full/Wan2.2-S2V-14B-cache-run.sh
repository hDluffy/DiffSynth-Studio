#!/usr/bin/env bash
set -euo pipefail

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
DATA_PROCESS_CONFIG_FILE=${DATA_PROCESS_CONFIG_FILE:-${CONFIG_FILE:-examples/wanvideo/model_training/full/accelerate_config_data_process.yaml}}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-examples/wanvideo/model_training/train.py}
MODEL_BASE_PATH=${MODEL_BASE_PATH:-${DIFFSYNTH_MODEL_BASE_PATH:-./models}}
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE_PATH}"

DATASET_BASE_PATH=${DATASET_BASE_PATH:-./data/test_data}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/metadata.csv}
DATA_FILE_KEYS=${DATA_FILE_KEYS:-video,input_audio}
DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS:-0}

HEIGHT=${HEIGHT:-}
WIDTH=${WIDTH:-}
MAX_PIXELS=${MAX_PIXELS:-589824}
NUM_FRAMES=${NUM_FRAMES:-113}
MIN_NUM_FRAMES=${MIN_NUM_FRAMES:-81}
FRAME_RATE=${FRAME_RATE:-16}
FIX_FRAME_RATE=${FIX_FRAME_RATE:-1}
FRAME_COUNT_STRIDE=${FRAME_COUNT_STRIDE:-16}
FRAME_COUNT_REMAINDER=${FRAME_COUNT_REMAINDER:-1}
FRAME_COUNT_ROUNDING=${FRAME_COUNT_ROUNDING:-nearest}
MAX_FRAME_PADDING=${MAX_FRAME_PADDING:-8}
AUDIO_SAMPLE_RATE=${AUDIO_SAMPLE_RATE:-16000}
AUDIO_DURATION_POLICY=${AUDIO_DURATION_POLICY:-trim_pad}
AUDIO_DURATION_TOLERANCE_SECONDS=${AUDIO_DURATION_TOLERANCE_SECONDS:-0.05}
MAX_AUDIO_PADDING_SECONDS=${MAX_AUDIO_PADDING_SECONDS:-0.5}
MAX_AUDIO_TRIMMING_SECONDS=${MAX_AUDIO_TRIMMING_SECONDS:-}
DATA_PROCESSING_LOG_SAMPLES=${DATA_PROCESSING_LOG_SAMPLES:-8}
RESUME_FEATURE_CACHE=${RESUME_FEATURE_CACHE:-0}
TILED=${TILED:-1}
TILE_SIZE=${TILE_SIZE:-30,52}
TILE_STRIDE=${TILE_STRIDE:-15,26}

if [ -n "${HEIGHT}" ] && [ -n "${WIDTH}" ]; then
  DATA_FEATURE_SIZE_TAG=${DATA_FEATURE_SIZE_TAG:-${HEIGHT}x${WIDTH}_f${MIN_NUM_FRAMES}-${NUM_FRAMES}_${FRAME_COUNT_STRIDE}n${FRAME_COUNT_REMAINDER}_fps${FRAME_RATE}}
else
  DATA_FEATURE_SIZE_TAG=${DATA_FEATURE_SIZE_TAG:-max_pixels_${MAX_PIXELS}_f${MIN_NUM_FRAMES}-${NUM_FRAMES}_${FRAME_COUNT_STRIDE}n${FRAME_COUNT_REMAINDER}_fps${FRAME_RATE}}
fi
DATA_FEATURE_CACHE_PATH=${DATA_FEATURE_CACHE_PATH:-${DATASET_BASE_PATH}/Wan2.2-S2V-14B_full_${DATA_FEATURE_SIZE_TAG}_features}

DIT_MODEL_ID_WITH_ORIGIN_PATH=${DIT_MODEL_ID_WITH_ORIGIN_PATH:-Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors}
MODEL_ID_WITH_ORIGIN_PATHS=${MODEL_ID_WITH_ORIGIN_PATHS:-${DIT_MODEL_ID_WITH_ORIGIN_PATH},Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth}
AUDIO_PROCESSOR_PATH=${AUDIO_PROCESSOR_PATH:-Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/}
OFFLOAD_MODELS=${OFFLOAD_MODELS:-${MODEL_ID_WITH_ORIGIN_PATHS%%,*}}
FP8_MODELS=${FP8_MODELS:-}
TRAINABLE_MODELS=${TRAINABLE_MODELS:-dit}
REMOVE_PREFIX_IN_CKPT=${REMOVE_PREFIX_IN_CKPT:-pipe.dit.}
EXTRA_INPUTS=${EXTRA_INPUTS:-input_image,input_audio}
USE_GRADIENT_CHECKPOINTING_OFFLOAD=${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-1}
NUM_PROCESSES=${NUM_PROCESSES:-}
S2V_REF_ROPE_MODE=${S2V_REF_ROPE_MODE:-legacy_time_offset}
S2V_REF_SOURCE_ID=${S2V_REF_SOURCE_ID:-1.0}
S2V_REF_ROPE_THETA=${S2V_REF_ROPE_THETA:-10000.0}
S2V_REF_TIME_BASE=${S2V_REF_TIME_BASE:-30}
S2V_REF_TIME_MARGIN=${S2V_REF_TIME_MARGIN:-9}

export PYTHONPATH="${PYTHONPATH:-.}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_accelerate() {
  local config_file="$1"
  shift
  local cmd=("${ACCELERATE_BIN}" launch --config_file "${config_file}")
  if [ -n "${NUM_PROCESSES}" ]; then
    cmd+=(--num_processes "${NUM_PROCESSES}")
  fi
  cmd+=("$@")
  "${cmd[@]}"
}

SIZE_ARGS=(
  --max_pixels "${MAX_PIXELS}"
  --num_frames "${NUM_FRAMES}"
  --frame_rate "${FRAME_RATE}"
  --frame_count_stride "${FRAME_COUNT_STRIDE}"
  --frame_count_remainder "${FRAME_COUNT_REMAINDER}"
  --frame_count_rounding "${FRAME_COUNT_ROUNDING}"
  --min_num_frames "${MIN_NUM_FRAMES}"
  --max_frame_padding "${MAX_FRAME_PADDING}"
  --audio_sample_rate "${AUDIO_SAMPLE_RATE}"
  --audio_duration_policy "${AUDIO_DURATION_POLICY}"
  --audio_duration_tolerance_seconds "${AUDIO_DURATION_TOLERANCE_SECONDS}"
  --max_audio_padding_seconds "${MAX_AUDIO_PADDING_SECONDS}"
  --data_processing_log_samples "${DATA_PROCESSING_LOG_SAMPLES}"
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
if [ -n "${MAX_AUDIO_TRIMMING_SECONDS}" ]; then
  SIZE_ARGS+=(--max_audio_trimming_seconds "${MAX_AUDIO_TRIMMING_SECONDS}")
fi

TILE_ARGS=()
if [ "${TILED}" = "1" ] || [ "${TILED}" = "true" ] || [ "${TILED}" = "True" ]; then
  TILE_ARGS+=(--tiled --tile_size "${TILE_SIZE}" --tile_stride "${TILE_STRIDE}")
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

RAW_DATA_ARGS=(
  --dataset_base_path "${DATASET_BASE_PATH}"
  --dataset_metadata_path "${DATASET_METADATA_PATH}"
  --data_file_keys "${DATA_FILE_KEYS}"
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  "${SIZE_ARGS[@]}"
  "${TILE_ARGS[@]}"
)

echo "[Wan2.2-S2V] Pre-extracting data features:"
echo "  dataset_base_path: ${DATASET_BASE_PATH}"
echo "  dataset_metadata_path: ${DATASET_METADATA_PATH}"
echo "  data_file_keys: ${DATA_FILE_KEYS}"
echo "  config_file: ${DATA_PROCESS_CONFIG_FILE}"
echo "  feature_cache_path: ${DATA_FEATURE_CACHE_PATH}"
echo "  max_pixels: ${MAX_PIXELS}"
echo "  size: ${HEIGHT:-auto}x${WIDTH:-auto}, frames=${MIN_NUM_FRAMES}-${NUM_FRAMES}"
echo "  frame_count: ${FRAME_COUNT_STRIDE}n+${FRAME_COUNT_REMAINDER}, rounding=${FRAME_COUNT_ROUNDING}, max_padding=${MAX_FRAME_PADDING}"
echo "  frame_rate: ${FRAME_RATE}"
echo "  audio: sample_rate=${AUDIO_SAMPLE_RATE}, policy=${AUDIO_DURATION_POLICY}, tolerance=${AUDIO_DURATION_TOLERANCE_SECONDS}s, max_padding=${MAX_AUDIO_PADDING_SECONDS}s"
echo "  fix_frame_rate: ${FIX_FRAME_RATE}"
echo "  tiled: ${TILED}"
echo "  tile_size: ${TILE_SIZE}"
echo "  tile_stride: ${TILE_STRIDE}"
echo "  offload_models: ${OFFLOAD_MODELS:-none}"
echo "  fp8_models: ${FP8_MODELS:-none}"
echo "  s2v_ref_rope_mode: ${S2V_REF_ROPE_MODE}"
echo "  resume_feature_cache: ${RESUME_FEATURE_CACHE}"
echo "  pytorch_cuda_alloc_conf: ${PYTORCH_CUDA_ALLOC_CONF}"

RESUME_CACHE_ARGS=()
if [ "${RESUME_FEATURE_CACHE}" = "1" ] || [ "${RESUME_FEATURE_CACHE}" = "true" ] || [ "${RESUME_FEATURE_CACHE}" = "True" ]; then
  RESUME_CACHE_ARGS+=(--resume_feature_cache)
fi

run_accelerate "${DATA_PROCESS_CONFIG_FILE}" "${TRAIN_SCRIPT}" \
  "${RAW_DATA_ARGS[@]}" \
  --dataset_repeat 1 \
  "${MODEL_ARGS[@]}" \
  "${RESUME_CACHE_ARGS[@]}" \
  --task sft:data_process \
  --output_path "${DATA_FEATURE_CACHE_PATH}"
