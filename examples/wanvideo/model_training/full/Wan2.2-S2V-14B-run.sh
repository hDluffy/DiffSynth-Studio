#!/usr/bin/env bash
set -euo pipefail

ACCELERATE_BIN=${ACCELERATE_BIN:-/data-training/miniconda/bin/accelerate}
CONFIG_FILE=${CONFIG_FILE:-examples/wanvideo/model_training/full/accelerate_config_zero3.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-examples/wanvideo/model_training/train.py}

DATASET_BASE_PATH=${DATASET_BASE_PATH:-/data-training/train_data_5031}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/metadata.csv}
DATA_FILE_KEYS=${DATA_FILE_KEYS:-video,input_audio}
MAX_PIXELS=${MAX_PIXELS:-589824}
NUM_FRAMES=${NUM_FRAMES:-81}
FRAME_RATE=${FRAME_RATE:-16}
DATASET_REPEAT=${DATASET_REPEAT:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
NUM_EPOCHS=${NUM_EPOCHS:-100}
SAVE_STEPS=${SAVE_STEPS:-200}
OUTPUT_PATH=${OUTPUT_PATH:-./models/train/Wan2.2-S2V-14B_full}
EXTRA_INPUTS=${EXTRA_INPUTS:-input_image,input_audio}
USE_GRADIENT_CHECKPOINTING_OFFLOAD=${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-1}
NUM_PROCESSES=${NUM_PROCESSES:-}

S2V_REF_ROPE_MODE=${S2V_REF_ROPE_MODE:-source_id_local}
S2V_REF_SOURCE_ID=${S2V_REF_SOURCE_ID:-1.0}
S2V_REF_ROPE_THETA=${S2V_REF_ROPE_THETA:-10000.0}
S2V_REF_TIME_BASE=${S2V_REF_TIME_BASE:-30}
S2V_REF_TIME_MARGIN=${S2V_REF_TIME_MARGIN:-9}

export PYTHONPATH="${PYTHONPATH:-.}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LAUNCH_ARGS=(--config_file "${CONFIG_FILE}")
if [ -n "${NUM_PROCESSES}" ]; then
  LAUNCH_ARGS+=(--num_processes "${NUM_PROCESSES}")
fi

MODEL_ARGS=(
  --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth"
  --audio_processor_path "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/"
  --trainable_models "dit"
  --remove_prefix_in_ckpt "pipe.dit."
  --extra_inputs "${EXTRA_INPUTS}"
  --s2v_ref_rope_mode "${S2V_REF_ROPE_MODE}"
  --s2v_ref_source_id "${S2V_REF_SOURCE_ID}"
  --s2v_ref_rope_theta "${S2V_REF_ROPE_THETA}"
  --s2v_ref_time_base "${S2V_REF_TIME_BASE}"
  --s2v_ref_time_margin "${S2V_REF_TIME_MARGIN}"
)
if [ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD}" = "1" ] || [ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD}" = "true" ] || [ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD}" = "True" ]; then
  MODEL_ARGS+=(--use_gradient_checkpointing_offload)
else
  MODEL_ARGS+=(--use_gradient_checkpointing)
fi

echo "[Wan2.2-S2V] dataset_base_path=${DATASET_BASE_PATH}"
echo "[Wan2.2-S2V] num_frames=${NUM_FRAMES}, max_pixels=${MAX_PIXELS}, s2v_ref_rope_mode=${S2V_REF_ROPE_MODE}"
echo "[Wan2.2-S2V] config_file=${CONFIG_FILE}, gradient_checkpointing_offload=${USE_GRADIENT_CHECKPOINTING_OFFLOAD}"

"${ACCELERATE_BIN}" launch "${LAUNCH_ARGS[@]}" "${TRAIN_SCRIPT}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "${DATA_FILE_KEYS}" \
  --max_pixels "${MAX_PIXELS}" \
  --num_frames "${NUM_FRAMES}" \
  --frame_rate "${FRAME_RATE}" \
  --fix_frame_rate True \
  --dataset_repeat "${DATASET_REPEAT}" \
  "${MODEL_ARGS[@]}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --save_steps "${SAVE_STEPS}" \
  --output_path "${OUTPUT_PATH}"
