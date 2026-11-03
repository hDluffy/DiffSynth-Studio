#!/usr/bin/env bash
# Usage: bash examples/wanvideo/model_training/faceid/run.sh cache|train
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
MODE=${1:-}
[[ "$MODE" == cache || "$MODE" == train ]] || { echo 'Usage: run.sh cache|train' >&2; exit 2; }
: "${FACEID_WEIGHTS:?Set FACEID_WEIGHTS to a local arcface_torch iResNet100 backbone.pth}"
: "${FACEID_VAE_PATH:?Set FACEID_VAE_PATH to the local Wan2.1 VAE used by preprocessing}"
: "${CACHE_PATH:?Set CACHE_PATH to a new FaceID feature-cache directory}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH=${MODEL_BASE_PATH:-/data-training/models}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
NUM_PROCESSES=${NUM_PROCESSES:-8}
DIT_ORIGIN=${DIT_ORIGIN:-Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors}
TEXT_ORIGIN=${TEXT_ORIGIN:-Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth}
AUDIO_ORIGIN=${AUDIO_ORIGIN:-Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors}
COMMON=(
  --faceid_weights "$FACEID_WEIGHTS" --faceid_vae_path "$FACEID_VAE_PATH"
  --num_frames 81 --min_num_frames 81 --frame_rate 16 --fix_frame_rate True
  --frame_count_stride 4 --frame_count_remainder 1 --frame_count_rounding floor
  --max_frame_padding "${MAX_FRAME_PADDING:-1}" --max_pixels "${MAX_PIXELS:-589824}"
  --lora_base_model dit --lora_target_modules 'q,k,v,o,ffn.0,ffn.2' --lora_rank "${LORA_RANK:-32}"
  --extra_inputs input_image,input_audio --use_gradient_checkpointing_offload
  --s2v_ref_rope_mode "${S2V_REF_ROPE_MODE:-source_id_local}"
  --remove_prefix_in_ckpt pipe.dit.
)
if [[ "$MODE" == cache ]]; then
  : "${DATASET_BASE_PATH:?Set DATASET_BASE_PATH}"
  : "${DATASET_METADATA_PATH:?Set DATASET_METADATA_PATH}"
  : "${FACEID_DETECTOR_PATH:?Set FACEID_DETECTOR_PATH to local SCRFD det_10g.onnx}"
  VAE_JSON=$(python -c 'import json,sys; print(json.dumps([sys.argv[1]]))' "$FACEID_VAE_PATH")
  ARGS=(
    --task sft:data_process --dataset_base_path "$DATASET_BASE_PATH"
    --dataset_metadata_path "$DATASET_METADATA_PATH" --data_file_keys video,input_audio
    --model_id_with_origin_paths "$DIT_ORIGIN,$TEXT_ORIGIN,$AUDIO_ORIGIN" --model_paths "$VAE_JSON"
    --audio_processor_path "${AUDIO_PROCESSOR_PATH:-Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/}"
    --faceid_detector_path "$FACEID_DETECTOR_PATH" --tiled
    --faceid_min_face_size "${FACEID_MIN_FACE_SIZE:-48}"
    --faceid_min_similarity "${FACEID_MIN_SIMILARITY:-0.2}"
    --faceid_det_threshold "${FACEID_DET_THRESHOLD:-0.6}"
    --output_path "$CACHE_PATH"
  )
  if [[ ${RESUME_FEATURE_CACHE:-0} == 1 ]]; then ARGS+=(--resume_feature_cache); fi
  CONFIG=${DATA_PROCESS_CONFIG:-examples/wanvideo/model_training/full/accelerate_config_data_process.yaml}
else
  ARGS=(
    --task sft:train --dataset_base_path "$CACHE_PATH" --require_cache_manifest
    --model_id_with_origin_paths "$DIT_ORIGIN"
    --learning_rate "${LEARNING_RATE:-1e-5}" --num_epochs "${NUM_EPOCHS:-10}"
    --save_steps "${SAVE_STEPS:-50}" --gradient_accumulation_steps 1
    --faceid_weight "${FACEID_WEIGHT:-0.05}" --faceid_warmup_steps "${FACEID_WARMUP_STEPS:-100}"
    --faceid_every "${FACEID_EVERY:-2}" --faceid_frames "${FACEID_FRAMES:-2}"
    --faceid_sigma_min "${FACEID_SIGMA_MIN:-0.1}" --faceid_sigma_max "${FACEID_SIGMA_MAX:-0.5}"
    --sigma_sampling_mode "${SIGMA_SAMPLING_MODE:-original}"
    --sigma_sampling_min "${SIGMA_SAMPLING_MIN:-${FACEID_SIGMA_MIN:-0.1}}"
    --sigma_sampling_max "${SIGMA_SAMPLING_MAX:-${FACEID_SIGMA_MAX:-0.5}}"
    --sigma_sampling_prob "${SIGMA_SAMPLING_PROB:-0.5}"
    --output_path "${OUTPUT_PATH:-./models/train/Wan2.2-S2V-14B_faceid_lora}"
    --enable_tensorboard_log
  )
  if [[ -n ${BASE_CHECKPOINT:-} ]]; then ARGS+=(--resume_from_checkpoint "$BASE_CHECKPOINT"); fi
  if [[ -n ${LORA_CHECKPOINT:-} ]]; then ARGS+=(--lora_checkpoint "$LORA_CHECKPOINT"); fi
  CONFIG=${TRAIN_CONFIG:-examples/wanvideo/model_training/full/accelerate_config_zero3.yaml}
fi
"$ACCELERATE_BIN" launch --config_file "$CONFIG" --num_processes "$NUM_PROCESSES" \
  examples/wanvideo/model_training/train_faceid.py "${COMMON[@]}" "${ARGS[@]}"
