#!/usr/bin/env bash
set -e

DATASET_BASE_PATH=${DATASET_BASE_PATH:-data/dpo/wan22-s2v}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/metadata.csv}
OUTPUT_PATH=${OUTPUT_PATH:-./models/train/Wan2.2-S2V-14B_dpo_lora}
HEIGHT=${HEIGHT:-448}
WIDTH=${WIDTH:-832}
NUM_FRAMES=${NUM_FRAMES:-81}
DATASET_REPEAT=${DATASET_REPEAT:-20}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
NUM_EPOCHS=${NUM_EPOCHS:-1}
LORA_RANK=${LORA_RANK:-32}
DPO_BETA=${DPO_BETA:-0.1}
DPO_LAMBDA_SFT=${DPO_LAMBDA_SFT:-0.1}
S2V_REF_ROPE_MODE=${S2V_REF_ROPE_MODE:-legacy_time_offset}
S2V_REF_SOURCE_ID=${S2V_REF_SOURCE_ID:-1.0}
S2V_REF_ROPE_THETA=${S2V_REF_ROPE_THETA:-10000.0}
S2V_REF_TIME_BASE=${S2V_REF_TIME_BASE:-30}
S2V_REF_TIME_MARGIN=${S2V_REF_TIME_MARGIN:-9}

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path "$DATASET_BASE_PATH" \
  --dataset_metadata_path "$DATASET_METADATA_PATH" \
  --data_file_keys "chosen_video,rejected_video,input_audio,s2v_pose_video,input_image" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --dataset_repeat "$DATASET_REPEAT" \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \
  --audio_processor_path "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \
  --learning_rate "$LEARNING_RATE" \
  --num_epochs "$NUM_EPOCHS" \
  --task dpo \
  --dpo_reference_free \
  --dpo_beta "$DPO_BETA" \
  --dpo_lambda_sft "$DPO_LAMBDA_SFT" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_PATH" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank "$LORA_RANK" \
  --extra_inputs "input_image,input_audio,s2v_pose_video" \
  --s2v_ref_rope_mode "$S2V_REF_ROPE_MODE" \
  --s2v_ref_source_id "$S2V_REF_SOURCE_ID" \
  --s2v_ref_rope_theta "$S2V_REF_ROPE_THETA" \
  --s2v_ref_time_base "$S2V_REF_TIME_BASE" \
  --s2v_ref_time_margin "$S2V_REF_TIME_MARGIN" \
  --use_gradient_checkpointing_offload
