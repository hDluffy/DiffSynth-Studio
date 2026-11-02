#modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "wanvideo/Wan2.2-S2V-14B/*" --local_dir ./data/diffsynth_example_dataset
MODEL_BASE_PATH=${MODEL_BASE_PATH:-${DIFFSYNTH_MODEL_BASE_PATH:-/data-training/models}}
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE_PATH}"

S2V_REF_ROPE_MODE=${S2V_REF_ROPE_MODE:-source_id_local}
S2V_REF_SOURCE_ID=${S2V_REF_SOURCE_ID:-1.0}
S2V_REF_ROPE_THETA=${S2V_REF_ROPE_THETA:-10000.0}
S2V_REF_TIME_BASE=${S2V_REF_TIME_BASE:-30}
S2V_REF_TIME_MARGIN=${S2V_REF_TIME_MARGIN:-9}

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path /data-training/new_train_data \
  --dataset_metadata_path /data-training/hjq/DiffSynth-Studio/metadata_9s-10s.csv  \
  --data_file_keys "video,input_audio" \
  --max_pixels 589824 \
  --num_frames 145 \
  --dataset_repeat 1 \
  --frame_rate 16 \
  --fix_frame_rate True \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \
  --audio_processor_path "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \
  --learning_rate 1e-4 \
  --num_epochs 100 \
  --save_steps 50 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-S2V-14B_lora_9s" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,input_audio" \
  --s2v_ref_rope_mode "${S2V_REF_ROPE_MODE}" \
  --s2v_ref_source_id "${S2V_REF_SOURCE_ID}" \
  --s2v_ref_rope_theta "${S2V_REF_ROPE_THETA}" \
  --s2v_ref_time_base "${S2V_REF_TIME_BASE}" \
  --s2v_ref_time_margin "${S2V_REF_TIME_MARGIN}" \
  --use_gradient_checkpointing_offload
