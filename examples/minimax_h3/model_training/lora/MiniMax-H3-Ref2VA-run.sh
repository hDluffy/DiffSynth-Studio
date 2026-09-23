#modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-Ref2VA/*" --local_dir ./data/diffsynth_example_dataset

# Optional: fuse the DeCFG training adapter into the DiT while training, for a better optimization landscape on this CFG-distilled base. Training only -- do not load it at inference.
# modelscope download --model DiffSynth-Studio/MiniMax-H3-TrainingAdapter --include model_ref2va.safetensors --local_dir ./models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter
#   --preset_lora_path "./models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter/model_ref2va.safetensors" \
#   --preset_lora_model "dit"

# stage 1 (data process)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/0x1300000000000816 \
  --dataset_metadata_path data/0x1300000000000816/metadata.json \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 736 \
  --width 1280 \
  --num_frames 362 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:Ref2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:Ref2VA/audio_vae/model.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Ref2VA-0x1300000000000816-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing_offload \
  --task "sft:data_process"

# stage 2 (train)
accelerate launch \
  --config_file examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-Ref2VA-0x1300000000000816-cache \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 736 \
  --width 1280 \
  --num_frames 362 \
  --dataset_repeat 5 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/transformer/model*.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Ref2VA/0x1300000000000816" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing_offload \
  --find_unused_parameters \
  --task "sft:train"
