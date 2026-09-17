#!/usr/bin/env bash
# Run once on node2; node3 is started over passwordless SSH.

NODES="node2 node3" MASTER_NODE=node2 LOCAL_NODE=node2 \
TRAIN_LAUNCHER=examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode-cache-run.sh \
ACCELERATE_BIN=/data-training/miniconda/bin/accelerate \
MODEL_BASE_PATH=/data-training/models COMM_IFNAME=bond0 \
DATASET_BASE_PATH=/data-training/train_data_5s \
DATA_FEATURE_CACHE_PATH=/data-training/train_data_5s/cache_s2v_f81-113_16n1_fps16_10s \
NUM_FRAMES=113 MIN_NUM_FRAMES=81 FRAME_COUNT_STRIDE=16 FRAME_COUNT_REMAINDER=1 \
FRAME_COUNT_ROUNDING=nearest MAX_FRAME_PADDING=8 \
OUTPUT_PATH=./models/train/Wan2.2-S2V-14B_variable_length \
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-/data-training/hjq/DiffSynth-Studio/models/train/merge_lv2_sa_step-600.safetensors}" \
ENABLE_TENSORBOARD_LOG=1 \
USE_GRADIENT_CHECKPOINTING_OFFLOAD="${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-1}" \
S2V_REF_ROPE_MODE=source_id_local TRAIN_LOG=train-cache.log \
bash script/launch_wan22_s2v_multinode.sh "$@"
