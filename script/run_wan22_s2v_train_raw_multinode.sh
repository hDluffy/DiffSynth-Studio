#!/usr/bin/env bash
# Fixed-length raw-data training. Use the cache scripts for mixed 81/97/113 frames.

NODES="node2 node3" MASTER_NODE=node2 LOCAL_NODE=node2 \
TRAIN_LAUNCHER=examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode-run.sh \
ACCELERATE_BIN=/data-training/miniconda/bin/accelerate \
MODEL_BASE_PATH=/data-training/models COMM_IFNAME=bond0 \
DATASET_BASE_PATH=/data-training/train_data_5s \
DATASET_METADATA_PATH=/data-training/train_data_5s/metadata.csv \
NUM_FRAMES=81 MIN_NUM_FRAMES=81 FRAME_RATE=16 \
OUTPUT_PATH=./models/train/Wan2.2-S2V-14B_raw_f81 \
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-/data-training/hjq/DiffSynth-Studio/models/train/merge_lv2_sa_step-600.safetensors}" \
ENABLE_TENSORBOARD_LOG=1 USE_GRADIENT_CHECKPOINTING_OFFLOAD=1 \
S2V_REF_ROPE_MODE=source_id_local TRAIN_LOG=train-raw.log \
bash script/launch_wan22_s2v_multinode.sh "$@"
