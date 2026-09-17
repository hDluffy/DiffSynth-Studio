#!/usr/bin/env bash
# Run from the DiffSynth-Studio repository root.

ACCELERATE_BIN=/data-training/miniconda/bin/accelerate \
MODEL_BASE_PATH=/data-training/models \
DATASET_BASE_PATH=/data-training/train_data_5s \
DATASET_METADATA_PATH=/data-training/train_data_5s/metadata.csv \
DATA_FEATURE_CACHE_PATH=/data-training/train_data_5s/cache_s2v_f81-81_16n1_fps16_v1 \
NUM_PROCESSES=8 NUM_FRAMES=81 MIN_NUM_FRAMES=81 \
FRAME_COUNT_STRIDE=16 FRAME_COUNT_REMAINDER=1 FRAME_COUNT_ROUNDING=nearest \
MAX_FRAME_PADDING=8 S2V_REF_ROPE_MODE=source_id_local \
RESUME_FEATURE_CACHE="${RESUME_FEATURE_CACHE:-1}" \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-cache-run.sh
