#!/usr/bin/env bash

export FACEID_WEIGHTS="/data-training/hjq/DiffSynth-Studio/faceid_model/backbone.pth"
export FACEID_DETECTOR_PATH="/data-training/hjq/DiffSynth-Studio/faceid_model/det_10g.onnx"
export FACEID_VAE_PATH="/data-training/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors"
export MODEL_BASE_PATH="/data-training/models"
export DATASET_BASE_PATH="/data-training/multi_face"
export DATASET_METADATA_PATH="/data-training/multi_face/metadata_faceid_1000.csv"
export CACHE_PATH="/data-training/multi_face/cache_faceid_f81_p589824_sp1000_sim0.5"
