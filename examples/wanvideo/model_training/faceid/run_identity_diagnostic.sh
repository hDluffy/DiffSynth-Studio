#!/usr/bin/env bash
# Override BASE_CHECKPOINT, LORA_CHECKPOINT, TRAIN_CONFIG, DEVICE, OUTPUT as needed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/data-training/models}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
OUTPUT=${OUTPUT:-diagnostics/faceid_threeway_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$(dirname "$OUTPUT")"
python -u examples/wanvideo/model_training/faceid/diagnose_identity.py \
  --config "${TRAIN_CONFIG:-models/train/Wan2.2-S2V-14B_faceid_lora_every1_faceidw0.1/faceid_training_config.json}" \
  --base-checkpoint "${BASE_CHECKPOINT:-a800_step-9000_sa_rope.safetensors}" \
  --lora-checkpoint "${LORA_CHECKPOINT:-models/train/Wan2.2-S2V-14B_faceid_lora_every1_faceidw0.1/step-200.safetensors}" \
  --device "${DEVICE:-cuda:0}" --output "$OUTPUT" "$@" 2>&1 | tee "${OUTPUT}.log"
python examples/wanvideo/model_training/faceid/summarize_identity.py "$OUTPUT"
cp examples/wanvideo/model_training/faceid/{diagnose_identity.py,summarize_identity.py,run_identity_diagnostic.sh} "$OUTPUT/"
