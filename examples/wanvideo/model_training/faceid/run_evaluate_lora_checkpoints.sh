#!/usr/bin/env bash
# Evaluate original training checkpoints, including base-only; no training is launched.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/data-training/models}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
LORA_DIR=${LORA_DIR:-models/train/Wan2.2-S2V-14B_faceid_lora_every1_faceidw0.1}
TRAIN_CONFIG=${TRAIN_CONFIG:-$LORA_DIR/faceid_training_config.json}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-a800_step-9000_sa_rope.safetensors}
OUTPUT=${OUTPUT:-diagnostics/faceid_checkpoints_$(date +%Y%m%d_%H%M%S)}
# Only step-N.safetensors, excluding step-N-comfyui.safetensors.
mapfile -t CHECKPOINTS < <(python - "$LORA_DIR" <<'PY'
import re,sys
from pathlib import Path
files=[p for p in Path(sys.argv[1]).glob('step-*.safetensors') if re.fullmatch(r'step-\d+\.safetensors',p.name)]
for path in sorted(files,key=lambda p:int(p.stem.split('-')[1])): print(path)
PY
)
if [[ ${#CHECKPOINTS[@]} == 0 ]]; then echo "No original step-N.safetensors files in $LORA_DIR" >&2; exit 2; fi
mkdir -p "$(dirname "$OUTPUT")"
python -u examples/wanvideo/model_training/faceid/evaluate_lora_checkpoints.py \
  --config "$TRAIN_CONFIG" --base-checkpoint "$BASE_CHECKPOINT" \
  --lora-checkpoints "${CHECKPOINTS[@]}" --device "${DEVICE:-cuda:0}" \
  --output "$OUTPUT" "$@" 2>&1 | tee "${OUTPUT}.log"
