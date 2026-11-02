"""Inspect the exact cached data consumed by FaceID training, without loading GPUs."""
import argparse
import json
from collections import Counter
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('cache_path', type=Path)
args = parser.parse_args()
manifest = json.loads((args.cache_path / '_cache_manifest.json').read_text())
if manifest.get('status') != 'complete' or 'faceid' not in manifest.get('config', {}):
    raise SystemExit('Not a complete FaceID cache')
counts = Counter()
for path in sorted(args.cache_path.rglob('*.pth')):
    shared = torch.load(path, map_location='cpu', weights_only=True)[0]
    face = shared['faceid']
    if face['model_sha256'] != manifest['config']['faceid']['model_sha256']:
        raise ValueError(f'Fingerprint mismatch: {path}')
    counts['samples'] += 1
    counts['valid_reference_samples'] += int(face['reference_valid'])
    counts['valid_frame_annotations'] += int(face['valid'].sum())
    counts['samples_with_faceid_supervision'] += int(face['reference_valid'] and face['valid'].any())
    if shared['input_latents'].shape[2] != 21 or face['num_frames'] != 81:
        raise ValueError(f'Not an 81-frame sample: {path}')
    if not torch.isfinite(face['reference_embedding']).all():
        raise ValueError(f'Non-finite reference: {path}')
    if face['reference_valid'] and not torch.isclose(face['reference_embedding'].norm(), torch.tensor(1.), atol=1e-4):
        raise ValueError(f'Invalid reference embedding norm: {path}')
if counts['samples'] != manifest['source_items']:
    raise ValueError('Cache count differs from manifest')
print(json.dumps(dict(counts), indent=2))
if not counts['samples_with_faceid_supervision']:
    raise SystemExit('No usable FaceID samples. Check detection, reference images and thresholds before training.')
