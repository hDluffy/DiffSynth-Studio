#!/usr/bin/env python3
"""Convert DiffSynth/PEFT LoRA keys without loading tensors or changing weights.

Usage:
  python script/convert_lora_to_comfyui.py input.safetensors
  python script/convert_lora_to_comfyui.py input.safetensors -o output.safetensors

Only Python's standard library is required. Existing different outputs are never
replaced; an already matching output is verified and reused.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile


SUFFIXES = {
    '.lora_A.default.weight': '.lora_down.weight',
    '.lora_B.default.weight': '.lora_up.weight',
    '.lora_A.weight': '.lora_down.weight',
    '.lora_B.weight': '.lora_up.weight',
    '.lora_down.weight': '.lora_down.weight',
    '.lora_up.weight': '.lora_up.weight',
    '.alpha': '.alpha',
}
DTYPE_BYTES = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8}


def read_header(path):
    size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Invalid safetensors header')
        length = struct.unpack('<Q', prefix)[0]
        if length > min(size - 8, 100_000_000):
            raise ValueError('Invalid safetensors header length')
        header = json.loads(stream.read(length))
    return header, length + 8


def payload_hash(path, offset):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        stream.seek(offset)
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def converted_header(header, payload_size):
    converted = {}
    for key, value in header.items():
        if key == '__metadata__':
            converted[key] = value
            continue
        suffix = next((s for s in SUFFIXES if key.endswith(s)), None)
        if suffix is None:
            raise ValueError(f'Unsupported key (expected a LoRA checkpoint): {key}')
        module = key[:-len(suffix)]
        if module.startswith('pipe.dit.'):
            module = module[len('pipe.dit.'):]
        if not module.startswith('diffusion_model.'):
            module = 'diffusion_model.' + module
        target = module + SUFFIXES[suffix]
        if target in converted:
            raise ValueError(f'Key collision: {target}')
        converted[target] = value
    tensors = {k: v for k, v in converted.items() if k != '__metadata__'}
    down_keys = [k for k in tensors if k.endswith('.lora_down.weight')]
    if not down_keys:
        raise ValueError('No LoRA pairs found')
    end = 0
    for key, value in sorted(tensors.items(), key=lambda item: item[1]['data_offsets'][0]):
        start, stop = value['data_offsets']
        shape = value['shape']
        if value['dtype'] not in DTYPE_BYTES:
            raise ValueError(f'Unsupported floating dtype: {key}: {value["dtype"]}')
        if start != end or stop - start != math.prod(shape) * DTYPE_BYTES[value['dtype']]:
            raise ValueError(f'Invalid data offsets: {key}')
        end = stop
        if key.endswith('.alpha'):
            if math.prod(shape) != 1:
                raise ValueError(f'Alpha must have one element: {key}')
            if key[:-len('.alpha')] + '.lora_down.weight' not in tensors:
                raise ValueError(f'Alpha has no LoRA pair: {key}')
            continue
        if len(shape) != 2 or any(d <= 0 for d in shape):
            raise ValueError(f'Expected nonempty linear LoRA matrix: {key}: {shape}')
        if key.endswith('.lora_down.weight'):
            other = key.replace('.lora_down.weight', '.lora_up.weight')
            if other not in tensors or len(tensors[other]['shape']) != 2 or shape[0] != tensors[other]['shape'][1]:
                raise ValueError(f'Missing/mismatched LoRA pair: {key}')
        else:
            other = key.replace('.lora_up.weight', '.lora_down.weight')
            if other not in tensors:
                raise ValueError(f'Unpaired LoRA up matrix: {key}')
    if end != payload_size:
        raise ValueError('Tensor payload length does not match file size')
    return converted, len(down_keys)


def convert(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if source == destination:
        raise ValueError('Input and output must be different files')
    header, offset = read_header(source)
    converted, pairs = converted_header(header, source.stat().st_size - offset)
    digest = payload_hash(source, offset)
    if destination.exists():
        existing, existing_offset = read_header(destination)
        if existing != converted or payload_hash(destination, existing_offset) != digest:
            raise FileExistsError(f'Output exists with different content; choose another output: {destination}')
        print(f'Existing output verified: {destination}')
    else:
        raw = json.dumps(converted, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        raw += b' ' * (-len(raw) % 8)
        # Verify a temporary file before publishing; hard-link publication cannot
        # overwrite a destination created concurrently by another process.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.' + destination.name + '.',
                                             suffix='.tmp', delete=False) as out:
                temporary = Path(out.name)
                out.write(struct.pack('<Q', len(raw)))
                out.write(raw)
                with source.open('rb') as stream:
                    stream.seek(offset)
                    shutil.copyfileobj(stream, out, 8 * 1024 * 1024)
            if read_header(temporary)[0] != converted or payload_hash(temporary, 8 + len(raw)) != digest:
                raise RuntimeError('Converted file verification failed')
            os.link(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f'Created: {destination}')
    print(f'LoRA pairs: {pairs}; payload SHA256: {digest}')
    print('Weight data verified unchanged. Source file preserved.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, help='DiffSynth/PEFT LoRA safetensors file')
    parser.add_argument('-o', '--output', type=Path, help='Default: <input-stem>-comfyui.safetensors')
    args = parser.parse_args()
    output = args.output or args.input.with_name(args.input.stem + '-comfyui.safetensors')
    try:
        convert(args.input, output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f'Conversion failed: {error}\n')


if __name__ == '__main__':
    main()
