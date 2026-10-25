#!/usr/bin/env python3
"""Convert a DiffSynth MiniMax-H3 LoRA checkpoint for ComfyUI.

DiffSynth exports MiniMax-H3 LoRA keys such as::

    blocks.0.attn.qkv_proj.lora_A.default.weight

ComfyUI's model key map expects the same key under ``diffusion_model.``::

    diffusion_model.blocks.0.attn.qkv_proj.lora_A.default.weight

This script only changes tensor names. Tensor values, shapes, dtypes, and source
metadata are preserved.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

from safetensors import safe_open
from safetensors.torch import load_file, save_file


COMFY_PREFIX = "diffusion_model."
LORA_A_SUFFIXES = (".lora_A.default.weight", ".lora_A.weight")
LORA_B_SUFFIXES = (".lora_B.default.weight", ".lora_B.weight")


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}-comfy{input_path.suffix}")


def strip_one_suffix(key: str, suffixes: tuple[str, ...]) -> str | None:
    for suffix in suffixes:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def validate_lora_pairs(keys: list[str]) -> int:
    lora_a = {
        prefix
        for key in keys
        if (prefix := strip_one_suffix(key, LORA_A_SUFFIXES)) is not None
    }
    lora_b = {
        prefix
        for key in keys
        if (prefix := strip_one_suffix(key, LORA_B_SUFFIXES)) is not None
    }
    if not lora_a and not lora_b:
        raise ValueError("No LoRA A/B tensors were found in the input checkpoint.")
    if lora_a != lora_b:
        missing_b = sorted(lora_a - lora_b)
        missing_a = sorted(lora_b - lora_a)
        raise ValueError(
            "Unpaired LoRA tensors: "
            f"missing B={missing_b[:5]}, missing A={missing_a[:5]}"
        )
    return len(lora_a)


def convert(input_path: Path, output_path: Path, force: bool = False) -> None:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input checkpoint does not exist: {input_path}")
    if input_path.suffix != ".safetensors":
        raise ValueError(f"Input must be a .safetensors file: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Pass --force to replace it."
        )

    with safe_open(input_path, framework="pt", device="cpu") as source:
        source_keys = list(source.keys())
        metadata = dict(source.metadata() or {})

    pair_count = validate_lora_pairs(source_keys)
    state_dict = load_file(input_path, device="cpu")
    converted = {
        key if key.startswith(COMFY_PREFIX) else f"{COMFY_PREFIX}{key}": tensor
        for key, tensor in state_dict.items()
    }
    if len(converted) != len(state_dict):
        raise ValueError("Key collision occurred after adding the ComfyUI prefix.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.setdefault("format", "pt")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        save_file(converted, temporary_path, metadata=metadata)
        with safe_open(temporary_path, framework="pt", device="cpu") as result:
            result_keys = list(result.keys())
        if len(result_keys) != len(source_keys):
            raise RuntimeError(
                f"Validation failed: expected {len(source_keys)} tensors, "
                f"found {len(result_keys)}."
            )
        if not all(key.startswith(COMFY_PREFIX) for key in result_keys):
            raise RuntimeError("Validation failed: an output key lacks diffusion_model prefix.")
        validate_lora_pairs(result_keys)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    size_mib = output_path.stat().st_size / (1024**2)
    print(f"Converted: {input_path}")
    print(f"Output:    {output_path}")
    print(f"Tensors:   {len(source_keys)} ({pair_count} LoRA target layers)")
    print(f"Size:      {size_mib:.1f} MiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add the diffusion_model. prefix required by ComfyUI to a "
            "DiffSynth MiniMax-H3 LoRA checkpoint."
        )
    )
    parser.add_argument("input", type=Path, help="DiffSynth LoRA .safetensors file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path (default: INPUT-comfy.safetensors)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output file if it already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.input)
    try:
        convert(args.input, output_path, force=args.force)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}")


if __name__ == "__main__":
    main()
