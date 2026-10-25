#!/usr/bin/env python3
"""Generate metadata.json for a video training dataset.

Expected dataset layout::

    dataset_root/
    ├── videos/<sample_id>.mp4
    ├── prompts/<sample_id>.txt
    └── reference_images/<sample_id>.jpg
        or reference_images/<sample_id>_1.jpg, <sample_id>_2.png, ...

Execution example (run from the project root)::

    python tool/generate_metadata.py data/tmp03/kiva_tmp

To choose another output file or frame rate::

    python tool/generate_metadata.py data/tmp03/kiva_tmp \
        --output data/tmp03/kiva_tmp/metadata.json --frame-rate 24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metadata.json from videos, prompts, and reference_images."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Dataset directory containing videos/, prompts/, and reference_images/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: <dataset_root>/metadata.json).",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=24,
        help="frame_rate written to every item (default: 24).",
    )
    return parser.parse_args()


def relative_posix(path: Path, root: Path) -> str:
    """Return a JSON-friendly path relative to the dataset root."""
    return path.relative_to(root).as_posix()


def reference_sort_key(path: Path, sample_id: str) -> tuple[int, int, str]:
    """Sort the base image first, then numbered images in numeric order."""
    suffix = path.stem[len(sample_id) :]
    if not suffix:
        return (0, 0, path.name)
    if suffix.startswith("_") and suffix[1:].isdigit():
        return (1, int(suffix[1:]), path.name)
    return (2, 0, path.name)


def find_references(reference_dir: Path, sample_id: str) -> list[Path]:
    references = []
    for path in reference_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        suffix = path.stem[len(sample_id) :] if path.stem.startswith(sample_id) else None
        if suffix == "" or (
            suffix is not None
            and suffix.startswith("_")
            and suffix[1:].isdigit()
        ):
            references.append(path)

    return sorted(references, key=lambda path: reference_sort_key(path, sample_id))


def build_metadata(dataset_root: Path, frame_rate: int) -> list[dict[str, object]]:
    videos_dir = dataset_root / "videos"
    prompts_dir = dataset_root / "prompts"
    reference_dir = dataset_root / "reference_images"

    for directory in (videos_dir, prompts_dir, reference_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Required directory does not exist: {directory}")

    videos = sorted(
        path
        for path in videos_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise ValueError(f"No supported video files found in: {videos_dir}")

    metadata: list[dict[str, object]] = []
    missing_prompts: list[Path] = []

    for video_path in videos:
        sample_id = video_path.stem
        prompt_path = prompts_dir / f"{sample_id}.txt"
        if not prompt_path.is_file():
            missing_prompts.append(prompt_path)
            continue

        references = find_references(reference_dir, sample_id)
        video = relative_posix(video_path, dataset_root)
        metadata.append(
            {
                "video": video,
                "prompt": prompt_path.read_text(encoding="utf-8-sig").strip(),
                "input_audio": video,
                "references": [
                    {
                        "type": "image",
                        "image": relative_posix(image_path, dataset_root),
                    }
                    for image_path in references
                ],
                "frame_rate": frame_rate,
            }
        )

    if missing_prompts:
        paths = "\n".join(f"  - {path}" for path in missing_prompts)
        raise FileNotFoundError(f"Missing prompt files:\n{paths}")

    return metadata


def main() -> int:
    args = parse_args()
    if args.frame_rate <= 0:
        print("error: --frame-rate must be greater than 0", file=sys.stderr)
        return 2

    dataset_root = args.dataset_root.resolve()
    output_path = (args.output or dataset_root / "metadata.json").resolve()

    try:
        metadata = build_metadata(dataset_root, args.frame_rate)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    reference_count = sum(len(item["references"]) for item in metadata)
    print(
        f"Wrote {len(metadata)} items with {reference_count} reference images "
        f"to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
