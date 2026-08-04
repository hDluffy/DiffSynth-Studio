#!/usr/bin/env python3
"""Convert the 5S JSON list into Wan S2V-style metadata.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_INPUT = Path("/data/work/5S_Data/valid.json")
DEFAULT_OUTPUT = Path("/data/work/metadata.csv")
FIELDNAMES = ["video", "s2v_pose_video", "input_audio", "prompt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSON records with mp4/prompt fields to metadata.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSON path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--media-prefix",
        default="mv",
        help="Prefix added before media filenames in the CSV. Default: mv",
    )
    parser.add_argument(
        "--video-key",
        default="video",
        help="JSON key that contains the video filename. Default: video",
    )
    parser.add_argument(
        "--prompt-key",
        default="prompt",
        help="JSON key that contains the prompt text. Default: prompt",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("data", "items", "records"):
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list, or a dict with data/items/records list")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Record {index} is not a JSON object")
        records.append(item)
    return records


def prefixed_path(filename: str, prefix: str) -> str:
    raw_path = str(filename).strip().replace("\\", "/")
    if not raw_path:
        return ""

    path = PurePosixPath(raw_path)
    if path.is_absolute() or not prefix:
        return path.as_posix()

    prefix_path = PurePosixPath(prefix.strip("/"))
    if path.parts and path.parts[0] == prefix_path.as_posix():
        return path.as_posix()
    return (prefix_path / path).as_posix()


def derive_audio_path(video_path: str) -> str:
    path = PurePosixPath(video_path)
    return path.with_suffix(".mp3").as_posix()


def to_metadata_rows(
    records: list[dict[str, Any]],
    media_prefix: str,
    video_key: str,
    prompt_key: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        video_name = record.get(video_key)
        prompt = record.get(prompt_key)

        if not isinstance(video_name, str) or not video_name.strip():
            raise ValueError(f"Record {index} is missing non-empty '{video_key}'")
        if not isinstance(prompt, str):
            raise ValueError(f"Record {index} is missing string '{prompt_key}'")

        video = prefixed_path(video_name, media_prefix)
        rows.append(
            {
                "video": video,
                "s2v_pose_video": "",
                "input_audio": derive_audio_path(video),
                "prompt": prompt,
            }
        )
    return rows


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    rows = to_metadata_rows(records, args.media_prefix, args.video_key, args.prompt_key)
    write_metadata(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
