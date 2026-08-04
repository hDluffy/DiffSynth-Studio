#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/batch_mp4_to_mp3.sh INPUT_DIR OUTPUT_DIR

Extract synced MP3 audio from all .mp4 files in INPUT_DIR to OUTPUT_DIR:
  - only .mp4 files are processed
  - output .mp3 uses the same basename as the input .mp4
  - input mp4 should be the converted video with already-synced internal audio

Environment:
  AUDIO_BITRATE=192k     Output MP3 bitrate.
  AUDIO_RATE=44100       Output sample rate.

Example:
  scripts/batch_mp4_to_mp3.sh ./converted_mp4 ./output_mp3
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg is not installed or not in PATH." >&2
  exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "Error: ffprobe is not installed or not in PATH." >&2
  exit 1
fi

input_dir="${1%/}"
output_dir="${2%/}"
audio_bitrate="${AUDIO_BITRATE:-192k}"
audio_rate="${AUDIO_RATE:-44100}"

if [[ ! -d "$input_dir" ]]; then
  echo "Error: input directory does not exist: $input_dir" >&2
  exit 1
fi

mkdir -p "$output_dir"

shopt -s nullglob nocaseglob
mp4_files=("$input_dir"/*.mp4)

if [[ ${#mp4_files[@]} -eq 0 ]]; then
  echo "No .mp4 files found in: $input_dir"
  exit 0
fi

echo "Input:    $input_dir"
echo "Output:   $output_dir"
echo "Format:   mp3"
echo

converted_count=0
skipped_count=0

for input_file in "${mp4_files[@]}"; do
  filename="$(basename "$input_file")"
  stem="${filename%.*}"
  output_file="$output_dir/${stem}.mp3"
  has_audio="$(
    ffprobe -v error -select_streams a:0 -show_entries stream=index \
      -of csv=p=0 "$input_file" | head -n 1
  )"

  if [[ -z "$has_audio" ]]; then
    echo "Skipping without audio: $filename"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  echo "Extracting: $filename"
  ffmpeg -hide_banner -y \
    -i "$input_file" \
    -map 0:a:0 \
    -vn \
    -ar "$audio_rate" \
    -c:a libmp3lame \
    -b:a "$audio_bitrate" \
    "$output_file"
  converted_count=$((converted_count + 1))
done

echo
echo "Done. Extracted $converted_count file(s), skipped $skipped_count without audio."
