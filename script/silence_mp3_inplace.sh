#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./silence_mp3_inplace.sh TARGET_DIR [options]

Recursively replace every .mp3 under TARGET_DIR with a silent MP3 in place.
The script decodes each source MP3, applies volume=0, and re-encodes it, so it
preserves the actual audio duration without relying on readable duration tags.
By default it keeps the source sample rate, channel count, and bitrate when
ffprobe can read them.

Options:
  --bitrate BITRATE       Override output bitrate, for example 128k or 192k.
  --sample-rate RATE      Override output sample rate, for example 44100.
  --channels N            Override output channel count, usually 1 or 2.
  --dry-run               Print what would be processed without overwriting.
  -h, --help              Show this help.

Examples:
  ./silence_mp3_inplace.sh ./train_data/mv
  ./silence_mp3_inplace.sh ./train_data/mv --bitrate 192k --sample-rate 44100 --channels 2
USAGE
}

die() {
  echo "Error: $*" >&2
  exit 1
}

probe_stream_value() {
  local input_file="$1"
  local key="$2"

  ffprobe -v error \
    -select_streams a:0 \
    -show_entries "stream=${key}" \
    -of default=noprint_wrappers=1:nokey=1 \
    "$input_file" 2>/dev/null | head -n 1 || true
}

probe_format_value() {
  local input_file="$1"
  local key="$2"

  ffprobe -v error \
    -show_entries "format=${key}" \
    -of default=noprint_wrappers=1:nokey=1 \
    "$input_file" 2>/dev/null | head -n 1 || true
}

is_positive_number() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]] && [[ "$1" != "0" ]] && [[ "$1" != "0.0" ]]
}

is_positive_integer() {
  [[ "$1" =~ ^[0-9]+$ ]] && (( "$1" > 0 ))
}

bitrate_from_bits_per_second() {
  local bits_per_second="$1"

  if is_positive_integer "$bits_per_second"; then
    echo "$(( (bits_per_second + 999) / 1000 ))k"
  else
    echo ""
  fi
}

target_dir=""
bitrate_override=""
sample_rate_override=""
channels_override=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --bitrate)
      [[ $# -ge 2 ]] || die "--bitrate requires a value."
      bitrate_override="$2"
      shift 2
      ;;
    --sample-rate)
      [[ $# -ge 2 ]] || die "--sample-rate requires a value."
      sample_rate_override="$2"
      shift 2
      ;;
    --channels)
      [[ $# -ge 2 ]] || die "--channels requires a value."
      channels_override="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      [[ -z "$target_dir" ]] || die "only one TARGET_DIR can be specified."
      target_dir="$1"
      shift
      ;;
  esac
done

[[ -n "$target_dir" ]] || {
  usage >&2
  exit 1
}

[[ -d "$target_dir" ]] || die "target directory does not exist: $target_dir"

target_dir="$(cd "$target_dir" && pwd -P)"

command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg is not installed or not in PATH."
command -v ffprobe >/dev/null 2>&1 || die "ffprobe is not installed or not in PATH."
command -v find >/dev/null 2>&1 || die "find is not installed or not in PATH."

if [[ -n "$sample_rate_override" ]] && ! is_positive_integer "$sample_rate_override"; then
  die "--sample-rate must be a positive integer."
fi

if [[ -n "$channels_override" ]] && ! is_positive_integer "$channels_override"; then
  die "--channels must be a positive integer."
fi

mapfile -d "" mp3_files < <(find "$target_dir" -type f -iname "*.mp3" -print0 | sort -z)

if [[ ${#mp3_files[@]} -eq 0 ]]; then
  echo "No .mp3 files found in: $target_dir"
  exit 0
fi

echo "Target:  $target_dir"
echo "Mode:    $([[ "$dry_run" -eq 1 ]] && echo "dry-run" || echo "overwrite in place")"
echo "Files:   ${#mp3_files[@]}"
echo

processed_count=0
failed_count=0

for input_file in "${mp3_files[@]}"; do
  duration="$(probe_format_value "$input_file" duration)"
  if ! is_positive_number "$duration"; then
    duration="unknown"
  fi

  sample_rate="${sample_rate_override:-$(probe_stream_value "$input_file" sample_rate)}"
  if ! is_positive_integer "$sample_rate"; then
    sample_rate=44100
  fi

  channels="${channels_override:-$(probe_stream_value "$input_file" channels)}"
  if ! is_positive_integer "$channels"; then
    channels=2
  fi
  if (( channels > 2 )); then
    echo "Warning: MP3 supports mono/stereo best; using 2 channels for: $input_file" >&2
    channels=2
  fi

  bitrate="$bitrate_override"
  if [[ -z "$bitrate" ]]; then
    bitrate="$(bitrate_from_bits_per_second "$(probe_stream_value "$input_file" bit_rate)")"
  fi
  if [[ -z "$bitrate" ]]; then
    bitrate="$(bitrate_from_bits_per_second "$(probe_format_value "$input_file" bit_rate)")"
  fi
  if [[ -z "$bitrate" ]]; then
    bitrate="192k"
  fi

  echo "Silencing: $input_file"
  echo "  duration=${duration}s sample_rate=${sample_rate} channels=${channels} bitrate=${bitrate}"

  if [[ "$dry_run" -eq 1 ]]; then
    processed_count=$((processed_count + 1))
    continue
  fi

  input_dir="${input_file%/*}"
  input_name="${input_file##*/}"
  tmp_file="$(mktemp "${input_dir}/.${input_name%.mp3}.silent.XXXXXX.mp3")"

  if ffmpeg -hide_banner -loglevel error -y \
    -i "$input_file" \
    -map 0:a:0 \
    -vn \
    -af "volume=0" \
    -ar "$sample_rate" \
    -ac "$channels" \
    -c:a libmp3lame \
    -b:a "$bitrate" \
    -f mp3 \
    "$tmp_file"; then
    chmod --reference="$input_file" "$tmp_file" 2>/dev/null || true
    mv -f "$tmp_file" "$input_file"
    processed_count=$((processed_count + 1))
  else
    echo "Failed: $input_file" >&2
    rm -f "$tmp_file"
    failed_count=$((failed_count + 1))
  fi
done

echo
echo "Done. Processed $processed_count file(s), failed $failed_count."

if [[ "$failed_count" -gt 0 ]]; then
  exit 1
fi
