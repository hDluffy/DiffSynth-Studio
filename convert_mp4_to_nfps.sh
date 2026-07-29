#!/usr/bin/env bash
set -euo pipefail

DEFAULT_FPS="16"
FPS="${DEFAULT_FPS}"
INPUT_PATH=""
OUTPUT_PATH=""
CRF="${CRF:-16}"
PRESET="${PRESET:-slow}"
AUDIO_BITRATE="${AUDIO_BITRATE:-192k}"
OVERWRITE=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  ./convert_mp4_to_nfps.sh -i INPUT -o OUTPUT [-r FPS]
  ./convert_mp4_to_nfps.sh INPUT OUTPUT [FPS]

Options:
  -i, --input PATH          Input directory or a single video file.
  -o, --output PATH         Output directory, or output file for single-file input.
  -r, --fps FPS             Target frame rate. Default: 16.
      --crf VALUE           libx264 CRF. Lower means better quality/larger file. Default: 16.
      --preset VALUE        libx264 preset. Default: slow.
      --audio-bitrate RATE  AAC audio bitrate. Default: 192k.
  -y, --overwrite           Overwrite existing outputs.
      --dry-run             Print planned conversions without running ffmpeg.
  -h, --help                Show this help.

Examples:
  ./convert_mp4_to_nfps.sh -i /data/videos -o /data/videos_16fps
  ./convert_mp4_to_nfps.sh -i /data/in.mp4 -o /data/out.mp4 -r 16
USAGE
}

die() {
  echo "Error: $*" >&2
  exit 1
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is required but was not found in PATH."
}

is_video_file() {
  local path="${1,,}"
  case "${path}" in
    *.mp4|*.mov|*.mkv|*.avi|*.webm|*.m4v|*.flv|*.wmv|*.mpg|*.mpeg|*.3gp|*.ts|*.mts|*.m2ts)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

looks_like_output_file() {
  is_video_file "$1"
}

fps_label() {
  local label="$1"
  label="${label//./p}"
  echo "${label}"
}

probe_duration() {
  ffprobe -v error \
    -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    "$1" 2>/dev/null || true
}

is_positive_number() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v n="$1" 'BEGIN { exit !(n > 0) }'
}

same_path() {
  [[ "$(realpath -m "$1")" == "$(realpath -m "$2")" ]]
}

duration_tolerance() {
  awk -v fps="$FPS" 'BEGIN {
    tolerance = (1 / fps) + 0.02
    if (tolerance < 0.08) tolerance = 0.08
    printf "%.6f", tolerance
  }'
}

check_duration() {
  local src="$1"
  local dst="$2"
  local src_duration="$3"

  [[ -n "${src_duration}" && "${src_duration}" != "N/A" ]] || return 0
  is_positive_number "${src_duration}" || return 0

  local dst_duration delta tolerance ok
  dst_duration="$(probe_duration "${dst}")"
  [[ -n "${dst_duration}" && "${dst_duration}" != "N/A" ]] || {
    echo "WARN: could not probe output duration: ${dst}" >&2
    return 0
  }

  delta="$(awk -v a="${src_duration}" -v b="${dst_duration}" 'BEGIN {
    d = a - b
    if (d < 0) d = -d
    printf "%.6f", d
  }')"
  tolerance="$(duration_tolerance)"
  ok="$(awk -v d="${delta}" -v t="${tolerance}" 'BEGIN { print (d <= t) ? 1 : 0 }')"

  if [[ "${ok}" != "1" ]]; then
    echo "WARN: duration changed by ${delta}s: ${src} -> ${dst}" >&2
  fi
}

output_for_directory_input() {
  local src="$1"
  local input_root="$2"
  local output_root="$3"
  local rel rel_dir base stem out_dir out_path label

  rel="${src#"${input_root%/}"/}"
  rel_dir="$(dirname "${rel}")"
  base="$(basename "${rel}")"
  stem="${base%.*}"

  if [[ "${rel_dir}" == "." ]]; then
    out_dir="${output_root%/}"
  else
    out_dir="${output_root%/}/${rel_dir}"
  fi

  out_path="${out_dir}/${stem}.mp4"
  if same_path "${src}" "${out_path}"; then
    label="$(fps_label "${FPS}")"
    out_path="${out_dir}/${stem}_${label}fps.mp4"
  fi

  echo "${out_path}"
}

output_for_file_input() {
  local src="$1"
  local output="$2"
  local base stem out_path label

  base="$(basename "${src}")"
  stem="${base%.*}"

  if [[ -d "${output}" || "${output}" == */ ]] || ! looks_like_output_file "${output}"; then
    out_path="${output%/}/${stem}.mp4"
  else
    out_path="${output}"
  fi

  if same_path "${src}" "${out_path}"; then
    label="$(fps_label "${FPS}")"
    out_path="$(dirname "${out_path}")/${stem}_${label}fps.mp4"
  fi

  echo "${out_path}"
}

convert_one() {
  local src="$1"
  local dst="$2"
  local dst_dir tmp src_duration filter

  dst_dir="$(dirname "${dst}")"
  mkdir -p "${dst_dir}"

  if [[ -e "${dst}" && "${OVERWRITE}" -ne 1 ]]; then
    echo "SKIP: ${dst} already exists"
    return 0
  fi

  src_duration="$(probe_duration "${src}")"
  filter="fps=fps=${FPS}:round=near"
  if [[ -n "${src_duration}" && "${src_duration}" != "N/A" ]] && is_positive_number "${src_duration}"; then
    filter="${filter},tpad=stop_mode=clone:stop_duration=1"
  fi

  echo "CONVERT: ${src} -> ${dst} (${FPS} fps)"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  tmp="$(mktemp "${dst_dir}/.$(basename "${dst}").tmp.XXXXXX.mp4")"

  local cmd=(
    ffmpeg
    -hide_banner
    -loglevel error
    -stats
    -y
    -i "${src}"
    -map 0:v:0
    -map "0:a?"
    -vf "${filter}"
    -c:v libx264
    -preset "${PRESET}"
    -crf "${CRF}"
    -pix_fmt yuv420p
    -c:a aac
    -b:a "${AUDIO_BITRATE}"
    -movflags +faststart
    -map_metadata 0
    -max_muxing_queue_size 1024
  )

  if [[ -n "${src_duration}" && "${src_duration}" != "N/A" ]] && is_positive_number "${src_duration}"; then
    cmd+=(-t "${src_duration}")
  fi

  cmd+=("${tmp}")

  if ! "${cmd[@]}"; then
    rm -f "${tmp}"
    echo "FAIL: ${src}" >&2
    return 1
  fi

  mv -f "${tmp}" "${dst}"
  check_duration "${src}" "${dst}" "${src_duration}"
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      [[ $# -ge 2 ]] || die "$1 requires a path."
      INPUT_PATH="$2"
      shift 2
      ;;
    -o|--output)
      [[ $# -ge 2 ]] || die "$1 requires a path."
      OUTPUT_PATH="$2"
      shift 2
      ;;
    -r|--fps|--frame-rate)
      [[ $# -ge 2 ]] || die "$1 requires a frame rate."
      FPS="$2"
      shift 2
      ;;
    --crf)
      [[ $# -ge 2 ]] || die "$1 requires a value."
      CRF="$2"
      shift 2
      ;;
    --preset)
      [[ $# -ge 2 ]] || die "$1 requires a value."
      PRESET="$2"
      shift 2
      ;;
    --audio-bitrate)
      [[ $# -ge 2 ]] || die "$1 requires a value."
      AUDIO_BITRATE="$2"
      shift 2
      ;;
    -y|--overwrite)
      OVERWRITE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        POSITIONAL+=("$1")
        shift
      done
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${INPUT_PATH}" && ${#POSITIONAL[@]} -ge 1 ]]; then
  INPUT_PATH="${POSITIONAL[0]}"
fi
if [[ -z "${OUTPUT_PATH}" && ${#POSITIONAL[@]} -ge 2 ]]; then
  OUTPUT_PATH="${POSITIONAL[1]}"
fi
if [[ "${FPS}" == "${DEFAULT_FPS}" && ${#POSITIONAL[@]} -ge 3 ]]; then
  FPS="${POSITIONAL[2]}"
fi
[[ ${#POSITIONAL[@]} -le 3 ]] || die "Too many positional arguments."

[[ -n "${INPUT_PATH}" ]] || {
  usage
  die "input path is required."
}
[[ -n "${OUTPUT_PATH}" ]] || {
  usage
  die "output path is required."
}
[[ -e "${INPUT_PATH}" ]] || die "input path does not exist: ${INPUT_PATH}"
is_positive_number "${FPS}" || die "FPS must be a positive number: ${FPS}"
is_positive_number "${CRF}" || die "CRF must be a positive number: ${CRF}"

need_command ffmpeg
need_command ffprobe
need_command realpath

if [[ -d "${INPUT_PATH}" ]]; then
  if looks_like_output_file "${OUTPUT_PATH}"; then
    die "directory input requires output to be a directory, not a video file path: ${OUTPUT_PATH}"
  fi

  mapfile -d '' VIDEO_FILES < <(
    find "${INPUT_PATH}" -type f \( \
      -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.avi' -o \
      -iname '*.webm' -o -iname '*.m4v' -o -iname '*.flv' -o -iname '*.wmv' -o \
      -iname '*.mpg' -o -iname '*.mpeg' -o -iname '*.3gp' -o -iname '*.ts' -o \
      -iname '*.mts' -o -iname '*.m2ts' \
    \) -print0 | sort -z
  )

  [[ ${#VIDEO_FILES[@]} -gt 0 ]] || die "no video files found under: ${INPUT_PATH}"
  mkdir -p "${OUTPUT_PATH}"

  failed=0
  for src in "${VIDEO_FILES[@]}"; do
    dst="$(output_for_directory_input "${src}" "${INPUT_PATH}" "${OUTPUT_PATH}")"
    if ! convert_one "${src}" "${dst}"; then
      failed=$((failed + 1))
    fi
  done

  [[ "${failed}" -eq 0 ]] || die "${failed} file(s) failed."
else
  is_video_file "${INPUT_PATH}" || die "input file does not look like a supported video: ${INPUT_PATH}"
  dst="$(output_for_file_input "${INPUT_PATH}" "${OUTPUT_PATH}")"
  convert_one "${INPUT_PATH}" "${dst}"
fi
