#!/usr/bin/env bash
set -uo pipefail

# Copy one local file or directory to the same absolute path on other nodes.
# The default node list matches Wan2.2-S2V-14B_cache_dist_run_8.sh. Override
# NODES to use the four-node layout or another cluster topology.

usage() {
  cat <<'EOF'
Usage:
  bash ./sync_folder_to_dist_nodes.sh [options] PATH [HOST ...]

Synchronize a file or the contents of a directory to the same absolute path
on every target host. The current host is skipped automatically.

Options:
  --delete                  For directories, delete remote-only files.
  --before-command COMMAND  Run COMMAND on each target before synchronization.
  --after-command COMMAND   Run COMMAND after that target synchronizes.
  --dry-run                 Preview rsync and commands without changing hosts.
  -h, --help                Show this help message.

Remote commands are interpreted by the remote user's shell. Relative paths
start in that user's default login directory, so absolute paths are preferred.

Host selection (highest priority first):
  1. HOST arguments after PATH
  2. The space-separated NODES environment variable
  3. node1 through node8

Environment variables:
  NODES          Space-separated host list.
  CURRENT_NODE   Hostname/alias to skip in addition to the detected hostname.
  SSH_USER       Optional remote SSH user.
  SSH_PORT       Optional remote SSH port.
  SSH_OPTIONS    Extra space-separated ssh options.
  RSYNC_OPTIONS  Extra space-separated rsync options.

Examples:
  # Sync one model file using the default node1 ... node8 list.
  # When this runs on node1, the script automatically targets node2 ... node8.
  bash ./sync_folder_to_dist_nodes.sh /data-training/wan_s2v_4steps.safetensors

  # Sync all contents of a directory to the same directory on the other nodes.
  # Existing files with a different size or modification time are overwritten.
  bash ./sync_folder_to_dist_nodes.sh /data-training/DiffSynth-Studio/models/train/cache

  # Use the four-node layout from Wan2.2-S2V-14B_cache_dist_run_4.sh.
  NODES="node1 node2 node3 node4" \
    bash ./sync_folder_to_dist_nodes.sh /data-training/models

  # Sync only to explicit hosts. HOST arguments override the NODES variable.
  bash ./sync_folder_to_dist_nodes.sh /data-training/models node2 node3

  # Preview an exact directory mirror. No files, commands, or deletions occur.
  bash ./sync_folder_to_dist_nodes.sh --dry-run --delete /data-training/models

  # Make an exact directory mirror. WARNING: remote-only files are deleted.
  bash ./sync_folder_to_dist_nodes.sh --delete /data-training/models node2 node3

  # Back up an existing remote file before replacing it. The existence check
  # prevents a missing old file from causing the before command to fail.
  bash ./sync_folder_to_dist_nodes.sh \
    --before-command \
      'if [ -e /data-training/model.bin ]; then mv /data-training/model.bin /data-training/model.bin.bak; fi' \
    /data-training/model.bin

  # Run a verification command after each node synchronizes successfully.
  bash ./sync_folder_to_dist_nodes.sh \
    --after-command 'sha256sum /data-training/model.bin' \
    /data-training/model.bin

  # Keep every existing remote file and transfer only files that are missing.
  RSYNC_OPTIONS="--ignore-existing" \
    bash ./sync_folder_to_dist_nodes.sh /data-training/models

  # Do not replace a remote file when its modification time is newer.
  RSYNC_OPTIONS="--update" \
    bash ./sync_folder_to_dist_nodes.sh /data-training/models

  # Compare file contents by checksum instead of only size and modification time.
  RSYNC_OPTIONS="--checksum" \
    bash ./sync_folder_to_dist_nodes.sh /data-training/models

  # Connect with a custom SSH user, port, and non-interactive SSH option.
  SSH_USER=training SSH_PORT=2222 SSH_OPTIONS="-o BatchMode=yes" \
    bash ./sync_folder_to_dist_nodes.sh /data-training/models
EOF
}

shell_quote() {
  local value="$1"
  value=${value//\'/\'\\\'\'}
  printf "'%s'" "$value"
}

run_remote_command() {
  local node="$1" remote="$2" stage="$3" remote_command="$4"
  if [ -z "$remote_command" ]; then
    return 0
  fi
  if [ "$dry_run" = true ]; then
    echo "[$node] Would run $stage command: $remote_command"
    return 0
  fi

  echo "[$node] Running $stage command: $remote_command"
  ssh ${ssh_args[@]+"${ssh_args[@]}"} "$remote" "$remote_command"
}

delete_remote=false
dry_run=false
before_command=""
after_command=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --delete)
      delete_remote=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    --before-command)
      if [ "$#" -lt 2 ]; then
        echo "--before-command requires COMMAND." >&2
        exit 2
      fi
      before_command="$2"
      shift 2
      ;;
    --after-command)
      if [ "$#" -lt 2 ]; then
        echo "--after-command requires COMMAND." >&2
        exit 2
      fi
      after_command="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [ "$#" -eq 0 ]; then
  echo "PATH is required." >&2
  usage >&2
  exit 2
fi

input_path="$1"
shift

if [ ! -e "$input_path" ] || [ ! -r "$input_path" ]; then
  echo "Path does not exist or is not readable: $input_path" >&2
  exit 1
fi

# Resolve a relative input once so every remote receives the same absolute path.
if [ -d "$input_path" ]; then
  sync_type=directory
  if ! sync_path="$(cd -- "$input_path" && pwd -P)"; then
    echo "Failed to resolve directory: $input_path" >&2
    exit 1
  fi
  if [ "$sync_path" = "/" ]; then
    echo "Refusing to synchronize the filesystem root directory." >&2
    exit 1
  fi
  remote_parent="$sync_path"
  rsync_source="$sync_path/"
  rsync_destination="$sync_path/"
elif [ -f "$input_path" ]; then
  sync_type=file
  input_parent="$(dirname -- "$input_path")"
  input_name="$(basename -- "$input_path")"
  if ! resolved_parent="$(cd -- "$input_parent" && pwd -P)"; then
    echo "Failed to resolve file path: $input_path" >&2
    exit 1
  fi
  if [ "$resolved_parent" = "/" ]; then
    sync_path="/$input_name"
  else
    sync_path="$resolved_parent/$input_name"
  fi
  remote_parent="$resolved_parent"
  rsync_source="$sync_path"
  rsync_destination="$sync_path"
else
  echo "Only regular files and directories are supported: $input_path" >&2
  exit 1
fi

if [ "$#" -gt 0 ]; then
  node_list=("$@")
else
  nodes="${NODES:-node1 node2 node3 node4 node5 node6 node7 node8}"
  read -r -a node_list <<< "$nodes"
fi

if [ "${#node_list[@]}" -eq 0 ]; then
  echo "No target hosts were provided." >&2
  exit 1
fi

for command_name in ssh rsync; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done

ssh_args=()
if [ -n "${SSH_PORT:-}" ]; then
  if ! [[ "$SSH_PORT" =~ ^[0-9]+$ ]] || [ "$SSH_PORT" -le 0 ] || [ "$SSH_PORT" -gt 65535 ]; then
    echo "SSH_PORT=$SSH_PORT must be an integer from 1 to 65535." >&2
    exit 1
  fi
  ssh_args+=(-p "$SSH_PORT")
fi
if [ -n "${SSH_OPTIONS:-}" ]; then
  read -r -a extra_ssh_args <<< "$SSH_OPTIONS"
  ssh_args+=("${extra_ssh_args[@]}")
fi

rsync_args=(-a --partial --human-readable --progress)
# Modern rsync can send path arguments through its protocol instead of asking
# the remote shell to parse them. This also keeps paths containing spaces safe.
rsync_help="$(rsync --help 2>&1 || true)"
if [[ "$rsync_help" == *"--secluded-args"* ]] || [[ "$rsync_help" == *"--protect-args"* ]]; then
  rsync_args+=(-s)
fi
if [ "$delete_remote" = true ] && [ "$sync_type" = directory ]; then
  rsync_args+=(--delete-after)
elif [ "$delete_remote" = true ]; then
  echo "Note: --delete has no effect when synchronizing a single file." >&2
fi
if [ "$dry_run" = true ]; then
  rsync_args+=(--dry-run)
fi
if [ -n "${RSYNC_OPTIONS:-}" ]; then
  read -r -a extra_rsync_args <<< "$RSYNC_OPTIONS"
  rsync_args+=("${extra_rsync_args[@]}")
fi

rsync_ssh=(ssh)
if [ -n "${SSH_PORT:-}" ]; then
  rsync_ssh+=(-p "$SSH_PORT")
fi
if [ -n "${SSH_OPTIONS:-}" ]; then
  rsync_ssh+=("${extra_ssh_args[@]}")
fi
printf -v rsync_ssh_command '%q ' "${rsync_ssh[@]}"
rsync_args+=(-e "${rsync_ssh_command% }")

short_host="$(hostname -s)"
full_host="$(hostname -f 2>/dev/null || hostname)"
current_node="${CURRENT_NODE:-}"

targets=()
for i in "${!node_list[@]}"; do
  node="${node_list[$i]}"
  if [ -z "$node" ]; then
    continue
  fi
  for ((j = 0; j < i; j++)); do
    if [ "$node" = "${node_list[$j]}" ]; then
      echo "Duplicate host in node list: $node" >&2
      exit 1
    fi
  done

  compare_node="${node#*@}"
  compare_node="${compare_node#[}"
  compare_node="${compare_node%]}"
  if [ "$compare_node" = "$short_host" ] || [ "$compare_node" = "$full_host" ] || \
     { [ -n "$current_node" ] && [ "$compare_node" = "$current_node" ]; }; then
    echo "Skipping current host: $node"
    continue
  fi
  targets+=("$node")
done

if [ "${#targets[@]}" -eq 0 ]; then
  echo "No remote hosts remain after skipping the current host."
  exit 0
fi

echo "Source and destination path: $sync_path"
echo "Path type: $sync_type"
echo "Target hosts: ${targets[*]}"
echo "Delete remote-only files: $delete_remote"
if [ -n "$before_command" ]; then
  echo "Before command: $before_command"
fi
if [ -n "$after_command" ]; then
  echo "After command: $after_command"
fi
echo "Dry run: $dry_run"

failed_nodes=()
quoted_remote_parent="$(shell_quote "$remote_parent")"

for node in "${targets[@]}"; do
  remote="$node"
  if [ -n "${SSH_USER:-}" ] && [[ "$remote" != *@* ]]; then
    remote="${SSH_USER}@${remote}"
  fi

  echo
  echo "[$node] Synchronizing $sync_path"

  if ! run_remote_command "$node" "$remote" before "$before_command"; then
    echo "[$node] Before command failed; skipping synchronization." >&2
    failed_nodes+=("$node")
    continue
  fi

  if [ "$dry_run" = false ]; then
    if ! ssh ${ssh_args[@]+"${ssh_args[@]}"} "$remote" "mkdir -p -- $quoted_remote_parent"; then
      echo "[$node] Failed to create destination parent directory." >&2
      failed_nodes+=("$node")
      continue
    fi
  fi

  if ! rsync "${rsync_args[@]}" "$rsync_source" "${remote}:${rsync_destination}"; then
    echo "[$node] Synchronization failed." >&2
    failed_nodes+=("$node")
    continue
  fi

  if ! run_remote_command "$node" "$remote" after "$after_command"; then
    echo "[$node] After command failed." >&2
    failed_nodes+=("$node")
    continue
  fi
  echo "[$node] Synchronization completed."
done

if [ "${#failed_nodes[@]}" -gt 0 ]; then
  echo >&2
  echo "Synchronization failed on: ${failed_nodes[*]}" >&2
  exit 1
fi

echo
echo "Synchronization completed on all target hosts."
