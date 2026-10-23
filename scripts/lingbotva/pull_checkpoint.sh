#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-PAI-4051}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-/mnt/workspace/shenyibo/nmx_vla_dropvideo/results/giftbox_8node8gpu1ga_ws3-16_1758ep_12000pre_dropvideo}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/noematrix/lingbotva_ckpts/0626ckpt}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/home/noematrix/lingbot-va-pretrained-ckpt}"
NUM_STEPS="${NUM_STEPS:--1}"

usage() {
    cat <<'EOF'
Usage: scripts/lingbotva/pull_checkpoint.sh [options]

Download a LingBot-VA transformer checkpoint from server and rewrite
its local config paths.

Options:
  --remote-host HOST       SSH host alias.
  --remote-run-dir PATH    Remote training run directory.
  --local-root PATH        Local checkpoint root. Default: /home/noematrix/lingbotva_ckpts
  --local-run-dir PATH     Local training run directory. Default: local-root/basename(remote-run-dir)
  --model-path PATH        Local pretrained model path.
  --num-steps NUM_STEPS    Checkpoint step to download. Default: -1 (latest checkpoint).
  -h, --help               Show this help.

Environment variables with the same names also work:
  REMOTE_HOST, REMOTE_RUN_DIR, LOCAL_ROOT, LOCAL_RUN_DIR, PRETRAINED_MODEL_PATH,
  NUM_STEPS
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote-host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        --remote-run-dir)
            REMOTE_RUN_DIR="$2"
            shift 2
            ;;
        --local-root)
            LOCAL_ROOT="$2"
            shift 2
            ;;
        --local-run-dir)
            LOCAL_RUN_DIR="$2"
            shift 2
            ;;
        --model-path)
            PRETRAINED_MODEL_PATH="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$NUM_STEPS" != "-1" && ! "$NUM_STEPS" =~ ^[0-9]+$ ]]; then
    echo "NUM_STEPS must be -1 or a non-negative integer: $NUM_STEPS" >&2
    exit 2
fi

shell_quote() {
    printf "%q" "$1"
}

rsync_remote_path() {
    printf "%s:%q" "$REMOTE_HOST" "$1"
}

remote_file_exists() {
    ssh "$REMOTE_HOST" "test -f $(shell_quote "$1")"
}

download_file_to_checkpoint_root() {
    local remote_file="$1"
    local local_dir="$2"

    if remote_file_exists "$remote_file"; then
        rsync -a --human-readable "$(rsync_remote_path "$remote_file")" "$local_dir/"
        return 0
    fi

    return 1
}

download_file_with_fallbacks() {
    local file_name="$1"
    local local_dir="$2"
    shift 2

    local remote_dir
    for remote_dir in "$@"; do
        if download_file_to_checkpoint_root "${remote_dir%/}/$file_name" "$local_dir"; then
            return 0
        fi
    done

    return 1
}

rewrite_config_file() {
    local config_file="$1"
    [[ -f "$config_file" ]] || return 0

    PRETRAINED_MODEL_PATH="$PRETRAINED_MODEL_PATH" \
    NORM_STATS_MANIFEST_PATH="$NORM_STATS_MANIFEST_PATH" \
    LOCAL_RUN_DIR="$LOCAL_RUN_DIR" \
    perl -0pi -e '
        use strict;
        use warnings;

        my $model = $ENV{"PRETRAINED_MODEL_PATH"};
        my $norm_manifest = $ENV{"NORM_STATS_MANIFEST_PATH"};
        my $out = $ENV{"LOCAL_RUN_DIR"};

        s/("wan22_pretrained_model_name_or_path"\s*:\s*)(?:"(?:[^"\\]|\\.)*"|null)/${1}"$model"/g;
        s/("model_path"\s*:\s*)(?:"(?:[^"\\]|\\.)*"|null)/${1}"$model"/g;
        s/("norm_stats_manifest_path"\s*:\s*)(?:"(?:[^"\\]|\\.)*"|null)/${1}"$norm_manifest"/g;
        s/("output_dir"\s*:\s*)(?:"(?:[^"\\]|\\.)*"|null)/${1}"$out"/g;

        s/^([ \t]*model_path:[ \t]*).*$/${1}$model/gm;
        s/^([ \t]*wan22_pretrained_model_name_or_path:[ \t]*).*$/${1}$model/gm;
        s/^([ \t]*norm_stats_manifest_path:[ \t]*).*$/${1}$norm_manifest/gm;
        s/^([ \t]*output_dir:[ \t]*).*$/${1}$out/gm;
    ' "$config_file"
}

REMOTE_RUN_DIR="${REMOTE_RUN_DIR%/}"
REMOTE_CHECKPOINT_ROOT="$REMOTE_RUN_DIR/checkpoints"
RUN_NAME="$(basename "$REMOTE_RUN_DIR")"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-${LOCAL_ROOT%/}/$RUN_NAME}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR%/}"

if [[ "$NUM_STEPS" == "-1" ]]; then
    echo "Finding latest checkpoint under ${REMOTE_HOST}:${REMOTE_CHECKPOINT_ROOT}"
else
    echo "Finding checkpoint_step_${NUM_STEPS} under ${REMOTE_HOST}:${REMOTE_CHECKPOINT_ROOT}"
fi

REMOTE_CHECKPOINT="$(
    ssh "$REMOTE_HOST" "REMOTE_CHECKPOINT_ROOT=$(shell_quote "$REMOTE_CHECKPOINT_ROOT") NUM_STEPS=$(shell_quote "$NUM_STEPS") bash -s" <<'REMOTE_FIND'
set -euo pipefail
shopt -s nullglob

if [[ "$NUM_STEPS" != "-1" ]]; then
    selected="$REMOTE_CHECKPOINT_ROOT/checkpoint_step_$NUM_STEPS"

    if [[ ! -d "$selected" ]]; then
        echo "No checkpoint_step_$NUM_STEPS directory found under $REMOTE_CHECKPOINT_ROOT" >&2
        exit 1
    fi

    if [[ ! -d "$selected/transformer" ]]; then
        echo "Requested checkpoint has no transformer directory: $selected" >&2
        exit 1
    fi

    printf "%s\n" "$selected"
    exit 0
fi

latest=""
latest_step=-1

for checkpoint_dir in "$REMOTE_CHECKPOINT_ROOT"/checkpoint_step_*; do
    [[ -d "$checkpoint_dir" ]] || continue

    checkpoint_name="${checkpoint_dir##*/}"
    step="${checkpoint_name#checkpoint_step_}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue

    if (( step > latest_step )); then
        latest_step="$step"
        latest="$checkpoint_dir"
    fi
done

if [[ -z "$latest" ]]; then
    echo "No checkpoint_step_* directory found under $REMOTE_CHECKPOINT_ROOT" >&2
    exit 1
fi

if [[ ! -d "$latest/transformer" ]]; then
    echo "Latest checkpoint has no transformer directory: $latest" >&2
    exit 1
fi

printf "%s\n" "$latest"
REMOTE_FIND
)"

CHECKPOINT_NAME="$(basename "$REMOTE_CHECKPOINT")"
LOCAL_CHECKPOINT_DIR="$LOCAL_RUN_DIR/$CHECKPOINT_NAME"
NORM_STATS_MANIFEST_PATH="$LOCAL_RUN_DIR/lingbot_action_norm_stats_manifest.json"

mkdir -p "$LOCAL_CHECKPOINT_DIR" "$LOCAL_RUN_DIR"

if [[ "$NUM_STEPS" == "-1" ]]; then
    echo "Latest checkpoint: $CHECKPOINT_NAME"
else
    echo "Requested checkpoint: $CHECKPOINT_NAME"
fi
echo "Local run dir: $LOCAL_RUN_DIR"
echo "Local checkpoint: $LOCAL_CHECKPOINT_DIR"

echo "Syncing transformer/"
rsync -a --delete --partial --human-readable --info=progress2 \
    "$(rsync_remote_path "$REMOTE_CHECKPOINT/transformer/")" \
    "$LOCAL_CHECKPOINT_DIR/transformer/"

echo "Syncing run-level files"
rsync -a --human-readable --exclude='*/' \
    "$(rsync_remote_path "$REMOTE_RUN_DIR/")" \
    "$LOCAL_RUN_DIR/"

echo "Checking required config files"
required_config_files=(
    "config.yaml"
    "lingbot_runtime_config.json"
    "resolved_config.yaml"
)

missing_files=()
for file_name in "${required_config_files[@]}"; do
    if ! download_file_with_fallbacks \
        "$file_name" \
        "$LOCAL_RUN_DIR" \
        "$REMOTE_RUN_DIR" \
        "$REMOTE_CHECKPOINT" \
        "$REMOTE_CHECKPOINT/transformer"
    then
        missing_files+=("$file_name")
    fi
done

if ! download_file_with_fallbacks \
    "lingbot_action_norm_stats_manifest.json" \
    "$LOCAL_RUN_DIR" \
    "$REMOTE_RUN_DIR" \
    "$REMOTE_CHECKPOINT" \
    "$REMOTE_CHECKPOINT/transformer"
then
    missing_files+=("lingbot_action_norm_stats_manifest.json")
fi

if (( ${#missing_files[@]} > 0 )); then
    printf "Missing required file(s) in remote checkpoint: %s\n" "${missing_files[*]}" >&2
    exit 1
fi

echo "Rewriting config paths"
for config_file in \
    "$LOCAL_RUN_DIR/config.yaml" \
    "$LOCAL_RUN_DIR/lingbot_runtime_config.json" \
    "$LOCAL_RUN_DIR/resolved_config.yaml"
do
    rewrite_config_file "$config_file"
done

if grep -E --quiet \
    '("wan22_pretrained_model_name_or_path"[[:space:]]*:[[:space:]]*"/mnt/workspace/)|(^[[:space:]]*model_path:[[:space:]]*/mnt/workspace/)' \
    "$LOCAL_RUN_DIR/config.yaml" \
    "$LOCAL_RUN_DIR/lingbot_runtime_config.json" \
    "$LOCAL_RUN_DIR/resolved_config.yaml"
then
    echo "Warning: remote base model path still appears in one or more config files." >&2
fi

echo "Done."
echo "Run dir: $LOCAL_RUN_DIR"
echo "Checkpoint: $LOCAL_CHECKPOINT_DIR"
echo "Transformer: $LOCAL_CHECKPOINT_DIR/transformer"
echo "Norm stats manifest: $NORM_STATS_MANIFEST_PATH"
