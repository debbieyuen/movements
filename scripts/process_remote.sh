#!/usr/bin/env bash
# Process a capture on the Lambda GPU box from your Mac:
#
#   scripts/process_remote.sh <video-file> --subject-height 1.57 [more args]
#
# Config via env vars (or edit the defaults below):
#   MOCAP_GPU_HOST   ssh host (e.g. ubuntu@203.0.113.7 or an ~/.ssh/config alias)
#   MOCAP_GPU_DIR    repo checkout on the box   (default: ~/movements)
#   MOCAP_GPU_ENV    conda env with the mocap package (default: mocap)
#
# Flow: rsync the video up -> run `python -m mocap.process` remotely ->
# rsync the finished dataset/<clip_id>/ back next to this repo.

set -euo pipefail

HOST="${MOCAP_GPU_HOST:?set MOCAP_GPU_HOST to your GPU box ssh host}"
RDIR="${MOCAP_GPU_DIR:-~/movements}"
CENV="${MOCAP_GPU_ENV:-mocap}"

VIDEO="${1:?usage: process_remote.sh <video> --subject-height <m> [args...]}"
shift
CLIP_ID="$(basename "${VIDEO%.*}")"

echo "==> uploading $(basename "$VIDEO")"
ssh "$HOST" "mkdir -p $RDIR/inbox"
rsync -av --progress "$VIDEO" "$HOST:$RDIR/inbox/"

echo "==> processing on $HOST (clip: $CLIP_ID)"
ssh "$HOST" "cd $RDIR && MUJOCO_GL=egl conda run --no-capture-output -n $CENV \
  python -m mocap.process inbox/$(basename "$VIDEO") --clip-id '$CLIP_ID' $*"

echo "==> downloading dataset/$CLIP_ID"
mkdir -p "$(dirname "$0")/../dataset"
rsync -av --progress "$HOST:$RDIR/dataset/$CLIP_ID/" \
  "$(dirname "$0")/../dataset/$CLIP_ID/"

echo "==> done: dataset/$CLIP_ID/preview.mp4"
