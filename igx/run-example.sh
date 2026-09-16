#!/bin/bash
# Runs a Python script (an ARRUS example) against the two us4OEM+ boards over Ethernet, in the
# runtime image, with the RDMA receiver and, when DISPLAY is set, the script's live window.
#
#   igx/run-example.sh <script.py> [args...]
#
# The script runs with the current directory mounted at /work, so put us4r.prototxt next to it
# (igx/config has two: copy one as us4r.prototxt). Display: run from `ssh -X` (DISPLAY set by ssh,
# cookie in ~/.Xauthority) or on the IGX desktop (DISPLAY=:0 after `xhost +local:`); without a
# DISPLAY the container gets MPLBACKEND=Agg and the examples' Display2D returns at once.
#
# Environment passed through when set: US4R_ETH_DEVICES (default 192.168.0.2,192.168.4.2, master
# first), US4R_ETH_RECEIVER (unset = rdma), US4R_ETH_BUSY_POLL_US, US4R_ETH_RX_ROW_MIRROR,
# US4R_ETH_BYTE_SWAP, ARRUS_HOST_STALL_MS, ARRUS_HOST_PARK, and any variable named in EXTRA_ENV.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <script.py> [args...]"; exit 2; }
IMAGE=${ARRUS_RUNTIME_IMAGE:-us4r-arrus-runtime}
DEVICES=${US4R_ETH_DEVICES:-192.168.0.2,192.168.4.2}
docker image inspect "$IMAGE" > /dev/null 2>&1 || { echo "image $IMAGE missing: run igx/build-arrus.sh first"; exit 1; }
ENV_ARGS=(-e US4R_ETH_ALLOW_WRITES=1 -e US4R_ETH_DEVICES="$DEVICES")
for v in US4R_ETH_RECEIVER US4R_ETH_BUSY_POLL_US US4R_ETH_RX_ROW_MIRROR US4R_ETH_BYTE_SWAP US4R_ETH_IBV_DEVICE \
         ARRUS_HOST_STALL_MS ARRUS_HOST_PARK ARRUS_HOST_HS_RESUME ${EXTRA_ENV:-}; do
  [ -n "${!v:-}" ] && ENV_ARGS+=(-e "$v=${!v}")
done
DISPLAY_ARGS=()
if [ -n "${DISPLAY:-}" ]; then
  XAUTH=${XAUTHORITY:-$HOME/.Xauthority}
  DISPLAY_ARGS=(-e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix)
  [ -f "$XAUTH" ] && DISPLAY_ARGS+=(-v "$XAUTH":/root/.Xauthority:ro)
else
  DISPLAY_ARGS=(-e MPLBACKEND=Agg)
fi
# A script given by a path outside the current directory is mounted read-only at /script;
# the working directory (config, outputs) is always the current one.
SCRIPT=$1; shift
SCRIPT_ARGS=()
if [ -f "$SCRIPT" ]; then
  SDIR=$(cd "$(dirname "$SCRIPT")" && pwd)
  if [ "$SDIR" != "$PWD" ]; then SCRIPT_ARGS=(-v "$SDIR":/script:ro); SCRIPT="/script/$(basename "$SCRIPT")"; fi
fi
[ -t 0 ] && TTY=(-it) || TTY=()
exec docker run --rm "${TTY[@]}" --network host --gpus all \
  --device /dev/infiniband/uverbs0 --device /dev/infiniband/uverbs1 --device /dev/infiniband/rdma_cm --ulimit memlock=-1 \
  -v "$PWD":/work "${SCRIPT_ARGS[@]}" "${ENV_ARGS[@]}" "${DISPLAY_ARGS[@]}" \
  "$IMAGE" -c "cd /work && exec python3 $SCRIPT $*"
