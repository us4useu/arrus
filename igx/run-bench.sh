#!/bin/bash
# Runs the C++ throughput bench (arrus/core/examples/ThroughputTest.cpp) from the build tree
# against the two boards over Ethernet, in the build image, with the RDMA receiver.
#
#   igx/run-bench.sh <label> <nSamples> <rxDepth> <hostDepth> <seconds> <pri_us> [pri_us ...]
#   e.g. igx/run-bench.sh single 1024 8 8 10 1000        # 1 MiB elements, rx 8, 10 s at PRI 1 ms
#        THROUGHPUT_NTX=16 igx/run-bench.sh pwi 1024 4 4 30 1000
#
# Config: igx/config/us4r_eth_bench_synthetic64.prototxt (no probe needed; HV is enabled by
# THROUGHPUT_HV=10 so the pulsers of STANDARD boards do not fault). Output: igx/out/bench/<label>.log
# and the summary line "PRI <us> us: N frames in T s [steady F fps, M MB/s] ...". THROUGHPUT_* and
# the driver's US4R_ETH_* switches are passed through when set (see the bench file header).
set -euo pipefail
[ $# -ge 6 ] || { echo "usage: $0 <label> <nSamples> <rxDepth> <hostDepth> <seconds> <pri_us> [pri_us ...]"; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/out
BIN=$OUT/build/arrus/core/throughput-test
[ -x "$BIN" ] || { echo "bench binary missing at $BIN: run igx/build-arrus.sh first"; exit 1; }
US4=${US4_ROOT_DIR:-$OUT/us4-install}
IMAGE=${ARRUS_BUILD_IMAGE:-us4r-build-py}
DEVICES=${US4R_ETH_DEVICES:-192.168.0.2,192.168.4.2}
CFG=${CFG:-$HERE/config/us4r_eth_bench_synthetic64.prototxt}
LABEL=$1; shift
mkdir -p "$OUT/bench"
ENV_ARGS=(-e US4R_ETH_ALLOW_WRITES=1 -e US4R_ETH_DEVICES="$DEVICES" -e THROUGHPUT_HV="${THROUGHPUT_HV:-10}" -e THROUGHPUT_MODE="${THROUGHPUT_MODE:-HOST}")
for v in $(env | grep -oE '^(THROUGHPUT|US4R_ETH|ARRUS_HOST|ARRUS_SYNC)_[A-Z0-9_]+' | grep -vE '^(THROUGHPUT_HV|THROUGHPUT_MODE)$'); do
  ENV_ARGS+=(-e "$v=${!v}")
done
docker run --rm --network host \
  --device /dev/infiniband/uverbs0 --device /dev/infiniband/uverbs1 --device /dev/infiniband/rdma_cm --ulimit memlock=-1 \
  -v "$OUT/build":/build -v "$US4":/us4:ro -v "$(dirname "$CFG")":/cfg:ro \
  -e LD_LIBRARY_PATH=/us4/lib64:/build/arrus/core "${ENV_ARGS[@]}" \
  "$IMAGE" -c "cd /cfg && /build/arrus/core/throughput-test ./$(basename "$CFG") $*" 2>&1 | tee "$OUT/bench/$LABEL.log" | grep -E '^PRI |STALL|^content|rejected \(|\[ERROR\]|\[WARNING\] RDMA' || true
echo "log: $OUT/bench/$LABEL.log"
