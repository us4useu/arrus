#!/bin/bash
# Builds ARRUS for the IGX inside the us4r-build-py image: the C++ core, the bench examples
# (throughput-test, minimal-acquisition) and the Python wheel, which embeds libarrus-core and
# the us4OEM driver libraries from the driver install. Then builds the runtime image with the
# wheel installed. Everything lands under igx/out (git-ignored).
#
#   igx/build-arrus.sh                 full build (first run: conan compiles boost/protobuf, ~1 h on the IGX)
#   igx/build-arrus.sh --rebuild       skip conan/cmake configure, just rebuild changed sources
#
# Inputs (environment, all optional):
#   US4_ROOT_DIR       driver install with lib64/ and include/   (default igx/out/us4-install, see build-driver.sh)
#   ARRUS_PY_VERSION   python for the wheel                        (default 3.8, the image's python3)
#   ARRUS_BUILD_IMAGE  build image                                 (default us4r-build-py, see build-images.sh)
#   J                  parallel jobs                               (default nproc)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
OUT=$HERE/out
US4=${US4_ROOT_DIR:-$OUT/us4-install}
PYVER=${ARRUS_PY_VERSION:-3.8}
IMAGE=${ARRUS_BUILD_IMAGE:-us4r-build-py}
J=${J:-$(nproc)}
MODE=${1:-full}

[ -d "$US4/lib64" ] && [ -d "$US4/include" ] || { echo "driver install not found at $US4 (lib64/ and include/): run igx/build-driver.sh first, or set US4_ROOT_DIR"; exit 1; }
docker image inspect "$IMAGE" > /dev/null 2>&1 || { echo "image $IMAGE missing: run igx/build-images.sh first"; exit 1; }
mkdir -p "$OUT/build" "$OUT/conan"

CONFIGURE='
set -e
export CONAN_USER_HOME=/conan
# conan profile: gcc 9 on aarch64 with the C++11 ABI. libstdc++ (old ABI) links but leaves
# libarrus-core with unresolved protobuf/boost symbols at load time.
if [ ! -f /conan/.conan/profiles/default ]; then conan profile new default --detect > /dev/null; fi
conan profile update settings.compiler.libcxx=libstdc++11 default
conan profile update settings.build_type=Release default
conan install /src -if /build --build=missing
cmake -S /src -B /build -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DUs4_ROOT_DIR=/us4 \
  -DARRUS_BUILD_PY=ON -DARRUS_PY_VERSION='"$PYVER"' -DARRUS_EMBED_DEPS=ON
'
BUILD='
set -e
cmake --build /build -j '"$J"'
ls -la /build/api/python/dist/*.whl
'
case "$MODE" in
  full)      SCRIPT="$CONFIGURE$BUILD" ;;
  --rebuild) SCRIPT="$BUILD" ;;
  *) echo "usage: $0 [--rebuild]"; exit 2 ;;
esac

docker run --rm -v "$ROOT":/src -v "$OUT/build":/build -v "$OUT/conan":/conan -v "$US4":/us4:ro "$IMAGE" -c "$SCRIPT"

WHEEL=$(ls "$OUT"/build/api/python/dist/arrus-*-linux_aarch64.whl | head -1)
echo "wheel: $WHEEL"
# The runtime image: a small build context holding only the wheel (the build tree is gigabytes).
rm -rf "$OUT/wheel" && mkdir -p "$OUT/wheel" && cp "$WHEEL" "$OUT/wheel/"
echo "runtime image: us4r-arrus-runtime (wheel + cupy + tk)"
docker build -q -f "$HERE/docker/Dockerfile.runtime" --build-arg BASE="$IMAGE" --build-arg WHEEL="$(basename "$WHEEL")" -t us4r-arrus-runtime "$OUT/wheel" > /dev/null
docker run --rm --gpus all us4r-arrus-runtime -c 'python3 -c "import arrus, arrus.session, cupy; print(\"arrus import OK, cupy\", cupy.__version__)"'
echo "bench binary: $OUT/build/arrus/core/throughput-test"
