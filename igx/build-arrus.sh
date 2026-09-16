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
#
# The cp310 wheel for a native install on the IGX host (Ubuntu 22.04, system Python 3.10):
#   ARRUS_PY_VERSION=3.10 ARRUS_BUILD_IMAGE=us4r-build-py310 igx/build-arrus.sh
# (image from igx/docker/Dockerfile.py310, built by build-images.sh). Each Python version gets its
# own build tree, igx/out/build-py<version>; the wheels collect in igx/out/wheel/. The runtime
# docker image is built only for the image's own Python (3.8); the cp310 wheel is for the host.
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
BUILD_DIR=$OUT/build-py$PYVER
mkdir -p "$BUILD_DIR" "$OUT/conan" "$OUT/wheel"

CONFIGURE='
set -e
export CONAN_USER_HOME=/conan
# conan profile: gcc 9 on aarch64 with the C++11 ABI. libstdc++ (old ABI) links but leaves
# libarrus-core with unresolved protobuf/boost symbols at load time.
if [ ! -f /conan/.conan/profiles/default ]; then conan profile new default --detect > /dev/null; fi
conan profile update settings.compiler.libcxx=libstdc++11 default
conan profile update settings.build_type=Release default
conan install /src -if /build --build=missing
# Point cmake at the requested interpreter, its headers and its shared library explicitly:
# FindPythonLibs does not look inside a pyenv prefix (the 3.10 image), and it must not pick
# another version from /usr when two are installed.
PYEXE=$(command -v python'"$PYVER"')
PYINC=$($PYEXE -c "import sysconfig; print(sysconfig.get_paths()[\"include\"])")
PYLIB=$($PYEXE -c "import sysconfig, os; print(os.path.join(sysconfig.get_config_var(\"LIBDIR\"), sysconfig.get_config_var(\"LDLIBRARY\")))")
echo "python: $PYEXE include $PYINC library $PYLIB"
cmake -S /src -B /build -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DUs4_ROOT_DIR=/us4 \
  -DARRUS_BUILD_PY=ON -DARRUS_PY_VERSION='"$PYVER"' -DARRUS_EMBED_DEPS=ON \
  -DPYTHON_EXECUTABLE=$PYEXE -DPYTHON_INCLUDE_DIR=$PYINC -DPYTHON_LIBRARY=$PYLIB
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

docker run --rm -v "$ROOT":/src -v "$BUILD_DIR":/build -v "$OUT/conan":/conan -v "$US4":/us4:ro "$IMAGE" -c "$SCRIPT"
# The container runs as root; hand the outputs back to the user so they can be cleaned or moved.
docker run --rm -v "$OUT":/out "$IMAGE" -c "chown -R $(id -u):$(id -g) /out/build-py$PYVER /out/conan /out/wheel"

WHEEL=$(ls "$BUILD_DIR"/api/python/dist/arrus-*-linux_aarch64.whl | head -1)
cp "$WHEEL" "$OUT/wheel/"
echo "wheel: $OUT/wheel/$(basename "$WHEEL")"
if [ "$PYVER" = "3.8" ]; then
  # The runtime image: a small build context holding only this wheel (the build tree is gigabytes).
  CTX=$(mktemp -d) && cp "$WHEEL" "$CTX/"
  echo "runtime image: us4r-arrus-runtime (wheel + cupy + tk)"
  docker build -q -f "$HERE/docker/Dockerfile.runtime" --build-arg BASE="$IMAGE" --build-arg WHEEL="$(basename "$WHEEL")" -t us4r-arrus-runtime "$CTX" > /dev/null
  rm -rf "$CTX"
  docker run --rm --gpus all us4r-arrus-runtime -c 'python3 -c "import arrus, arrus.session, cupy; print(\"arrus import OK, cupy\", cupy.__version__)"'
else
  echo "no runtime image for Python $PYVER (the image's python is 3.8); install the wheel on the host, see README section 5b"
fi
echo "bench binary: $BUILD_DIR/arrus/core/throughput-test"
