#!/bin/bash
# Builds the us4OEM driver with its Ethernet port (us4r-api, branch ref-M_OEM-296) in the
# us4r-build image and installs it to igx/out/us4-install (lib64/, include/, bin/), the tree
# igx/build-arrus.sh links ARRUS against. Follows the "Building" section of the us4r-api README.
#
#   US4R_API_DIR=/path/to/us4r-api igx/build-driver.sh          (default: ../us4r-api next to this repo)
#   igx/build-driver.sh --rebuild                               skip conan/cmake configure
#
# First run: conan builds Boost 1.70 from source on aarch64 (45-90 minutes, once; the cache is
# kept in igx/out/driver/conan-cache). The driver has no release tags; the reference bench ran
# commit 8a977c1a (marker mirror-unless-bit12-2026-09-16dh) or later on ref-M_OEM-296. The
# marker of the installed library: strings igx/out/us4-install/lib64/libUs4OEM.so | grep US4R-ETH-BUILD
# US4R_EMBED_DEPS=ON puts the boost shared libraries the driver links into lib64/ (with an
# $ORIGIN rpath): without them ARRUS cannot link against libUs4OEM.so, and the wheel embeds them.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
API=${US4R_API_DIR:-$(cd "$HERE/../.." && pwd)/us4r-api}
OUT=$HERE/out
SCRATCH=$OUT/driver
INSTALL=${US4_INSTALL_DIR:-$OUT/us4-install}   # override only to build into another tree
J=${J:-$(nproc)}
MODE=${1:-full}
[ -f "$API/CMakeLists.txt" ] && [ -f "$API/.conan/linux_aarch64.profile" ] || { echo "us4r-api checkout not found at $API (set US4R_API_DIR)"; exit 1; }
docker image inspect us4r-build > /dev/null 2>&1 || { echo "image us4r-build missing: run igx/build-images.sh first"; exit 1; }
mkdir -p "$SCRATCH/build" "$SCRATCH/conan-cache" "$INSTALL"
echo "us4r-api at $API: $(git -C "$API" rev-parse --short HEAD) ($(git -C "$API" branch --show-current))"
CONFIGURE='
set -e
conan install /usr/src/us4r-api --build missing -pr /usr/src/us4r-api/.conan/linux_aarch64.profile -s build_type=Release
cmake /usr/src/us4r-api -DCMAKE_BUILD_TYPE=Release -DUS4R_BUILD_KERNEL_SPACE_DRIVER=OFF -DUS4R_EMBED_DEPS=ON
'
BUILD='
set -e
make -j '"$J"'
rm -rf /install/lib64 /install/include /install/bin
cmake --install /scratch/build --prefix /install > /dev/null
echo "installed: $(ls /install)"
strings /install/lib64/libUs4OEM.so | grep -oE "US4R-ETH-BUILD=[^ \"]+" | head -1
'
case "$MODE" in
  full)      SCRIPT="$CONFIGURE$BUILD" ;;
  --rebuild) SCRIPT="$BUILD" ;;
  *) echo "usage: $0 [--rebuild]"; exit 2 ;;
esac
docker run --rm -v "$API":/usr/src/us4r-api -v "$SCRATCH":/scratch -v "$INSTALL":/install \
  -e CONAN_USER_HOME=/scratch/conan-cache -w /scratch/build us4r-build -c "$SCRIPT"
# The container runs as root; hand the scratch tree and the install back to the user.
docker run --rm -v "$SCRATCH":/scratch -v "$INSTALL":/install us4r-build -c "chown -R $(id -u):$(id -g) /scratch /install"
echo "driver install: $INSTALL"
