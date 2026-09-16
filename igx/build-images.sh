#!/bin/bash
# Builds the docker images for the IGX, in order:
#   us4r-build       the us4us build image (CUDA 11.7 devel, gcc 9, cmake, conan, python3.8,
#                    libibverbs-dev) from the us4r-api repository's .docker/build/Dockerfile
#   us4r-build-py    + SWIG, numpy, scipy, matplotlib          (igx/docker/Dockerfile.py)
#   us4r-build-py310 + Python 3.10 built with pyenv               (igx/docker/Dockerfile.py310)
# The runtime image (wheel + cupy + tk) is built by igx/build-arrus.sh once the wheel exists.
#
#   US4R_API_DIR=/path/to/us4r-api igx/build-images.sh     (default: ../us4r-api next to this repo)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
API=${US4R_API_DIR:-$(cd "$HERE/../.." && pwd)/us4r-api}
[ -f "$API/.docker/build/Dockerfile" ] || { echo "us4r-api checkout not found at $API (set US4R_API_DIR)"; exit 1; }
if ! docker image inspect us4r-build > /dev/null 2>&1; then
  echo "building us4r-build from $API/.docker/build/Dockerfile"
  docker build -f "$API/.docker/build/Dockerfile" --build-arg TARGETPLATFORM=linux/arm64 -t us4r-build "$API"
else
  echo "us4r-build exists, not rebuilt (docker rmi us4r-build to force)"
fi
docker build -f "$HERE/docker/Dockerfile.py" --build-arg BASE=us4r-build -t us4r-build-py "$HERE/docker"
# Python 3.10 from source (pyenv), for the cp310 wheel that installs natively on the IGX host.
docker build -f "$HERE/docker/Dockerfile.py310" --build-arg BASE=us4r-build-py -t us4r-build-py310 "$HERE/docker"
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E '^us4r-build'
