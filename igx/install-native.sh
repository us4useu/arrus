#!/bin/bash
# Installs the ARRUS cp310 wheel natively on the IGX host (Ubuntu 22.04, system Python 3.10), no
# docker at run time: a venv with the wheel, cupy and the CUDA runtime libraries from PyPI (the
# IGX ships the CUDA driver only; nothing in /usr/local/cuda is required). Idempotent.
#
#   igx/install-native.sh [wheel]        default wheel: igx/out/wheel/arrus-*-cp310-*.whl
#   . ~/arrus-venv/bin/activate          then: cd <dir with us4r.prototxt> && python3 plane_wave_imaging.py
#
# Needs network (PyPI). No root: the venv is created without pip and pip is bootstrapped from
# bootstrap.pypa.io. The venv sees the system packages (tkinter for the examples' window).
# CUDA: the runtime wheels are pinned to the oldest 12.x on PyPI (12.3), which runs on the IGX's
# 535 driver (CUDA 12.2) through CUDA minor-version compatibility; cupy compiles its kernels to
# SASS for the local GPU, so no newer PTX reaches the driver. Set CUDA_WHEELS_VERSION to pin
# another 12.x if your driver is newer.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=${ARRUS_VENV:-$HOME/arrus-venv}
WHEEL=${1:-$(ls "$HERE"/out/wheel/arrus-*-cp310-cp310-linux_aarch64.whl 2>/dev/null | head -1)}
CU=${CUDA_WHEELS_VERSION:-12.3}
[ -n "$WHEEL" ] && [ -f "$WHEEL" ] || { echo "cp310 wheel not found: build it with ARRUS_PY_VERSION=3.10 ARRUS_BUILD_IMAGE=us4r-build-py310 igx/build-arrus.sh, or pass its path"; exit 1; }
python3 --version | grep -q ' 3\.10\.' || { echo "system python3 is $(python3 --version), the wheel is for 3.10"; exit 1; }
[ -d "$VENV" ] || python3 -m venv --without-pip --system-site-packages "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
python3 -m pip --version > /dev/null 2>&1 || curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --quiet
pip install --quiet --upgrade "numpy<2" "scipy<1.14"
pip install --quiet "$WHEEL"
case "$CU" in
  12.3) pip install --quiet "cupy-cuda12x<14" "nvidia-cuda-runtime-cu12==12.3.*" "nvidia-cuda-nvrtc-cu12==12.3.*" "nvidia-cublas-cu12==12.3.*" \
          "nvidia-cufft-cu12==11.0.12.*" "nvidia-curand-cu12==10.3.4.*" "nvidia-cusolver-cu12==11.5.4.*" "nvidia-cusparse-cu12==12.2.0.*" "nvidia-nvjitlink-cu12==12.3.*" ;;
  *)    pip install --quiet "cupy-cuda12x<14" "nvidia-cuda-runtime-cu12==$CU.*" "nvidia-cuda-nvrtc-cu12==$CU.*" "nvidia-cublas-cu12==$CU.*" \
          nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 "nvidia-nvjitlink-cu12==$CU.*" ;;
esac
# The CUDA libraries live inside the venv (site-packages/nvidia/*/lib); cupy finds them through
# the loader path, so the activation script exports it, together with the bench defaults.
NVLIBS=$(find "$VENV/lib/python3.10/site-packages/nvidia" -maxdepth 2 -name lib -type d | paste -sd:)
if ! grep -q 'ARRUS IGX' "$VENV/bin/activate"; then
  cat >> "$VENV/bin/activate" <<EOT

# ARRUS IGX: CUDA runtime wheels for cupy, and the Ethernet bench defaults (igx/install-native.sh)
export LD_LIBRARY_PATH="$NVLIBS\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export US4R_ETH_DEVICES="\${US4R_ETH_DEVICES:-192.168.0.2,192.168.4.2}"
export US4R_ETH_ALLOW_WRITES=1
EOT
fi
. "$VENV/bin/activate"
python3 - <<'PY'
import arrus, arrus.session, cupy, numpy, matplotlib
x = cupy.arange(8); assert int(x.sum()) == 28
print(f"arrus {arrus.__version__} native, numpy {numpy.__version__}, cupy {cupy.__version__} on the GPU: OK")
PY
echo "venv: $VENV  (. $VENV/bin/activate)"
