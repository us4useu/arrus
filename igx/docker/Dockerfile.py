# ARRUS build image with the Python toolchain: SWIG for the bindings, numpy/scipy/matplotlib for
# the package and its examples. Base: the us4r-build image built from the us4r-api repository
# (.docker/build/Dockerfile, linux/arm64: CUDA 11.7 devel on Ubuntu 20.04, gcc 9, cmake 3.21,
# conan 1.59, python3.8, libibverbs-dev).
ARG BASE=us4r-build
FROM ${BASE}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends swig python3-dev python3-pip \
 && rm -rf /var/lib/apt/lists/* \
 && pip3 install --no-cache-dir numpy scipy matplotlib
