#!/bin/sh

TARGETPLATFORM=$1

if [ "$TARGETPLATFORM" = "linux/amd64" ]; \
    then export CMAKE_PLATFORM="x86_64"; \
elif [ "$TARGETPLATFORM" = "linux/arm64" ]; \
    then export CMAKE_PLATFORM="aarch64"; \
else echo "Unsupported platform: $TARGETPLATFORM"; exit 1; \
fi

wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${CMAKE_PLATFORM}.tar.gz \
 && tar -xvf cmake-${CMAKE_VERSION}-linux-${CMAKE_PLATFORM}.tar.gz --directory /opt \
 && mv /opt/cmake-${CMAKE_VERSION}-linux-${CMAKE_PLATFORM} /opt/cmake \
 && ln -s /opt/cmake/bin/* /bin
