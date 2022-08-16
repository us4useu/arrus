ARG TARGETPLATFORM
FROM --platform=$TARGETPLATFORM nvidia/cuda:11.2.0-devel-ubuntu20.04

ARG TARGETPLATFORM

RUN echo "PLATFORM $TARGETPLATFORM"

LABEL maintainer="us4us ltd. <support@us4us.eu>"
USER root

WORKDIR /tmp

# Settings
ENV CMAKE_VERSION=3.21.3
ENV PYTHON_VERSIONS="3.8"

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# Install requirements
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    $(for x in $PYTHON_VERSIONS; do printf "python$x python$x-dev "; done) \
    python3 \
    python3-dev \
    python3-pip \
    wget \
    git \
    vim \
    patchelf \
    g++-9 \
    doxygen \
    latexmk \
    texlive-latex-extra \
    texlive-fonts-recommended


RUN echo "alias python=python3" >> /root/.bashrc

# Cmake
COPY .docker/build/install_cmake.sh .
RUN chmod +x install_cmake.sh && ./install_cmake.sh $TARGETPLATFORM

# Swig
RUN apt-get update && apt-get install -yq --no-install-recommends libpcre3-dev \
 && wget https://netcologne.dl.sourceforge.net/project/swig/swig/swig-4.0.2/swig-4.0.2.tar.gz \
 && tar -xvf swig-4.0.2.tar.gz && cd swig-4.0.2 \
 && ./configure && make && make install

# Python dependencies and conan
RUN for x in $PYTHON_VERSIONS; do python$x -m pip install virtualenv setuptools==52.0.0 wheel==0.36.2 Jinja2==3.0.3 sphinx==3.3.1 sphinx_rtd_theme==0.5.0 six breathe docutils==0.16 "git+https://github.com/pjarosik/matlabdomain@master#egg=sphinxcontrib-matlabdomain" pydevops-us4us==0.1.0  && python$x -m pip install conan; done

# Set Git user mail and name (needed to autimatically publish docs on github pages).
RUN git config --global user.email "dev@us4us.eu" && git config --global user.name "us4us developer"

WORKDIR /

ENTRYPOINT ["/bin/bash"]
