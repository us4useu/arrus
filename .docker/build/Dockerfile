ARG TARGETPLATFORM
FROM --platform=$TARGETPLATFORM nvidia/cuda:11.2.0-devel-ubuntu20.04

ARG TARGETPLATFORM
ARG PYTHON_VERSION="3.8"

RUN echo "PLATFORM $TARGETPLATFORM"

LABEL maintainer="us4us ltd. <support@us4us.eu>"
USER root

WORKDIR /tmp

# Settings
ENV CMAKE_VERSION=3.21.3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# Install requirements

# For Python 3.10 on ubuntu20.04 we need to use external ppa
RUN if [ "$PYTHON_VERSION" = "3.10" ] ; then apt-get update && apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa -y ; fi

RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-distutils \
    wget \
    git \
    vim \
    patchelf \
    g++-9 \
    doxygen \
    latexmk \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-font-utils \
    ghostscript \
    curl

# Python PIP
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}


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
RUN python${PYTHON_VERSION} -m pip install virtualenv setuptools==67.8.0 wheel==0.36.2 Jinja2==3.0.3 sphinx==3.3.1 sphinx_rtd_theme==0.5.0 six breathe docutils==0.16 "git+https://github.com/pjarosik/matlabdomain@master#egg=sphinxcontrib-matlabdomain" pydevops-us4us==0.2.0 && python${PYTHON_VERSION} -m pip install conan==1.59.0

# Set Git user mail and name (needed to automatically publish docs on github pages).
RUN git config --global user.email "dev@us4us.eu" && git config --global user.name "us4us developer"

WORKDIR /

ENTRYPOINT ["/bin/bash"]
