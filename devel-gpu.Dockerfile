FROM nvidia/cuda:10.0-devel-ubuntu18.04

WORKDIR /usr/src/cuBERT
COPY . .

# use a recent cmake which solves https://gitlab.kitware.com/cmake/cmake/issues/18290
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        software-properties-common \
        wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        autoconf \
        automake \
        libtool \
        pkg-config \
        protobuf-compiler \
        libprotoc-dev \
        libprotobuf-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN rm -rf build && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DcuBERT_ENABLE_GPU=ON -DCUDA_ARCH_NAME=Common -DcuBERT_SYSTEM_PROTOBUF=ON .. && \
    make
