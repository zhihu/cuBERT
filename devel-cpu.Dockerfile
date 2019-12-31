FROM ubuntu:18.04

WORKDIR /usr/src/cuBERT
COPY . .

# download MKL from official repo
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        software-properties-common \
        wget
RUN wget -O - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.repos.intel.com/mkl all main'

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        g++ \
        cmake \
        autoconf \
        automake \
        libtool \
        pkg-config \
        protobuf-compiler \
        libprotoc-dev \
        libprotobuf-dev \
        intel-mkl-64bit-2019.5-075 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN rm -rf build && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DcuBERT_ENABLE_MKL_SUPPORT=ON -DcuBERT_SYSTEM_MKL=ON -DcuBERT_SYSTEM_PROTOBUF=ON .. && \
    make
