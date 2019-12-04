FROM ubuntu:18.04

WORKDIR /usr/src/cuBERT
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        g++ \
        cmake \
        autoconf \
        automake \
        libtool \
        pkg-config \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN rm -rf build && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DcuBERT_ENABLE_MKL_SUPPORT=ON .. && \
    make
