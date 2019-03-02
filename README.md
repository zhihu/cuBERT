Fast implementation of BERT inference directly on NVIDIA (CUDA, CUBLAS) and Intel MKL
=====================================================================================

# Benchmark

### ai-gpu-01

* 2 * Tesla P4
* 28 * Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
* Debian GNU/Linux 8 (jessie)
* gcc (Debian 4.9.2-10+deb8u1) 4.9.2
* CUDA: release 9.0, V9.0.176
* MKL: 2019.0.1.20181227
* tensorflow: 1.12.0
* BERT: seq_length = 32

|batch size    |128 /ms|1 /ms|
|---           |---    |---  |
|tensorflow_gpu|255.2  |     |
|tensorflow_cpu|1504.0 |69.9 |
|cuBERT (GPU)  |184.6  |     |
|mklBERT (CPU) |984.9  |24.0 |

Note: MKL should be run under `OMP_NUM_THREADS=? KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,verbose,compact,1,0`

# API

[API .h header](/src/cuBERT.h)

# Build

```shell
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4

# install to /usr/local
sudo make install
```

# Dependency

### Protobuf

This library is built and linked against Google Protobuf 3.6.0 (same as
tensorflow 1.12). As different versions of Protobuf can not co-exist in 
one single program, cuBERT is in-compatible with other Protobuf versions.

Check tensorflow protobuf version at their [tensorflow/workspace.bzl](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/workspace.bzl).

|tensorflow|protobuf|
|---       |---     |
|1.13.1    |3.6.1.2 |
|1.12.0    |3.6.0   |
|1.11.0    |3.6.0   |
|1.10.1    |3.6.0   |
|1.10.0    |3.6.0   |

### CUDA

Libraries compiled by CUDA with different versions are not compatible.

### MKL

MKL is dynamically linked. We install both cuBERT and MKL in `sudo make install`.
