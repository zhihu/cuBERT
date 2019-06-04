Fast implementation of BERT inference directly on NVIDIA (CUDA, CUBLAS) and Intel MKL
=====================================================================================

[![Build Status](https://travis-ci.org/zhihu/cuBERT.svg?branch=master)](https://travis-ci.org/zhihu/cuBERT)

Highly customized and optimized BERT inference directly on NVIDIA (CUDA,
CUBLAS) or Intel MKL, *without* tensorflow and its framework overhead.

**ONLY** BERT (Transformer) is supported.

# Benchmark

### Environment

* Tesla P4
* 28 * Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
* Debian GNU/Linux 8 (jessie)
* gcc (Debian 4.9.2-10+deb8u1) 4.9.2
* CUDA: release 9.0, V9.0.176
* MKL: 2019.0.1.20181227
* tensorflow: 1.12.0
* BERT: seq_length = 32

### GPU (cuBERT)

|batch size|128 (ms) |32 (ms) |
|---       |---      |---     |
|tensorflow|255.2    |70.0    |
|cuBERT    |**184.6**|**54.5**|

### CPU (mklBERT)

|batch size|128 (ms) |1 (ms)  |
|---       |---      |---     |
|tensorflow|1504.0   |69.9    |
|mklBERT   |**984.9**|**24.0**|

Note: MKL should be run under `OMP_NUM_THREADS=?` to control its thread
number. Other environment variables and their possible values includes:

* `KMP_BLOCKTIME=0`
* `KMP_AFFINITY=granularity=fine,verbose,compact,1,0`

### Mixed Precision

cuBERT can be accelerated by [Tensor Core](https://developer.nvidia.com/tensor-cores)
and [Mixed Precision](https://devblogs.nvidia.com/tensor-cores-mixed-precision-scientific-computing)
on NVIDIA Volta and Turing GPUs. We support mixed precision as variables
stored in fp16 with computation taken in fp32. The typical accuracy error
is less than 1% compared with single precision inference, while the speed
achieves more than 2x acceleration.

# API

[API .h header](/src/cuBERT.h)

### Pooler

We support following 2 pooling method.

* The standard BERT pooler, which is defined as:

```python
with tf.variable_scope("pooler"):
  # We "pool" the model by simply taking the hidden state corresponding
  # to the first token. We assume that this has been pre-trained
  first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
  self.pooled_output = tf.layers.dense(
    first_token_tensor,
    config.hidden_size,
    activation=tf.tanh,
    kernel_initializer=create_initializer(config.initializer_range))
```

* Simple average pooler:

```python
self.pooled_output = tf.reduce_mean(self.sequence_output, axis=1)
```

### Output

Following outputs are supported:

|cuBERT_OutputType      |python code                   |
|---                    |---                           |
|cuBERT_LOGITS          |[`model.get_pooled_output() * output_weights + output_bias`](https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/run_classifier.py#L607)|
|cuBERT_PROBS           |`probs = tf.nn.softmax(logits, axis=-1)`|
|cuBERT_POOLED_OUTPUT   |`model.get_pooled_output()`   |
|cuBERT_SEQUENCE_OUTPUT |`model.get_sequence_output()` |
|cuBERT_EMBEDDING_OUTPUT|`model.get_embedding_output()`|

# Build from Source

```shell
mkdir build && cd build
# if build with CUDA
cmake -DCMAKE_BUILD_TYPE=Release -DcuBERT_ENABLE_GPU=ON ..
# or build with MKL
cmake -DCMAKE_BUILD_TYPE=Release -DcuBERT_ENABLE_MKL_SUPPORT=ON ..
make -j4

# install to /usr/local
# it will also install MKL if -DcuBERT_ENABLE_MKL_SUPPORT=ON
sudo make install
```

If you would like to run tfBERT_benchmark for performance comparison,
please first install tensorflow C API from https://www.tensorflow.org/install/lang_c.

### Run Unit Test

Download BERT test model `bert_frozen_seq32.pb` and `vocab.txt` from
[Dropbox](https://www.dropbox.com/sh/ulcdmu9ysyg5lk7/AADndzKXOrHIXLYRc5k60Q-Ta?dl=0), 
and put them under dir `build` before run `make test` or `./cuBERT_test`.

### Python

We provide simple Python wrapper by Cython, and it can be built and 
installed after C++ building as follows:

```shell
cd python
python setup.py bdist_wheel

# install
pip install dist/cuBERT-xxx.whl

# test
python cuBERT_test.py
```

Please check the Python API usage and examples at [cuBERT_test.py](/python/cuBERT_test.py)	
for more details.

### Java

Java wrapper is implemented through [JNA](https://github.com/java-native-access/jna)
. After installing maven and C++ building, it can be built as follows:

```shell
cd java
mvn clean package # -DskipTests
```

When using Java JAR, you need to specify `jna.library.path` to the 
location of `libcuBERT.so` if it is not installed to the system path.
And `jna.encoding` should be set to UTF8 as `-Djna.encoding=UTF8`
in the JVM start-up script.

Please check the Java API usage and example at [ModelTest.java](/java/src/test/java/com/zhihu/cubert/ModelTest.java)
for more details.

# Install

Pre-built python binary package (currently only with MKL on Linux) can
be installed as follows:

* Download and install [MKL](https://github.com/intel/mkl-dnn/releases)
to system path.

* Download the wheel package and `pip install cuBERT-xxx-linux_x86_64.whl`

* run `python -c 'import libcubert'` to verify your installation.

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

# Threading

We assume the typical usage case of cuBERT is for online serving, where
concurrent requests of different batch_size should be served as fast as
possible. Thus, throughput and latency should be balanced, especially in
pure CPU environment.

As the vanilla [class Bert](/src/cuBERT/Bert.h) is not thread-safe
because of its internal buffers for computation, a wrapper [class BertM](/src/cuBERT/BertM.h)
is written to hold locks of different `Bert` instances for thread safety.
`BertM` will choose one underlying `Bert` instance by a round-robin
manner, and consequence requests of the same `Bert` instance might be
queued by its corresponding lock.

### GPU

One `Bert` is placed on one GPU card. The maximum concurrent requests is
the number of usable GPU cards on one machine, which can be controlled
by `CUDA_VISIBLE_DEVICES` if it is specified.

### CPU

For pure CPU environment, it is more complicate than GPU. There are 2
level of parallelism:

1. Request level. Concurrent requests will compete CPU resource if the
online server itself is multi-threaded. If the server is single-threaded
(for example some server implementation in Python), things will be much
easier.

2. Operation level. The matrix operations are parallelized by OpenMP and
MKL. The maximum parallelism is controlled by `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, and many other environment variables. We refer our
users to first read [Using Threaded Intel® MKL in Multi-Thread Application](https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application)
 and [Recommended settings for calling Intel MKL routines from multi-threaded applications](https://software.intel.com/en-us/articles/recommended-settings-for-calling-intel-mkl-routines-from-multi-threaded-applications)
.

Thus, we introduce `CUBERT_NUM_CPU_MODELS` for better control of request
level parallelism. This variable specifies the number of `Bert` instances
created on CPU/memory, which acts same like `CUDA_VISIBLE_DEVICES` for
GPU.

* If you have limited number of CPU cores (old or desktop CPUs, or in
Docker), it is not necessary to use `CUBERT_NUM_CPU_MODELS`. For example
4 CPU cores, a request-level parallelism of 1 and operation-level
parallelism of 4 should work quite well.

* But if you have many CPU cores like 40, it might be better to try with
request-level parallelism of 5 and operation-level parallelism of 8.

Again, the per request latency and overall throughput should be balanced,
and it diffs from model `seq_length`, `batch_size`, your CPU cores, your
server QPS, and many many other things. You should take a lot benchmark
to achieve the best trade-off. Good luck!

# Authors

* fanliwen
* wangruixin
* fangkuan
* sunxian
