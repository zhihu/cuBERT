Fast implementation of BERT inference directly on NVIDIA (CUDA, CUBLAS) and Intel MKL
=====================================================================================

Highly customized and optimized BERT inference directly on NVIDIA (CUDA,
CUBLAS) or Intel MKL, without tensorflow and its framework overhead.

ONLY BERT (Transformer) is supported.

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

|batch size|128 (ms)|32 (ms)|
|---       |---     |---    |
|tensorflow|255.2   |70.0   |
|cuBERT    |184.6   |54.5   |

### CPU (mklBERT)

|batch size|128 (ms)|1 (ms)|
|---       |---     |---   |
|tensorflow|1504.0  |69.9  |
|mklBERT   |984.9   |24.0  |

Note: MKL should be run under `OMP_NUM_THREADS=? KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,verbose,compact,1,0`

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
|cuBERT_LOGITS          |`output_weights = tf.get_variable("output_weights", [1, hidden_size])` <br> `logits = tf.matmul(output_layer, output_weights, transpose_b=True)`|
|cuBERT_POOLED_OUTPUT   |`model.get_pooled_output()`   |
|cuBERT_SEQUENCE_OUTPUT |`model.get_sequence_output()` |
|cuBERT_EMBEDDING_OUTPUT|`model.get_embedding_output()`|

# Build

```shell
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4

# install to /usr/local
sudo make install
```

### Run Unit Test

Download BERT test model `bert_frozen_seq32.pb` and `vocab.txt` from
[Google Drive](https://drive.google.com/drive/folders/1UG9ijvwcf_Fe50EPiE8ObAJbupFLrs-k?usp=sharing).

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

# Authors

* fanliwen
* wangruixin
* fangkuan
