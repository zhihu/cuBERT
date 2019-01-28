Fast implementation of BERT inference directly on NVIDIA CUDA and CUBLAS
========================================================================

# Benchmark

### ai-gpu-01

* 2 * Tesla P4
* 28 * Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
* Debian GNU/Linux 8 (jessie)
* gcc (Debian 4.9.2-10+deb8u1) 4.9.2
* CUDA: release 9.0, V9.0.176
* tensorflow: 1.12.0

|          |ms   |
|---       |---  |
|tensorflow|255.2|
|cuBERT    |184.6|

# API

[API .h header](/cuBERT.h)
