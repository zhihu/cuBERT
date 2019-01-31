#include "math.h"
#include "cub/cub.cuh"
#include <cuda_runtime.h>

namespace cuBERT {
    __global__ void kernel_layer_norm_(float *inout,
                                       const int batch_size,
                                       const int channel,
                                       const float *__restrict__ beta,
                                       const float *__restrict__ gamma) {
        int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch_idx >= batch_size) {
            return;
        }

        // channel data: [batch_idx * channel, (batch_idx + 1) * channel)
        float mean = 0;
        float var = 0;
#pragma unroll
        for (int i = batch_idx * channel; i < (batch_idx + 1) * channel; ++i) {
            float t = __ldg(inout + i);
            mean += t;
            var += t * t;
        }
        mean = mean / channel;
        var = var / channel - mean * mean;

        // 1 / sqrt(var)
        var = rsqrtf(var + 1e-12);

#pragma unroll
        for (int i = 0; i < channel; ++i) {
            int j = batch_idx * channel + i;
            inout[j] = __ldg(beta + i) + __ldg(gamma + i) * var * (__ldg(inout + j) - mean);
        }
    }

    __host__ void layer_norm_(float *inout,
                              const int batch_size,
                              const int channel,
                              float *beta,
                              float *gamma,
                              void* stream) {
        const int blocks = (batch_size + 127) / 128;
        kernel_layer_norm_ << < blocks, 128, 0, (cudaStream_t) stream >> > (inout, batch_size, channel, beta, gamma);
    }

    __global__ void kernel_momentum_cub(const float *__restrict__ in,
                                        const float *__restrict__ inout,
                                        const int batch_size,
                                        const int channel,
                                        float *mean_out,
                                        float *var_out) {
        __shared__ typename cub::BlockReduce<float, 128>::TempStorage m_storage;
        __shared__ typename cub::BlockReduce<float, 128>::TempStorage v_storage;
        const float scale = 1.f / channel;
        for (int i = blockIdx.x; i < batch_size; i += gridDim.x) {
            float m_val = 0;
            float v_val = 0;
            for (int j = threadIdx.x; j < channel; j += blockDim.x) {
                const int X_index = i * channel + j;
                const float t = __ldg(in + X_index) + __ldg(inout + X_index);
                m_val += t;
                v_val += t * t;
            }
            m_val = cub::BlockReduce<float, 128>(m_storage).Sum(m_val);
            v_val = cub::BlockReduce<float, 128>(v_storage).Sum(v_val);
            if (threadIdx.x == 0) {
                const float mu = m_val * scale;
                mean_out[i] = mu;
                var_out[i] = v_val * scale - mu * mu;
            }
            __syncthreads();
        }
    }

    __global__ void kernel_batchnorm_(const float *__restrict__ in,
                                      float *inout,
                                      const int batch_size,
                                      const int channel,
                                      const float *__restrict__ mean_in,
                                      const float *__restrict__ var_in,
                                      const float *__restrict__ beta,
                                      const float *__restrict__ gamma) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * channel) {
            return;
        }

        int batch_idx = idx / channel;
        int channel_idx = idx % channel;

        float mean = __ldg(mean_in + batch_idx);
        float var = __ldg(var_in + batch_idx);

        // 1 / sqrt(var)
        var = rsqrtf(var + 1e-12);

        inout[idx] = __ldg(beta + channel_idx) +
                __ldg(gamma + channel_idx) * var * (__ldg(inout + idx) + __ldg(in + idx) - mean);
    }

    __host__ void layer_norm_(float *in,
                              float *inout,
                              const int batch_size,
                              const int channel,
                              float *mean_gpu,
                              float *var_gpu,
                              float *beta,
                              float *gamma,
                              void* stream) {
        kernel_momentum_cub <<<batch_size, 128, 0, (cudaStream_t) stream>>> (in, inout, batch_size, channel, mean_gpu, var_gpu);

        const int all_blocks = (batch_size * channel + 127) / 128;
        kernel_batchnorm_ <<<all_blocks, 128, 0, (cudaStream_t) stream>>> (in, inout, batch_size, channel, mean_gpu, var_gpu, beta, gamma);
    }
}
