#include "math.h"
#include "cub/cub.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "LayerNorm.h"

namespace cuBERT {

    template <typename T>
    __global__ void kernel_momentum_cub(const T *__restrict__ in,
                                        const T *__restrict__ inout,
                                        const int batch_size,
                                        const int channel,
                                        T *mean_out,
                                        T *var_out) {
        __shared__ typename cub::BlockReduce<float, 128>::TempStorage m_storage;
        __shared__ typename cub::BlockReduce<float, 128>::TempStorage v_storage;
        const float scale = 1.f / channel;
        for (int i = blockIdx.x; i < batch_size; i += gridDim.x) {
            float m_val = 0;
            float v_val = 0;
            for (int j = threadIdx.x; j < channel; j += blockDim.x) {
                const int X_index = i * channel + j;
#if __CUDA_ARCH__ >= 350
                const float t = (float) __ldg(in + X_index) + (float) __ldg(inout + X_index);
#else
                const float t = (float) in[X_index] + (float) inout[X_index];
#endif
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

    template <typename T>
    __global__ void kernel_batchnorm_(const T *__restrict__ in,
                                      T *inout,
                                      const int batch_size,
                                      const int channel,
                                      const T *__restrict__ mean_in,
                                      const T *__restrict__ var_in,
                                      const T *__restrict__ beta,
                                      const T *__restrict__ gamma) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * channel) {
            return;
        }

        int batch_idx = idx / channel;
        int channel_idx = idx % channel;

#if __CUDA_ARCH__ >= 350
        float mean = (float) __ldg(mean_in + batch_idx);
        float var = (float) __ldg(var_in + batch_idx);
#else
        float mean = (float) mean_in[batch_idx];
        float var = (float) var_in[batch_idx];
#endif

        // 1 / sqrt(var)
        var = rsqrtf(var + 1e-12);

#if __CUDA_ARCH__ >= 350
        float _beta = (float) __ldg(beta + channel_idx);
        float _gamma = (float) __ldg(gamma + channel_idx);
        float _inout = (float) __ldg(inout + idx);
        float _in = (float) __ldg(in + idx);
#else
        float _beta = (float) beta[channel_idx];
        float _gamma = (float) gamma[channel_idx];
        float _inout = (float) inout[idx];
        float _in = (float) in[idx];
#endif
        inout[idx] = _beta + _gamma * var * (_inout + _in - mean);
    }

    template <typename T>
    __host__ void layer_norm_(const T *in,
                              T *inout,
                              const int batch_size,
                              const int channel,
                              T *mean_gpu,
                              T *var_gpu,
                              const T *beta,
                              const T *gamma,
                              void *stream) {
        kernel_momentum_cub<T> <<<batch_size, 128, 0, (cudaStream_t) stream>>> (in, inout, batch_size, channel, mean_gpu, var_gpu);

        const int all_blocks = (batch_size * channel + 127) / 128;
        kernel_batchnorm_<T> <<<all_blocks, 128, 0, (cudaStream_t) stream>>> (in, inout, batch_size, channel, mean_gpu, var_gpu, beta, gamma);
    }

    template
    __host__ void layer_norm_<float>(const float *in,
                                     float *inout,
                                     const int batch_size,
                                     const int channel,
                                     float *mean_gpu,
                                     float *var_gpu,
                                     const float *beta,
                                     const float *gamma,
                                     void *stream);

    template
    __host__ void layer_norm_<half>(const half *in,
                                    half *inout,
                                    const int batch_size,
                                    const int channel,
                                    half *mean_gpu,
                                    half *var_gpu,
                                    const half *beta,
                                    const half *gamma,
                                    void *stream);
}
