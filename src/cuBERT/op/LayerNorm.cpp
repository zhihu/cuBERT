#include <algorithm>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <omp.h>

#include "cuBERT/common.h"
#include "LayerNorm.h"

namespace cuBERT {
    LayerNorm::LayerNorm(size_t max_batch_size, size_t channels, float *beta, float *gamma) {
        this->channels = channels;

        CUDA_CHECK(cudaMalloc(&this->beta_gpu, channels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&this->gamma_gpu, channels * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(beta_gpu, beta, channels * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gamma_gpu, gamma, channels * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&this->mean_gpu, max_batch_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&this->var_gpu, max_batch_size * sizeof(float)));

        this->beta_cpu = new float[channels];
        this->gamma_cpu = new float[channels];
        std::memcpy(beta_cpu, beta, channels * sizeof(float));
        std::memcpy(gamma_cpu, gamma, channels * sizeof(float));
    }

    LayerNorm::~LayerNorm() {
        delete[] gamma_cpu;
        delete[] beta_cpu;

        CUDA_CHECK(cudaFree(var_gpu));
        CUDA_CHECK(cudaFree(mean_gpu));

        CUDA_CHECK(cudaFree(gamma_gpu));
        CUDA_CHECK(cudaFree(beta_gpu));
    }

    void LayerNorm::compute_(size_t batch_size, float *inout_gpu, cudaStream_t stream) {
        layer_norm_(inout_gpu, batch_size, channels, beta_gpu, gamma_gpu, stream);
    }

    void LayerNorm::compute_(size_t batch_size, float *in_gpu, float *inout_gpu, cudaStream_t stream) {
        layer_norm_(in_gpu, inout_gpu,
                    batch_size, channels,
                    mean_gpu, var_gpu,
                    beta_gpu, gamma_gpu, stream);
    }

    void LayerNorm::compute_cpu_(size_t batch_size, float *inout) {
#pragma omp parallel for
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            float mean = 0;
            float var = 0;
#pragma unroll
            for (int i = batch_idx * channels; i < (batch_idx + 1) * channels; ++i) {
                float t = inout[i];
                mean += t;
                var += t * t;
            }
            mean = mean / channels;
            var = var / channels - mean * mean;

            // 1 / sqrt(var)
            var = 1.f / sqrtf(var + 1e-12f);

#pragma unroll
            for (int i = 0; i < channels; ++i) {
                int j = batch_idx * channels + i;
                inout[j] = beta_cpu[i] + gamma_cpu[i] * var * (inout[j] - mean);
            }
        }
    }

    void LayerNorm::compute_cpu_(size_t batch_size, float *in, float *inout) {
#pragma omp parallel for
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            float mean = 0;
            float var = 0;
#pragma unroll
            for (int i = batch_idx * channels; i < (batch_idx + 1) * channels; ++i) {
                float t = inout[i] + in[i];
                mean += t;
                var += t * t;
            }
            mean = mean / channels;
            var = var / channels - mean * mean;

            // 1 / sqrt(var)
            var = 1.f / sqrtf(var + 1e-12f);

#pragma unroll
            for (int i = 0; i < channels; ++i) {
                int j = batch_idx * channels + i;
                inout[j] = beta_cpu[i] + gamma_cpu[i] * var * (inout[j] + in[j] - mean);
            }
        }
    }
}
