#include <algorithm>
#include <cuda_runtime.h>

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
    }

    LayerNorm::~LayerNorm() {
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

    void LayerNorm::compute_cpu_(size_t batch_size, float *inout, cudaStream_t stream) {
        float *inout_gpu;
        CUDA_CHECK(cudaMalloc(&inout_gpu, batch_size * channels * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(inout_gpu, inout, batch_size * channels * sizeof(float), cudaMemcpyHostToDevice, stream));

        compute_(batch_size, inout_gpu, stream);

        // sync
        CUDA_CHECK(cudaMemcpy(inout, inout_gpu, batch_size * channels * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(inout_gpu));
    }

    void LayerNorm::compute_cpu_(size_t batch_size, float *in, float *inout, cudaStream_t stream) {
        float *in_gpu;
        float *inout_gpu;
        CUDA_CHECK(cudaMalloc(&in_gpu, batch_size * channels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&inout_gpu, batch_size * channels * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(in_gpu, in, batch_size * channels * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(inout_gpu, inout, batch_size * channels * sizeof(float), cudaMemcpyHostToDevice, stream));

        compute_(batch_size, in_gpu, inout_gpu, stream);

        // sync
        CUDA_CHECK(cudaMemcpy(inout, inout_gpu, batch_size * channels * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(inout_gpu));
        CUDA_CHECK(cudaFree(in_gpu));
    }
}
