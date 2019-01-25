//
// Created by 田露 on 2019/1/15.
//
#include <algorithm>
#include <cuda_runtime.h>
#include "LayerNorm.h"

namespace cuBERT {
    LayerNorm::LayerNorm(size_t channels, float *beta, float *gamma) {
        this->channels = channels;

        cudaMalloc(&this->beta_gpu, channels * sizeof(float));
        cudaMalloc(&this->gamma_gpu, channels * sizeof(float));

        cudaMemcpy(beta_gpu, beta, channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gamma_gpu, gamma, channels * sizeof(float), cudaMemcpyHostToDevice);
    }

    LayerNorm::~LayerNorm() {
        cudaFree(gamma_gpu);
        cudaFree(beta_gpu);
    }

    void LayerNorm::compute_(size_t batch_size, float *inout_gpu, cudaStream_t stream) {
        layer_norm_(inout_gpu, batch_size, channels, beta_gpu, gamma_gpu, stream);
    }

    void LayerNorm::compute_(size_t batch_size, float *in_gpu, float *inout_gpu, cudaStream_t stream) {
        layer_norm_(in_gpu, inout_gpu, batch_size, channels, beta_gpu, gamma_gpu, stream);
    }

    void LayerNorm::compute_cpu_(size_t batch_size, float *inout, cudaStream_t stream) {
        float *inout_gpu;
        cudaMalloc(&inout_gpu, batch_size * channels * sizeof(float));
        cudaMemcpyAsync(inout_gpu, inout, batch_size * channels * sizeof(float), cudaMemcpyHostToDevice, stream);

        compute_(batch_size, inout_gpu, stream);

        // sync
        cudaMemcpy(inout, inout_gpu, batch_size * channels * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(inout_gpu);
    }

    void LayerNorm::compute_cpu_(size_t batch_size, float *in, float *inout, cudaStream_t stream) {
        float *in_gpu;
        float *inout_gpu;
        cudaMalloc(&in_gpu, batch_size * channels * sizeof(float));
        cudaMalloc(&inout_gpu, batch_size * channels * sizeof(float));
        cudaMemcpyAsync(in_gpu, in, batch_size * channels * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(inout_gpu, inout, batch_size * channels * sizeof(float), cudaMemcpyHostToDevice, stream);

        compute_(batch_size, in_gpu, inout_gpu, stream);

        // sync
        cudaMemcpy(inout, inout_gpu, batch_size * channels * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(inout_gpu);
        cudaFree(in_gpu);
    }
}
