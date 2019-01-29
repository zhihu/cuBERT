#include <cmath>
#include <omp.h>

#include "cuBERT/common.h"
#include "Softmax.h"

namespace cuBERT {

    Softmax::Softmax(size_t max_batch_size, size_t channel) {
        this->channel = channel;
        CUDA_CHECK(cudaMalloc(&this->sum_gpu, sizeof(float) * max_batch_size));
    }

    Softmax::~Softmax() {
        CUDA_CHECK(cudaFree(sum_gpu));
    }

    void Softmax::compute_(size_t batch_size, float *inout_gpu, cudaStream_t stream) {
        softmax_(inout_gpu, batch_size, channel, sum_gpu, stream);
    }

    void Softmax::compute_cpu_(size_t batch_size, float *inout_cpu) {
#pragma omp parallel for
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            float sum = 0;
            for (int i = batch_idx * channel; i < (batch_idx + 1) * channel; ++i) {
                inout_cpu[i] = expf(inout_cpu[i]);
                sum += inout_cpu[i];
            }

            for (int i = batch_idx * channel; i < (batch_idx + 1) * channel; ++i) {
                inout_cpu[i] = inout_cpu[i] / sum;
            }
        }
    }
}
