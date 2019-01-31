#include <cmath>
#include <omp.h>

#include "cuBERT/common.h"
#include "Softmax.h"

namespace cuBERT {

    Softmax::Softmax(size_t max_batch_size, size_t channel) {
        this->channel = channel;
        this->sum_gpu = static_cast<float *>(cuBERT::malloc(sizeof(float) * max_batch_size));
    }

    Softmax::~Softmax() {
        cuBERT::free(sum_gpu);
    }

    void Softmax::compute_(size_t batch_size, float *inout_gpu, void* stream) {
        if (cuBERT::gpu()) {
            softmax_(inout_gpu, batch_size, channel, sum_gpu, stream);
            return;
        }

#pragma omp parallel for
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            float sum = 0;
            for (int i = batch_idx * channel; i < (batch_idx + 1) * channel; ++i) {
                inout_gpu[i] = expf(inout_gpu[i]);
                sum += inout_gpu[i];
            }

            for (int i = batch_idx * channel; i < (batch_idx + 1) * channel; ++i) {
                inout_gpu[i] = inout_gpu[i] / sum;
            }
        }
    }
}
