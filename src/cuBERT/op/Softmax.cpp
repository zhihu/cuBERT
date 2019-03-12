#include <cmath>

#include "cuBERT/common.h"
#include "Softmax.h"

namespace cuBERT {

    template<>
    void softmax_<true>(float *inout,
                        const int batch_size,
                        const int channel,
                        float *sum_gpu,
                        void *stream) {
#pragma omp parallel for
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            float sum = 0;
            for (int i = batch_idx * channel; i < (batch_idx + 1) * channel; ++i) {
                inout[i] = expf(inout[i]);
                sum += inout[i];
            }

            for (int i = batch_idx * channel; i < (batch_idx + 1) * channel; ++i) {
                inout[i] = inout[i] / sum;
            }
        }
    }

    Softmax::Softmax(size_t max_batch_size, size_t channel) {
        this->channel = channel;
        this->sum_gpu = static_cast<float *>(cuBERT::malloc(sizeof(float) * max_batch_size));
    }

    Softmax::~Softmax() {
        cuBERT::free(sum_gpu);
    }

    void Softmax::compute_(size_t batch_size, float *inout_gpu, void* stream) {
        softmax_<!cuBERT::gpu()>(inout_gpu, batch_size, channel, sum_gpu, stream);
    }
}
