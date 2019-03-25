#include <cmath>
#include <stdexcept>

#include "cuBERT/common.h"
#include "Softmax.h"

namespace cuBERT {

#ifdef HAVE_MKL
    template<>
    void softmax_<float>(float *inout,
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
#endif

    template <typename T>
    Softmax<T>::Softmax(size_t max_batch_size, size_t channel) {
        this->channel = channel;
        this->sum_gpu = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size));
    }

    template <typename T>
    Softmax<T>::~Softmax() {
        cuBERT::free(sum_gpu);
    }

    template <typename T>
    void Softmax<T>::compute_(size_t batch_size, T *inout_gpu, void* stream) {
        softmax_<T>(inout_gpu, batch_size, channel, sum_gpu, stream);
    }

    template class Softmax<float>;
#ifdef HAVE_CUDA
    template class Softmax<half>;
#endif
}
