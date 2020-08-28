#include <cfloat>
#include <cmath>
#include <stdexcept>

#include "cuBERT/common.h"
#include "Softmax.h"

namespace cuBERT {

#ifdef HAVE_MKL
    template<>
    void softmax_<float>(float *in,
                         float *out,
                         const int batch_size,
                         const int channel,
                         float *sum_gpu,
                         void *stream) {
#pragma omp parallel for
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            float max = -FLT_MAX;
            for (int i = batch_idx * channel; i < (batch_idx + 1) * channel; ++i) {
                if (in[i] > max) {
                    max = in[i];
                }
            }

            float sum = 0;
            for (int i = batch_idx * channel; i < (batch_idx + 1) * channel; ++i) {
                out[i] = expf(in[i] - max);
                sum += out[i];
            }

            for (int i = batch_idx * channel; i < (batch_idx + 1) * channel; ++i) {
                out[i] = out[i] / sum;
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
    void Softmax<T>::compute_(size_t batch_size, T *inout, void* stream) {
        softmax_<T>(inout, inout, batch_size, channel, sum_gpu, stream);
    }

    template <typename T>
    void Softmax<T>::compute_(size_t batch_size, T *in, T *out, void* stream) {
        softmax_<T>(in, out, batch_size, channel, sum_gpu, stream);
    }

    template class Softmax<float>;
#ifdef HAVE_CUDA
    template class Softmax<half>;
#endif
}
