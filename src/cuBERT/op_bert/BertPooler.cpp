#include <cmath>
#include <stdexcept>

#include "cuBERT/common.h"
#include "BertPooler.h"

namespace cuBERT {
#ifdef HAVE_MKL
    template<>
    void tanh_<float>(float *inout, const int N, void *stream) {
#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            inout[i] = tanhf(inout[i]);
        }
    }

    template<>
    void reduce_mean_1<float>(const float *in, float *out,
                              const int batch_size, const int seq_length, const int hidden_size,
                              void *stream) {
#pragma omp parallel for
        for (int idx = 0; idx < batch_size * hidden_size; ++idx) {
            size_t batch_idx = idx / hidden_size;
            size_t channel_idx = idx % hidden_size;

            float sum = 0;
            for (int seq_idx = 0; seq_idx < seq_length; ++seq_idx) {
                sum += in[channel_idx + seq_idx * hidden_size + batch_idx * seq_length * hidden_size];
            }
            out[idx] = sum / seq_length;
        }
    }
#endif

    template <typename T>
    BertPooler<T>::BertPooler(void* handle,
                              size_t seq_length, size_t hidden_size,
                              T *kernel, T *bias,
                              size_t max_batch_size) {
        this->handle = handle;

        this->hidden_size = hidden_size;
        this->seq_length = seq_length;

        this->kernel = static_cast<T *>(cuBERT::malloc(sizeof(T) * hidden_size * hidden_size));
        cuBERT::memcpy(this->kernel, kernel, sizeof(T) * hidden_size * hidden_size, 1);

        this->bias = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * hidden_size));
        for (int i = 0; i < max_batch_size; ++i) {
            cuBERT::memcpy(this->bias + hidden_size * i, bias, hidden_size * sizeof(T), 1);
        }
    }

    template <typename T>
    BertPooler<T>::~BertPooler() {
        cuBERT::free(bias);
        cuBERT::free(kernel);
    }

    template <typename T>
    void BertPooler<T>::compute(size_t batch_size, T *input, T *output) {
        void* streamId = cuBERT::blas_get_stream(handle);

        cuBERT::memcpyAsync(output, bias, sizeof(T) * batch_size * hidden_size, 3, streamId);

        cuBERT::blas_gemm(handle, false, false,
                          hidden_size, batch_size, hidden_size,
                          1.f,
                          kernel, hidden_size,
                          input, hidden_size * seq_length,
                          1.f,
                          output, hidden_size);

        tanh_<T>(output, batch_size * hidden_size, streamId);
    }

    template class BertPooler<float>;
#ifdef HAVE_CUDA
    template class BertPooler<half>;
#endif

    template <typename T>
    MeanPooler<T>::MeanPooler(void *handle, size_t seq_length, size_t hidden_size) {
        this->handle = handle;

        this->hidden_size = hidden_size;
        this->seq_length = seq_length;
    }

    template <typename T>
    void MeanPooler<T>::compute(size_t batch_size, T *in, T *out) {
        void* streamId = cuBERT::blas_get_stream(handle);
        reduce_mean_1<T>(in, out, batch_size, seq_length, hidden_size, streamId);
    }

    template class MeanPooler<float>;
#ifdef HAVE_CUDA
    template class MeanPooler<half>;
#endif
}
