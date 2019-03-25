#include <cmath>
#include <stdexcept>

#include "cuBERT/common.h"
#include "LayerNorm.h"

namespace cuBERT {

#ifdef HAVE_MKL
    template<>
    void layer_norm_<float>(const float *in,
                            float *inout,
                            const int batch_size,
                            const int channels,
                            float *mean_gpu,
                            float *var_gpu,
                            const float *beta,
                            const float *gamma,
                            void *stream) {
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
                inout[j] = beta[i] + gamma[i] * var * (inout[j] + in[j] - mean);
            }
        }
    }
#endif

    template <typename T>
    LayerNorm<T>::LayerNorm(size_t max_batch_size, size_t channels, T *beta, T *gamma) {
        this->channels = channels;

        this->mean_gpu = static_cast<T *>(cuBERT::malloc(max_batch_size * sizeof(T)));
        this->var_gpu = static_cast<T *>(cuBERT::malloc(max_batch_size * sizeof(T)));

        this->beta = static_cast<T *>(cuBERT::malloc(channels * sizeof(T)));
        this->gamma = static_cast<T *>(cuBERT::malloc(channels * sizeof(T)));
        cuBERT::memcpy(this->beta, beta, channels * sizeof(T), 1);
        cuBERT::memcpy(this->gamma, gamma, channels * sizeof(T), 1);
    }

    template <typename T>
    LayerNorm<T>::~LayerNorm() {
        cuBERT::free(gamma);
        cuBERT::free(beta);

        cuBERT::free(var_gpu);
        cuBERT::free(mean_gpu);
    }

    template <typename T>
    void LayerNorm<T>::compute_(size_t batch_size, T *in, T *inout, void* stream) {
        layer_norm_<T>(in, inout, batch_size, channels, mean_gpu, var_gpu, beta, gamma, stream);
    }

    template class LayerNorm<float>;
#ifdef HAVE_CUDA
    template class LayerNorm<half>;
#endif
}
