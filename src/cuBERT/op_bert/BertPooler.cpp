#include <omp.h>
#include <cmath>
#include <stdexcept>

#include "cuBERT/common.h"
#include "BertPooler.h"

namespace cuBERT {
    const static float ONE = 1;

    BertPooler::BertPooler(void* handle,
                           size_t seq_length, size_t hidden_size,
                           float *kernel, float *bias,
                           size_t max_batch_size) {
        this->handle = handle;

        this->hidden_size = hidden_size;
        this->seq_length = seq_length;

        this->kernel = static_cast<float *>(cuBERT::malloc(sizeof(float) * hidden_size * hidden_size));
        cuBERT::memcpy(this->kernel, kernel, sizeof(float) * hidden_size * hidden_size, 1);

        this->bias = static_cast<float *>(cuBERT::malloc(sizeof(float) * max_batch_size * hidden_size));
        for (int i = 0; i < max_batch_size; ++i) {
            cuBERT::memcpy(this->bias + hidden_size * i, bias, hidden_size * sizeof(float), 1);
        }
    }

    BertPooler::~BertPooler() {
        cuBERT::free(bias);
        cuBERT::free(kernel);
    }

    void BertPooler::compute(size_t batch_size, float *input, float *output) {
        void* streamId = cuBERT::blas_get_stream(handle);

        cuBERT::memcpyAsync(output, bias, sizeof(float) * batch_size * hidden_size, 3, streamId);

        cuBERT::blas_sgemm(handle, false, false,
                           hidden_size, batch_size, hidden_size,
                           ONE,
                           kernel, hidden_size,
                           input, hidden_size * seq_length,
                           ONE,
                           output, hidden_size);

        if (cuBERT::gpu()) {
#ifdef HAVE_CUDA
            tanh_(output, batch_size * hidden_size, streamId);
#else
            throw std::invalid_argument("Compile without CUDA, but run with GPU.");
#endif
        } else {
#pragma omp parallel for
            for (int i = 0; i < batch_size * hidden_size; ++i) {
                output[i] = tanhf(output[i]);
            }
        }
    }
}
