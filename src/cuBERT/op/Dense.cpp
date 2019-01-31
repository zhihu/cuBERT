#include "cuBERT/common.h"
#include "Dense.h"

namespace cuBERT {

    const static float ONE = 1;

    Dense::Dense(void* handle,
                 size_t inputs,
                 size_t units,
                 float *kernel,
                 float *bias,
                 size_t max_batch_size) {
        this->handle = handle;
        this->inputs = inputs;
        this->units = units;

        this->kernel = static_cast<float *>(cuBERT::malloc(sizeof(float) * inputs * units));
        cuBERT::memcpy(this->kernel, kernel, inputs * units * sizeof(float), 1);

        this->bias = static_cast<float *>(cuBERT::malloc(sizeof(float) * units * max_batch_size));
        for (int i = 0; i < max_batch_size; ++i) {
            cuBERT::memcpy(this->bias + units * i, bias, units * sizeof(float), 1);
        }
    }

    Dense::~Dense() {
        cuBERT::free(bias);
        cuBERT::free(kernel);
    }

    void Dense::compute(size_t batch_size, float *input, float *output) {
        _pre_compute(batch_size, output);
        _in_compute(batch_size, input, output);
    }

    void Dense::_pre_compute(size_t batch_size, float *output) {
        void* streamId = blas_get_stream(handle);
        cuBERT::memcpyAsync(output, bias, units * batch_size * sizeof(float), 3, streamId);
    }

    void Dense::_in_compute(size_t batch_size, float *input, float *output) {
        cuBERT::blas_sgemm(handle,
                           false, false,
                           units, batch_size, inputs,
                           ONE,
                           kernel, units,
                           input, inputs,
                           ONE,
                           output, units);
    }
}
