#include "cuBERT/common.h"
#include "Dense.h"

namespace cuBERT {

    template<typename T>
    Dense<T>::Dense(void* handle,
                 size_t inputs,
                 size_t units,
                 T *kernel,
                 T *bias,
                 size_t max_batch_size) {
        this->handle = handle;
        this->inputs = inputs;
        this->units = units;

        this->kernel = static_cast<T *>(cuBERT::malloc(sizeof(T) * inputs * units));
        cuBERT::memcpy(this->kernel, kernel, inputs * units * sizeof(T), 1);

        this->bias = static_cast<T *>(cuBERT::malloc(sizeof(T) * units * max_batch_size));
        for (int i = 0; i < max_batch_size; ++i) {
            cuBERT::memcpy(this->bias + units * i, bias, units * sizeof(T), 1);
        }
    }

    template<typename T>
    Dense<T>::~Dense() {
        cuBERT::free(bias);
        cuBERT::free(kernel);
    }

    template<typename T>
    void Dense<T>::compute(size_t batch_size, T *input, T *output) {
        _pre_compute(batch_size, output);
        _in_compute(batch_size, input, output);
    }

    template<typename T>
    void Dense<T>::_pre_compute(size_t batch_size, T *output) {
        void* streamId = blas_get_stream(handle);
        cuBERT::memcpyAsync(output, bias, units * batch_size * sizeof(T), 3, streamId);
    }

    template<typename T>
    void Dense<T>::_in_compute(size_t batch_size, T *input, T *output) {
        cuBERT::blas_gemm(handle,
                           false, false,
                           units, batch_size, inputs,
                           1.f,
                           kernel, units,
                           input, inputs,
                           1.f,
                           output, units);
    }

    template class Dense<float>;
#ifdef HAVE_CUDA
    template class Dense<half>;
#endif
}
