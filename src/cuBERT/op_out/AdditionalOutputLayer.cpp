#include "cuBERT/common.h"
#include "AdditionalOutputLayer.h"

namespace cuBERT {

    template <typename T>
    AdditionalOutputLayer<T>::AdditionalOutputLayer(void* handle, size_t hidden_size, T *output_weights) {
        this->handle = handle;
        this->hidden_size = hidden_size;

        this->output_weights = static_cast<T *>(cuBERT::malloc(sizeof(T) * hidden_size));
        cuBERT::memcpy(this->output_weights, output_weights, sizeof(T) * hidden_size, 1);
    }

    template <typename T>
    AdditionalOutputLayer<T>::~AdditionalOutputLayer() {
        cuBERT::free(output_weights);
    }

    template <typename T>
    void AdditionalOutputLayer<T>::compute(size_t batch_size, T *in_gpu, T *out_gpu) {
        // TODO: can be simplified by sapy
        cuBERT::blas_gemm(handle, false, false,
                           1, batch_size, hidden_size,
                           1.f,
                           output_weights, 1,
                           in_gpu, hidden_size,
                           0.f,
                           out_gpu, 1);
    }

    template class AdditionalOutputLayer<float>;
#ifdef HAVE_CUDA
    template class AdditionalOutputLayer<half>;
#endif
}
