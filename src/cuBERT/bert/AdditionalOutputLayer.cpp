#include "cuBERT/common.h"
#include "AdditionalOutputLayer.h"

namespace cuBERT {
    const static float ZERO = 0;
    const static float ONE = 1;

    AdditionalOutputLayer::AdditionalOutputLayer(void* handle, size_t hidden_size, float *output_weights) {
        this->handle = handle;
        this->hidden_size = hidden_size;

        this->output_weights = static_cast<float *>(cuBERT::malloc(sizeof(float) * hidden_size));
        cuBERT::memcpy(this->output_weights, output_weights, sizeof(float) * hidden_size, 1);
    }

    AdditionalOutputLayer::~AdditionalOutputLayer() {
        cuBERT::free(output_weights);
    }

    void AdditionalOutputLayer::compute(size_t batch_size, float *in_gpu, float *out_gpu) {
        // TODO: can be simplified by sapy
        cuBERT::blas_sgemm(handle, false, false,
                           1, batch_size, hidden_size,
                           ONE,
                           output_weights, 1,
                           in_gpu, hidden_size,
                           ZERO,
                           out_gpu, 1);
    }
}
