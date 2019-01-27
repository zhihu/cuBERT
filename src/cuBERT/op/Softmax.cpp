#include "cuBERT/common.h"
#include "Softmax.h"

namespace cuBERT {
    const static float ZERO = 0;
    static const float ONE = 1;

    Softmax::Softmax(cudnnHandle_t handle, size_t channel, bool accurate) {
        this->handle = handle;
        this->channel = channel;
        this->accurate = accurate;

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    }

    Softmax::~Softmax() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    }

    void Softmax::compute_(size_t batch_size, float *inout_gpu) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channel, 1, 1));

        CUDNN_CHECK(cudnnSoftmaxForward(handle,
                                        accurate ? CUDNN_SOFTMAX_ACCURATE : CUDNN_SOFTMAX_FAST,
                                        CUDNN_SOFTMAX_MODE_INSTANCE,
                                        &ONE, desc, inout_gpu,
                                        &ZERO, desc, inout_gpu));
    }
}
