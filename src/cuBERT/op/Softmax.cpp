#include "cuBERT/common.h"
#include "Softmax.h"

namespace cuBERT {

    Softmax::Softmax(size_t max_batch_size, size_t channel) {
        this->channel = channel;
        CUDA_CHECK(cudaMalloc(&this->sum_gpu, sizeof(float) * max_batch_size));
    }

    Softmax::~Softmax() {
        CUDA_CHECK(cudaFree(sum_gpu));
    }

    void Softmax::compute_(size_t batch_size, float *inout_gpu, cudaStream_t stream) {
        softmax_(inout_gpu, batch_size, channel, sum_gpu, stream);
    }
}
