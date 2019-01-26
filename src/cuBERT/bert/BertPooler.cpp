#include <cuda_runtime.h>
#include <cstring>

#include "cuBERT/common.h"
#include "BertPooler.h"

namespace cuBERT {
    const static float ONE = 1;

    BertPooler::BertPooler(cublasHandle_t handle,
                           size_t seq_length, size_t hidden_size,
                           float *kernel, float *bias,
                           size_t max_batch_size) {
        this->handle = handle;

        this->hidden_size = hidden_size;
        this->seq_length = seq_length;

        CUDA_CHECK(cudaMalloc(&this->kernel_gpu, sizeof(float) * hidden_size * hidden_size));
        CUDA_CHECK(cudaMemcpy(kernel_gpu, kernel, sizeof(float) * hidden_size * hidden_size, cudaMemcpyHostToDevice));

        auto *bias_broadcast = new float[hidden_size * max_batch_size];
        for (int i = 0; i < max_batch_size; ++i) {
            std::memcpy(bias_broadcast + hidden_size * i, bias, hidden_size * sizeof(float));
        }
        CUDA_CHECK(cudaMalloc(&this->bias_gpu, sizeof(float) * max_batch_size * hidden_size));
        CUDA_CHECK(cudaMemcpy(bias_gpu, bias_broadcast, sizeof(float) * max_batch_size * hidden_size, cudaMemcpyHostToDevice));
        delete[] bias_broadcast;
    }

    BertPooler::~BertPooler() {
        CUDA_CHECK(cudaFree(bias_gpu));
        CUDA_CHECK(cudaFree(kernel_gpu));
    }

    void BertPooler::compute(size_t batch_size, float *input_gpu, float *output_gpu) {
        cudaStream_t streamId = nullptr;
        CUBLAS_CHECK(cublasGetStream_v2(handle, &streamId));

        CUDA_CHECK(cudaMemcpyAsync(output_gpu, bias_gpu, sizeof(float) * batch_size * hidden_size, cudaMemcpyDeviceToDevice, streamId));

        CUBLAS_CHECK(cublasSgemm_v2(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    hidden_size, batch_size, hidden_size,
                                    &ONE,
                                    kernel_gpu, hidden_size,
                                    input_gpu, hidden_size * seq_length,
                                    &ONE,
                                    output_gpu, hidden_size));

        tanh_(output_gpu, batch_size * hidden_size, streamId);
    }
}
