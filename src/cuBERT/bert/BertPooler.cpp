#include <mkl.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>

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

        this->kernel_cpu = new float[hidden_size * hidden_size];
        std::memcpy(kernel_cpu, kernel, sizeof(float) * hidden_size * hidden_size);

        this->bias_cpu = new float[hidden_size * max_batch_size];
        for (int i = 0; i < max_batch_size; ++i) {
            std::memcpy(bias_cpu + hidden_size * i, bias, hidden_size * sizeof(float));
        }

        CUDA_CHECK(cudaMalloc(&this->kernel_gpu, sizeof(float) * hidden_size * hidden_size));
        CUDA_CHECK(cudaMemcpy(kernel_gpu, kernel, sizeof(float) * hidden_size * hidden_size, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&this->bias_gpu, sizeof(float) * max_batch_size * hidden_size));
        CUDA_CHECK(cudaMemcpy(bias_gpu, bias_cpu, sizeof(float) * max_batch_size * hidden_size, cudaMemcpyHostToDevice));
    }

    BertPooler::~BertPooler() {
        delete []bias_cpu;
        delete []kernel_cpu;

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

    void BertPooler::compute_cpu(size_t batch_size, float *input_cpu, float *output_cpu) {
        std::memcpy(output_cpu, bias_cpu, sizeof(float) * batch_size * hidden_size);

        cblas_sgemm(CblasColMajor,
                    CblasNoTrans, CblasNoTrans,
                    hidden_size, batch_size, hidden_size,
                    ONE,
                    kernel_cpu, hidden_size,
                    input_cpu, hidden_size * seq_length,
                    ONE,
                    output_cpu, hidden_size);

#pragma omp parallel for
        for (int i = 0; i < batch_size * hidden_size; ++i) {
            output_cpu[i] = tanhf(output_cpu[i]);
        }
    }
}
