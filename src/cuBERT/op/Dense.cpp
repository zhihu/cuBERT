#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "Dense.h"

namespace cuBERT {

    const static float ONE = 1;

    Dense::Dense(cublasHandle_t handle,
                 size_t inputs,
                 size_t units,
                 float *kernel,
                 float *bias,
                 size_t max_batch_size) {
        this->handle = handle;
        this->inputs = inputs;
        this->units = units;

        auto *bias_broadcast = new float[units * max_batch_size];
        for (int i = 0; i < max_batch_size; ++i) {
            std::memcpy(bias_broadcast + units * i, bias, units * sizeof(float));
        }

        cudaMalloc(&kernel_gpu, inputs * units * sizeof(float));
        cudaMalloc(&bias_gpu, max_batch_size * units * sizeof(float));

        cublasSetMatrix(units, inputs, sizeof(float), kernel, units, kernel_gpu, units);
        cublasSetMatrix(units, max_batch_size, sizeof(float), bias_broadcast, units, bias_gpu, units);

        delete[] bias_broadcast;
    }

    Dense::~Dense() {
        cudaFree(bias_gpu);
        cudaFree(kernel_gpu);
    }

    void Dense::compute(size_t batch_size, float *input_gpu, float *output_gpu) {
        cudaStream_t streamId = nullptr;
        cublasGetStream_v2(handle, &streamId);

        cudaMemcpyAsync(output_gpu, bias_gpu, units * batch_size * sizeof(float), cudaMemcpyDeviceToDevice, streamId);

        cublasSgemm_v2(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                units, batch_size, inputs,
                &ONE,
                kernel_gpu, units,
                input_gpu, inputs,
                &ONE,
                output_gpu, units);
    }

    void Dense::compute_cpu(size_t batch_size, float *input_cpu, float *output_cpu) {
        cudaStream_t streamId = nullptr;
        cublasGetStream_v2(handle, &streamId);

        float *input_gpu;
        cudaMalloc(&input_gpu, batch_size * inputs * sizeof(float));
        cublasSetMatrixAsync(inputs, batch_size, sizeof(float), input_cpu, inputs, input_gpu, inputs, streamId);

        float *output_gpu;
        cudaMalloc(&output_gpu, batch_size * units * sizeof(float));

        compute(batch_size, input_gpu, output_gpu);

        // sync
        cublasGetMatrix(units, batch_size, sizeof(float), output_gpu, units, output_cpu, units);

        cudaFree(output_gpu);
        cudaFree(input_gpu);
    }
}
