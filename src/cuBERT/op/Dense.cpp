#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mkl.h>

#include "cuBERT/common.h"
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

        this->kernel_cpu = new float[inputs * units];
        std::memcpy(kernel_cpu, kernel, inputs * units * sizeof(float));

        this->bias_cpu = new float[units * max_batch_size];
        for (int i = 0; i < max_batch_size; ++i) {
            std::memcpy(bias_cpu + units * i, bias, units * sizeof(float));
        }

        CUDA_CHECK(cudaMalloc(&kernel_gpu, inputs * units * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&bias_gpu, max_batch_size * units * sizeof(float)));

        CUBLAS_CHECK(cublasSetMatrix(units, inputs, sizeof(float), kernel, units, kernel_gpu, units));
        CUBLAS_CHECK(cublasSetMatrix(units, max_batch_size, sizeof(float), bias_cpu, units, bias_gpu, units));
    }

    Dense::~Dense() {
        CUDA_CHECK(cudaFree(bias_gpu));
        CUDA_CHECK(cudaFree(kernel_gpu));

        delete []bias_cpu;
        delete []kernel_cpu;
    }

    void Dense::compute(size_t batch_size, float *input_gpu, float *output_gpu) {
        _pre_compute(batch_size, output_gpu);
        _in_compute(batch_size, input_gpu, output_gpu);
    }

    void Dense::compute_cpu(size_t batch_size, float *input_cpu, float *output_cpu) {
        _pre_compute_cpu(batch_size, output_cpu);
        _in_compute_cpu(batch_size, input_cpu, output_cpu);
    }

    void Dense::_pre_compute(size_t batch_size, float *output_gpu) {
        cudaStream_t streamId = nullptr;
        CUBLAS_CHECK(cublasGetStream_v2(handle, &streamId));
        CUDA_CHECK(cudaMemcpyAsync(output_gpu, bias_gpu, units * batch_size * sizeof(float), cudaMemcpyDeviceToDevice, streamId));
    }

    void Dense::_in_compute(size_t batch_size, float *input_gpu, float *output_gpu) {
        CUBLAS_CHECK(cublasSgemm_v2(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                units, batch_size, inputs,
                &ONE,
                kernel_gpu, units,
                input_gpu, inputs,
                &ONE,
                output_gpu, units));
    }

    void Dense::_pre_compute_cpu(size_t batch_size, float *output_cpu) {
        std::memcpy(output_cpu, bias_cpu, units * batch_size * sizeof(float));
    }

    void Dense::_in_compute_cpu(size_t batch_size, float *input_cpu, float *output_cpu) {
        cblas_sgemm(CblasColMajor,
                    CblasNoTrans, CblasNoTrans,
                    units, batch_size, inputs,
                    ONE,
                    kernel_cpu, units,
                    input_cpu, inputs,
                    ONE,
                    output_cpu, units);
    }
}
