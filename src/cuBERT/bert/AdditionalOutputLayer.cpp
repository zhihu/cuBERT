#include <cuda_runtime.h>
#include <mkl.h>
#include <cstring>

#include "cuBERT/common.h"
#include "AdditionalOutputLayer.h"

namespace cuBERT {
    const static float ZERO = 0;
    const static float ONE = 1;

    AdditionalOutputLayer::AdditionalOutputLayer(cublasHandle_t handle, size_t hidden_size, float *output_weights) {
        this->handle = handle;
        this->hidden_size = hidden_size;

        CUDA_CHECK(cudaMalloc(&this->output_weights_gpu, sizeof(float) * hidden_size));
        CUDA_CHECK(cudaMemcpy(output_weights_gpu, output_weights, sizeof(float) * hidden_size, cudaMemcpyHostToDevice));

        this->output_weights_cpu = new float[hidden_size];
        std::memcpy(output_weights_cpu, output_weights, sizeof(float) * hidden_size);
    }

    AdditionalOutputLayer::~AdditionalOutputLayer() {
        delete []output_weights_cpu;

        CUDA_CHECK(cudaFree(output_weights_gpu));
    }

    void AdditionalOutputLayer::compute(size_t batch_size, float *in_gpu, float *out_gpu) {
        // TODO: can be simplified by sapy
        CUBLAS_CHECK(cublasSgemm_v2(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    1, batch_size, hidden_size,
                                    &ONE,
                                    output_weights_gpu, 1,
                                    in_gpu, hidden_size,
                                    &ZERO,
                                    out_gpu, 1));
    }

    void AdditionalOutputLayer::compute_cpu(size_t batch_size, float *in_cpu, float *out_cpu) {
        cblas_sgemm(CblasColMajor,
                    CblasNoTrans, CblasNoTrans,
                    1, batch_size, hidden_size,
                    ONE,
                    output_weights_cpu, 1,
                    in_cpu, hidden_size,
                    ZERO,
                    out_cpu, 1);
    }
}
