//
// Created by 田露 on 2019/1/21.
//

#include "cuBERT/common.h"
#include "GELU.h"

namespace cuBERT {
    void GELU::compute_(size_t N, float *inout_gpu, cudaStream_t stream) {
        gelu_(inout_gpu, N, stream);
    }

    void GELU::compute_cpu_(size_t N, float *inout, cudaStream_t stream) {
        float *inout_gpu;
        cudaMalloc(&inout_gpu, sizeof(float) * N);

        CUDA_CHECK(cudaMemcpyAsync(inout_gpu, inout, sizeof(float) * N, cudaMemcpyHostToDevice, stream));
        compute_(N, inout_gpu, stream);

        // sync
        CUDA_CHECK(cudaMemcpy(inout, inout_gpu, sizeof(float) * N, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(inout_gpu));
    }
}