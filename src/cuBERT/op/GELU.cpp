//
// Created by 田露 on 2019/1/21.
//

#include "GELU.h"

namespace cuBERT {
    void GELU::compute_(size_t N, float *inout_gpu, cudaStream_t stream) {
        gelu_(inout_gpu, N, stream);
    }

    void GELU::compute_cpu_(size_t N, float *inout, cudaStream_t stream) {
        float *inout_gpu;
        cudaMalloc(&inout_gpu, sizeof(float) * N);

        cudaMemcpyAsync(inout_gpu, inout, sizeof(float) * N, cudaMemcpyHostToDevice, stream);
        compute_(N, inout_gpu, stream);

        // sync
        cudaMemcpy(inout, inout_gpu, sizeof(float) * N, cudaMemcpyDeviceToHost);
        cudaFree(inout_gpu);
    }
}