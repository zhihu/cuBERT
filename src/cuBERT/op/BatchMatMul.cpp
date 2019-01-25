//
// Created by 田露 on 2019/1/18.
//
#include <cuda_runtime.h>

#include "BatchMatMul.h"

namespace cuBERT {
    BatchMatMul::BatchMatMul(cublasHandle_t handle,
                             bool transpose_a, bool transpose_b,
                             size_t M, size_t N, size_t K,
                             size_t max_batch_size,
                             float alpha, float beta)
            : alpha(alpha), beta(beta) {
        this->handle = handle;
        this->transpose_a = transpose_a;
        this->transpose_b = transpose_b;
        this->M = M;
        this->N = N;
        this->K = K;

        cudaMalloc(&in_A_array_gpu, max_batch_size * sizeof(float *));
        cudaMalloc(&in_B_array_gpu, max_batch_size * sizeof(float *));
        cudaMalloc(&out_array_gpu, max_batch_size * sizeof(float *));
    }

    BatchMatMul::~BatchMatMul() {
        cudaFree(out_array_gpu);
        cudaFree(in_B_array_gpu);
        cudaFree(in_A_array_gpu);
    }

    void BatchMatMul::compute(size_t batch_size, const float *in_A_gpu, const float *in_B_gpu, float *out_gpu) {
        if (batch_size == 1) {
            cublasSgemm_v2(handle,
                           (cublasOperation_t) transpose_b, (cublasOperation_t) transpose_a,
                           N, M, K,
                           &alpha,
                           in_B_gpu, transpose_b ? K : N,
                           in_A_gpu, transpose_a ? M : K,
                           &beta,
                           out_gpu, N);
        } else {
            cublasSgemmStridedBatched(handle,
                                      (cublasOperation_t) transpose_b, (cublasOperation_t) transpose_a,
                                      N, M, K,
                                      &alpha,
                                      in_B_gpu, transpose_b ? K : N, N * K,
                                      in_A_gpu, transpose_a ? M : K, M * K,
                                      &beta,
                                      out_gpu, N, M * N,
                                      batch_size);
        }
    }
}
