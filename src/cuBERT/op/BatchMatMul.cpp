#include <cuda_runtime.h>

#include "cuBERT/common.h"
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
    }

    void BatchMatMul::compute(size_t batch_size, const float *in_A_gpu, const float *in_B_gpu, float *out_gpu) {
        if (batch_size == 1) {
            CUBLAS_CHECK(cublasSgemm_v2(handle,
                                        (cublasOperation_t) transpose_b, (cublasOperation_t) transpose_a,
                                        N, M, K,
                                        &alpha,
                                        in_B_gpu, transpose_b ? K : N,
                                        in_A_gpu, transpose_a ? M : K,
                                        &beta,
                                        out_gpu, N));
        } else {
            CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
                                                   (cublasOperation_t) transpose_b, (cublasOperation_t) transpose_a,
                                                   N, M, K,
                                                   &alpha,
                                                   in_B_gpu, transpose_b ? K : N, N * K,
                                                   in_A_gpu, transpose_a ? M : K, M * K,
                                                   &beta,
                                                   out_gpu, N, M * N,
                                                   batch_size));
        }
    }
}
