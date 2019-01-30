#ifndef CUBERT_COMMON_H
#define CUBERT_COMMON_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mkl.h>

#include <cstdlib>
#include <cstdio>

namespace cuBERT {

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (cudaSuccess != err) {                                           \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",   \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                             \
    } } while(0)

#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t err = call;                                          \
    if (CUBLAS_STATUS_SUCCESS != err) {                                 \
        fprintf(stderr, "Cublas error in file '%s' in line %i.\n",      \
                __FILE__, __LINE__);                                    \
        exit(EXIT_FAILURE);                                             \
    } } while(0)

    inline void cblas_sgemm_strided_batch(CBLAS_LAYOUT Layout,
                                          CBLAS_TRANSPOSE transa,
                                          CBLAS_TRANSPOSE transb,
                                          int m,
                                          int n,
                                          int k,
                                          const float alpha,
                                          const float *A,
                                          int lda,
                                          long long int strideA,
                                          const float *B,
                                          int ldb,
                                          long long int strideB,
                                          const float beta,
                                          float *C,
                                          int ldc,
                                          long long int strideC,
                                          int batchCount) {
        const float *A_Array[batchCount];
        const float *B_Array[batchCount];
        float *C_Array[batchCount];
        for (int i = 0; i < batchCount; ++i) {
            A_Array[i] = A + strideA * i;
            B_Array[i] = B + strideB * i;
            C_Array[i] = C + strideC * i;
        }

        cblas_sgemm_batch(Layout,
                          &transa, &transb,
                          &m, &n, &k,
                          &alpha,
                          A_Array, &lda,
                          B_Array, &ldb,
                          &beta,
                          C_Array, &ldc,
                          1, &batchCount);
    }
}

#endif //CUBERT_COMMON_H
