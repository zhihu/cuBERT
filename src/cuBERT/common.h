#ifndef CUBERT_COMMON_H
#define CUBERT_COMMON_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

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

#define CUDNN_CHECK(call) do {                                          \
    cudnnStatus_t err = call;                                           \
    if (CUDNN_STATUS_SUCCESS != err) {                                  \
        fprintf(stderr, "Cudnn error in file '%s' in line %i.\n",       \
                __FILE__, __LINE__);                                    \
        exit(EXIT_FAILURE);                                             \
    } } while(0)

}

#endif //CUBERT_COMMON_H
