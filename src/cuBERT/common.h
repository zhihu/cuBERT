#ifndef CUBERT_COMMON_H
#define CUBERT_COMMON_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <iostream>

namespace cuBERT {

#define CUDA_CHECK(error) { cuda_check((error), __FILE__, __LINE__); }
    inline void cuda_check(cudaError_t error, const char *file, int line, bool abort=true) {
        if (error != cudaSuccess) {
            std::cerr << "Error at: "
                      << file << ":"
                      << line << ": "
                      << cudaGetErrorString(error) << std::endl;
            if (abort) {
                exit(error);
            }
        }
    }

#define CUBLAS_CHECK(error) { cublas_check((error), __FILE__, __LINE__); }
    inline void cublas_check(cublasStatus_t error, const char *file, int line, bool abort=true) {
        if (error != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Error at: "
                      << file << ":"
                      << line << ": "
                      << error << std::endl;
            if (abort) {
                exit(error);
            }
        }
    }

#define CUDNN_CHECK(error) { cudnn_check((error), __FILE__, __LINE__); }
    inline void cudnn_check(cudnnStatus_t error, const char *file, int line, bool abort=true) {
        if (error != CUDNN_STATUS_SUCCESS) {
            std::cerr << "Error at: "
                      << file << ":"
                      << line << ": "
                      << error << std::endl;
            if (abort) {
                exit(error);
            }
        }
    }
}

#endif //CUBERT_COMMON_H
