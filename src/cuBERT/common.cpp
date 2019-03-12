#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cublas.h>
#endif
#ifdef HAVE_MKL
#include <mkl.h>
#endif

#if defined HAVE_CUDA && defined HAVE_MKL
#error only one and extact one blas implementation should be selected
#endif

#if !defined HAVE_CUDA && !defined HAVE_MKL
#error only one and extact one blas implementation should be selected
#endif

#include "./common.h"

namespace cuBERT {

#ifdef HAVE_CUDA

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (0 != err) {                                                     \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",   \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                             \
    } } while(0)

#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t err = call;                                          \
    if (0 != err) {                                                     \
        fprintf(stderr, "Cublas error in file '%s' in line %i.\n",      \
                __FILE__, __LINE__);                                    \
        exit(EXIT_FAILURE);                                             \
    } } while(0)

#endif

    void initialize(bool force_cpu) {}

    void finalize() {}

    int get_gpu_count() {
#ifdef HAVE_CUDA
            int count;
            CUDA_CHECK(cudaGetDeviceCount(&count));
            return count;
#else
            return 0;
#endif
    }

    void set_gpu(int device) {
#ifdef HAVE_CUDA
            CUDA_CHECK(cudaSetDevice(device));
#endif
    }

    void *cuda_stream_create() {
#ifdef HAVE_CUDA
            cudaStream_t ptr;
            CUDA_CHECK(cudaStreamCreate(&ptr));
            return ptr;
#else
            return nullptr;
#endif
    }

    void cuda_stream_destroy(void *stream) {
#ifdef HAVE_CUDA
            CUDA_CHECK(cudaStreamDestroy((cudaStream_t) stream));
#endif
    }

    void cuda_stream_synchronize(void *stream) {
#ifdef HAVE_CUDA
            CUDA_CHECK(cudaStreamSynchronize((cudaStream_t) stream));
#endif
    }

    void *malloc(size_t size) {
#ifdef HAVE_CUDA
            void *ptr;
            CUDA_CHECK(cudaMalloc(&ptr, size));
            return ptr;
#else
            return std::malloc(size);
#endif
    }

    void free(void *ptr) {
#ifdef HAVE_CUDA
            CUDA_CHECK(cudaFree(ptr));
#else
            std::free(ptr);
#endif
    }

    void memcpy(void *dst, const void *src, size_t n, int kind) {
#ifdef HAVE_CUDA
            CUDA_CHECK(cudaMemcpy(dst, src, n, (cudaMemcpyKind) kind));
#else
            std::memcpy(dst, src, n);
#endif
    }

    void memcpyAsync(void *dst, const void *src, size_t n, int kind, void *stream) {
#ifdef HAVE_CUDA
            CUDA_CHECK(cudaMemcpyAsync(dst, src, n, (cudaMemcpyKind) kind, (cudaStream_t) stream));
#else
            std::memcpy(dst, src, n);
#endif
    }

    void fill_n(float *dst, size_t n, float value) {
#ifdef HAVE_CUDA
            auto *dst_cpu = new float[n];
            std::fill_n(dst_cpu, n, value);
            cuBERT::memcpy(dst, dst_cpu, sizeof(float) * n, 1);
            delete[]dst_cpu;
#else
            std::fill_n(dst, n, value);
#endif
    }

    void *blas_create() {
#ifdef HAVE_CUDA
            cublasHandle_t ptr;
            CUBLAS_CHECK(cublasCreate_v2(&ptr));
            return ptr;
#else
            return nullptr;
#endif
    }

    void blas_destroy(void *handle) {
#ifdef HAVE_CUDA
            CUBLAS_CHECK(cublasDestroy_v2((cublasHandle_t) handle));
#endif
    }

    void *blas_get_stream(void *handle) {
#ifdef HAVE_CUDA
            cudaStream_t streamId;
            CUBLAS_CHECK(cublasGetStream_v2((cublasHandle_t) handle, &streamId));
            return streamId;
#else
            return nullptr;
#endif
    }

    void blas_set_stream(void *handle, void *streamId) {
#ifdef HAVE_CUDA
            CUBLAS_CHECK(cublasSetStream_v2((cublasHandle_t) handle, (cudaStream_t) streamId));
#endif
    }

    void blas_sgemm(void *handle,
                    const bool TransA, const bool TransB,
                    const int M, const int N, const int K,
                    const float alpha,
                    const float *A, const int lda,
                    const float *B, const int ldb,
                    const float beta,
                    float *C, const int ldc) {
#ifdef HAVE_CUDA
            CUBLAS_CHECK(cublasSgemm_v2((cublasHandle_t) handle,
                                        TransA ? CUBLAS_OP_T : CUBLAS_OP_N, TransB ? CUBLAS_OP_T : CUBLAS_OP_N,
                                        M, N, K,
                                        &alpha,
                                        A, lda,
                                        B, ldb,
                                        &beta,
                                        C, ldc));
            return;
#endif
#ifdef HAVE_MKL
            cblas_sgemm(CblasColMajor,
                        TransA ? CblasTrans : CblasNoTrans, TransB ? CblasTrans : CblasNoTrans,
                        M, N, K,
                        alpha,
                        A, lda,
                        B, ldb,
                        beta,
                        C, ldc);
            return;
#endif
    }

    void blas_sgemm_batch(void *handle,
                          const bool TransA, const bool TransB,
                          int m, int n, int k,
                          const float alpha,
                          const float **Aarray, int lda,
                          const float **Barray, int ldb,
                          const float beta,
                          float **Carray, int ldc,
                          int batchCount) {
#ifdef HAVE_CUDA
            CUBLAS_CHECK(cublasSgemmBatched((cublasHandle_t) handle,
                                            TransA ? CUBLAS_OP_T : CUBLAS_OP_N, TransB ? CUBLAS_OP_T : CUBLAS_OP_N,
                                            m, n, k,
                                            &alpha,
                                            Aarray, lda,
                                            Barray, ldb,
                                            &beta,
                                            Carray, ldc,
                                            batchCount));
            return;
#endif
#ifdef HAVE_MKL
            CBLAS_TRANSPOSE transA = TransA ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE transB = TransB ? CblasTrans : CblasNoTrans;
            cblas_sgemm_batch(CblasColMajor,
                              &transA, &transB,
                              &m, &n, &k,
                              &alpha,
                              Aarray, &lda,
                              Barray, &ldb,
                              &beta,
                              Carray, &ldc,
                              1, &batchCount);
            return;
#endif
    }

    void blas_sgemm_strided_batch(void *handle,
                                  const bool TransA, const bool TransB,
                                  int m, int n, int k,
                                  const float alpha,
                                  const float *A, int lda, long long int strideA,
                                  const float *B, int ldb, long long int strideB,
                                  const float beta,
                                  float *C, int ldc, long long int strideC,
                                  int batchCount) {
#ifdef HAVE_CUDA
            CUBLAS_CHECK(cublasSgemmStridedBatched((cublasHandle_t) handle,
                                                    TransA ? CUBLAS_OP_T : CUBLAS_OP_N, TransB ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                    m, n, k,
                                                    &alpha,
                                                    A, lda, strideA,
                                                    B, ldb, strideB,
                                                    &beta,
                                                    C, ldc, strideC,
                                                    batchCount));
            return;
#endif
#ifdef HAVE_MKL
            const float *A_Array[batchCount];
            const float *B_Array[batchCount];
            float *C_Array[batchCount];
            for (int i = 0; i < batchCount; ++i) {
                A_Array[i] = A + strideA * i;
                B_Array[i] = B + strideB * i;
                C_Array[i] = C + strideC * i;
            }

            CBLAS_TRANSPOSE transA = TransA ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE transB = TransB ? CblasTrans : CblasNoTrans;
            cblas_sgemm_batch(CblasColMajor,
                              &transA, &transB,
                              &m, &n, &k,
                              &alpha,
                              A_Array, &lda,
                              B_Array, &ldb,
                              &beta,
                              C_Array, &ldc,
                              1, &batchCount);
            return;
#endif
    }
}
