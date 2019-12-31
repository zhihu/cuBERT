#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>
#ifdef HAVE_CUDA
#include <cuda.h>
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

#ifdef HAVE_CUDA
#if CUDA_VERSION == 9000
extern "C" {
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx  (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const void *alpha, /* host or device pointer */  
                                                      const void *const Aarray[], 
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const void *const Barray[],
                                                      cudaDataType Btype,
                                                      int ldb, 
                                                      const void *beta, /* host or device pointer */  
                                                      void *const Carray[],
                                                      cudaDataType Ctype,
                                                      int ldc,
                                                      int batchCount,
                                                      cudaDataType computeType,
                                                      cublasGemmAlgo_t algo); 
 
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx (cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb, 
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const void *alpha,  /* host or device pointer */
                                                                 const void *A,
                                                                 cudaDataType Atype,
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const void *B,
                                                                 cudaDataType Btype,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const void *beta,   /* host or device pointer */
                                                                 void *C,
                                                                 cudaDataType Ctype,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount,
                                                                 cudaDataType computeType,
                                                                 cublasGemmAlgo_t algo);
}
#endif
#endif

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

    void initialize() {}

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

    void gpu_info(int device) {
#ifdef HAVE_CUDA
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        fprintf(stderr, "Found device %d with properties:\n", device);
        fprintf(stderr, "  name: %s major: %d minor: %d memoryClockRate(kHz): %d\n", prop.name, prop.major, prop.minor, prop.memoryClockRate);
        fprintf(stderr, "  pciBusID: %d\n", prop.pciBusID);
        fprintf(stderr, "  totalMemory: %zu\n", prop.totalGlobalMem);
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
        cudaError_t err = cudaMalloc(&ptr, size);
        if (cudaSuccess == err) {
            return ptr;
        }

        size_t free, total;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        fprintf(stderr, "CUDA %s. Tried to allocate %zu (%zu total; %zu free)\n",
                cudaGetErrorString(err), size, total, free);
        exit(EXIT_FAILURE);
#else
        return mkl_malloc(size, 64);
#endif
    }

    void free(void *ptr) {
#ifdef HAVE_CUDA
            CUDA_CHECK(cudaFree(ptr));
#else
            mkl_free(ptr);
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

    template<>
    void blas_gemm<float>(void *handle,
                          const bool TransA, const bool TransB,
                          const int M, const int N, const int K,
                          const float alpha,
                          const float *A, const int lda,
                          const float *B, const int ldb,
                          const float beta,
                          float *C, const int ldc,
                          int algo) {
#ifdef HAVE_CUDA
            CUBLAS_CHECK(cublasGemmEx((cublasHandle_t) handle,
                                      TransA ? CUBLAS_OP_T : CUBLAS_OP_N, TransB ? CUBLAS_OP_T : CUBLAS_OP_N,
                                      M, N, K,
                                      &alpha,
                                      A, CUDA_R_32F, lda,
                                      B, CUDA_R_32F, ldb,
                                      &beta,
                                      C, CUDA_R_32F, ldc,
                                      CUDA_R_32F,
                                      (cublasGemmAlgo_t) algo));
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

    template<>
    void blas_gemm_batch<float>(void *handle,
                                const bool TransA, const bool TransB,
                                int m, int n, int k,
                                const float alpha,
                                const float **Aarray, int lda,
                                const float **Barray, int ldb,
                                const float beta,
                                float **Carray, int ldc,
                                int batchCount, 
                                int algo) {
#ifdef HAVE_CUDA
            CUBLAS_CHECK(cublasGemmBatchedEx((cublasHandle_t) handle,
                                             TransA ? CUBLAS_OP_T : CUBLAS_OP_N, TransB ? CUBLAS_OP_T : CUBLAS_OP_N,
                                             m, n, k,
                                             &alpha,
                                             (const void **) Aarray, CUDA_R_32F, lda,
                                             (const void **) Barray, CUDA_R_32F, ldb,
                                             &beta,
                                             (void **) Carray, CUDA_R_32F, ldc,
                                             batchCount,
                                             CUDA_R_32F,
                                             (cublasGemmAlgo_t) algo));
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

    template<>
    void blas_gemm_strided_batch<float>(void *handle,
                                        const bool TransA, const bool TransB,
                                        int m, int n, int k,
                                        const float alpha,
                                        const float *A, int lda, long long int strideA,
                                        const float *B, int ldb, long long int strideB,
                                        const float beta,
                                        float *C, int ldc, long long int strideC,
                                        int batchCount,
                                        int algo) {
#ifdef HAVE_CUDA
            CUBLAS_CHECK(cublasGemmStridedBatchedEx((cublasHandle_t) handle,
                                                    TransA ? CUBLAS_OP_T : CUBLAS_OP_N, TransB ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                    m, n, k,
                                                    &alpha,
                                                    A, CUDA_R_32F, lda, strideA,
                                                    B, CUDA_R_32F, ldb, strideB,
                                                    &beta,
                                                    C, CUDA_R_32F, ldc, strideC,
                                                    batchCount,
                                                    CUDA_R_32F,
                                                    (cublasGemmAlgo_t) algo));
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

#ifdef HAVE_CUDA
    template<>
    void blas_gemm<half>(void *handle,
                         const bool TransA, const bool TransB,
                         const int M, const int N, const int K,
                         const float alpha,
                         const half *A, const int lda,
                         const half *B, const int ldb,
                         const float beta,
                         half *C, const int ldc,
                         int algo) {
        CUBLAS_CHECK(cublasGemmEx((cublasHandle_t) handle,
                                   TransA ? CUBLAS_OP_T : CUBLAS_OP_N, TransB ? CUBLAS_OP_T : CUBLAS_OP_N,
                                   M, N, K,
                                   &alpha,
                                   A, CUDA_R_16F, lda,
                                   B, CUDA_R_16F, ldb,
                                   &beta,
                                   C, CUDA_R_16F, ldc,
                                   CUDA_R_32F,
                                   (cublasGemmAlgo_t) algo));
    }

    template<>
    void blas_gemm_batch<half>(void *handle,
                               const bool TransA, const bool TransB,
                               int m, int n, int k,
                               const float alpha,
                               const half **Aarray, int lda,
                               const half **Barray, int ldb,
                               const float beta,
                               half **Carray, int ldc,
                               int batchCount, 
                               int algo) {
        CUBLAS_CHECK(cublasGemmBatchedEx((cublasHandle_t) handle,
                                         TransA ? CUBLAS_OP_T : CUBLAS_OP_N, TransB ? CUBLAS_OP_T : CUBLAS_OP_N,
                                         m, n, k,
                                         &alpha,
                                         (const void **) Aarray, CUDA_R_16F, lda,
                                         (const void **) Barray, CUDA_R_16F, ldb,
                                         &beta,
                                         (void **) Carray, CUDA_R_16F, ldc,
                                         batchCount,
                                         CUDA_R_32F,
                                         (cublasGemmAlgo_t) algo));
    }

    template<>
    void blas_gemm_strided_batch<half>(void *handle,
                                       const bool TransA, const bool TransB,
                                       int m, int n, int k,
                                       const float alpha,
                                       const half *A, int lda, long long int strideA,
                                       const half *B, int ldb, long long int strideB,
                                       const float beta,
                                       half *C, int ldc, long long int strideC,
                                       int batchCount,
                                       int algo) {
            CUBLAS_CHECK(cublasGemmStridedBatchedEx((cublasHandle_t) handle,
                                                    TransA ? CUBLAS_OP_T : CUBLAS_OP_N, TransB ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                    m, n, k,
                                                    &alpha,
                                                    A, CUDA_R_16F, lda, strideA,
                                                    B, CUDA_R_16F, ldb, strideB,
                                                    &beta,
                                                    C, CUDA_R_16F, ldc, strideC,
                                                    batchCount,
                                                    CUDA_R_32F,
                                                    (cublasGemmAlgo_t) algo));
    }
#endif

    template <>
    void T2T<float, float>(const float* in, float* out, size_t n) {
        std::memcpy(out, in, n * sizeof(float));
    }

    template <>
    void T2T<half, half>(const half* in, half* out, size_t n) {
        std::memcpy(out, in, n * sizeof(half));
    }

    template <>
    void T2T<float, half>(const float* in, half* out, size_t n) {
        float2half(in, out, n);
    }

    template <>
    void T2T<half, float>(const half* in, float* out, size_t n) {
        half2float(in, out, n);
    }

    template <>
    int gemm_algo<float>(const char* env) {
        char* val = std::getenv(env);
        if (val == nullptr) {
#ifdef HAVE_CUDA
            return (int) CUBLAS_GEMM_DEFAULT;
#endif
#ifdef HAVE_MKL
            return 0;
#endif
        }
        return std::atoi(val);
    }

    template <>
    int gemm_algo<half>(const char* env) {
        char* val = std::getenv(env);
        if (val == nullptr) {
#ifdef HAVE_CUDA
            return (int) CUBLAS_GEMM_DEFAULT_TENSOR_OP;
#endif
#ifdef HAVE_MKL
            return 0;
#endif
        }
        return std::atoi(val);
    }
}
