#include <dlfcn.h>
#include <mkl.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>

#include "./common.h"

namespace cuBERT {

#define CUDA_CHECK(call) do {                                           \
    int err = call;                                                     \
    if (0 != err) {                                                     \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",   \
                __FILE__, __LINE__, get_error_string(err));             \
        exit(EXIT_FAILURE);                                             \
    } } while(0)

#define CUBLAS_CHECK(call) do {                                         \
    int err = call;                                                     \
    if (0 != err) {                                                     \
        fprintf(stderr, "Cublas error in file '%s' in line %i.\n",      \
                __FILE__, __LINE__);                                    \
        exit(EXIT_FAILURE);                                             \
    } } while(0)

    struct CUDA {
        void *cudart;
        void *cublas;

        const char *(*cudaGetErrorString)(int error);
        int (*cudaGetDeviceCount)(int *count);
        int (*cudaSetDevice)(int device);

        int (*cudaStreamCreate)(void **pStream);
        int (*cudaStreamDestroy)(void *stream);
        int (*cudaStreamSynchronize)(void *stream);

        int (*cudaMalloc)(void **devPtr, size_t size);
        int (*cudaFree)(void *devPtr);
        int (*cudaMemcpy)(void *dst, const void *src, size_t count, int kind);
        int (*cudaMemcpyAsync)(void *dst, const void *src, size_t count, int kind, void *stream);

        int (*cublasCreate_v2)(void **handle);
        int (*cublasDestroy_v2)(void *handle);
        int (*cublasGetStream_v2)(void *handle, void **streamId);
        int (*cublasSetStream_v2)(void *handle, void *streamId);

        int (*cublasSgemm_v2)(void *handle,
                              int transa, int transb,
                              int m, int n, int k,
                              const float *alpha,
                              const float *A, int lda,
                              const float *B, int ldb,
                              const float *beta,
                              float *C, int ldc);

        int (*cublasSgemmBatched)(void *handle,
                                  int transa, int transb,
                                  int m, int n, int k,
                                  const float *alpha,  /* host or device pointer */
                                  const float *const Aarray[], int lda,
                                  const float *const Barray[], int ldb,
                                  const float *beta,   /* host or device pointer */
                                  float *const Carray[], int ldc,
                                  int batchCount);

        int (*cublasSgemmStridedBatched)(void *handle,
                                         int transa, int transb,
                                         int m, int n, int k,
                                         const float *alpha,  /* host or device pointer */
                                         const float *A, int lda, long long int strideA,
                                         const float *B, int ldb, long long int strideB,
                                         const float *beta,   /* host or device pointer */
                                         float *C, int ldc, long long int strideC,
                                         int batchCount);
    };

    bool force_cpu;
    CUDA CUDA_SO;

#ifdef HAVE_CUDA
    const bool compile_with_cuda = true;
#else
    const bool compile_with_cuda = false;
#endif

    bool gpu() {
        // 1. COMPILE with CUDA
        // 2. libcudart.so runtime FOUND
        // 3. choose not force CPU
        return compile_with_cuda && !force_cpu && CUDA_SO.cudart != nullptr && CUDA_SO.cublas != nullptr;
    }

    void initialize(bool force_cpu) {
        CUDA_SO.cudart = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
        CUDA_SO.cublas = dlopen("libcublas.so", RTLD_NOW | RTLD_LOCAL);
        if (CUDA_SO.cudart != nullptr && CUDA_SO.cublas != nullptr) {
            std::cout << "CUDA runtime found" << std::endl;
            if (!compile_with_cuda) {
                std::cerr << "cuBERT is not compiled with CUDA" << std::endl;
            }

            CUDA_SO.cudaGetErrorString = (const char *(*)(int)) (dlsym(CUDA_SO.cudart, "cudaGetErrorString"));
            CUDA_SO.cudaGetDeviceCount = (int (*)(int *)) (dlsym(CUDA_SO.cudart, "cudaGetDeviceCount"));
            CUDA_SO.cudaSetDevice = (int (*)(int)) (dlsym(CUDA_SO.cudart, "cudaSetDevice"));

            CUDA_SO.cudaStreamCreate = (int (*)(void **)) (dlsym(CUDA_SO.cudart, "cudaStreamCreate"));
            CUDA_SO.cudaStreamDestroy = (int (*)(void *)) (dlsym(CUDA_SO.cudart, "cudaStreamDestroy"));
            CUDA_SO.cudaStreamSynchronize = (int (*)(void *)) (dlsym(CUDA_SO.cudart, "cudaStreamSynchronize"));

            CUDA_SO.cudaMalloc = (int (*)(void **, size_t)) (dlsym(CUDA_SO.cudart, "cudaMalloc"));
            CUDA_SO.cudaFree = (int (*)(void *)) (dlsym(CUDA_SO.cudart, "cudaFree"));
            CUDA_SO.cudaMemcpy = (int (*)(void *, const void *, size_t, int)) (dlsym(CUDA_SO.cudart, "cudaMemcpy"));
            CUDA_SO.cudaMemcpyAsync = (int (*)(void *, const void *, size_t, int, void *))
                    (dlsym(CUDA_SO.cudart, "cudaMemcpyAsync"));

            CUDA_SO.cublasCreate_v2 = (int (*)(void **)) (dlsym(CUDA_SO.cublas, "cublasCreate_v2"));
            CUDA_SO.cublasDestroy_v2 = (int (*)(void *)) (dlsym(CUDA_SO.cublas, "cublasDestroy_v2"));
            CUDA_SO.cublasGetStream_v2 = (int (*)(void *, void **)) (dlsym(CUDA_SO.cublas, "cublasGetStream_v2"));
            CUDA_SO.cublasSetStream_v2 = (int (*)(void *, void *)) (dlsym(CUDA_SO.cublas, "cublasSetStream_v2"));

            CUDA_SO.cublasSgemm_v2 = (int (*)(void *, int, int, int, int, int, const float *, const float *, int,
                                              const float *, int, const float *, float *, int))
                    (dlsym(CUDA_SO.cublas, "cublasSgemm_v2"));
            CUDA_SO.cublasSgemmBatched = (int (*)(void *, int, int, int, int, int, const float *, const float *const *,
                                                  int, const float *const *, int, const float *, float *const *, int,
                                                  int))
                    (dlsym(CUDA_SO.cublas, "cublasSgemmBatched"));

            CUDA_SO.cublasSgemmStridedBatched = (int (*)(void *, int, int, int, int, int, const float *, const float *,
                                                         int, long long int, const float *, int, long long int,
                                                         const float *, float *, int, long long int, int))
                    (dlsym(CUDA_SO.cublas, "cublasSgemmStridedBatched"));
        }

        cuBERT::force_cpu = force_cpu;
    }

    void finalize() {
        if (CUDA_SO.cudart != nullptr) {
            dlclose(CUDA_SO.cudart);
        }
        if (CUDA_SO.cublas != nullptr) {
            dlclose(CUDA_SO.cublas);
        }
    }

    const char *get_error_string(int error) {
        if (gpu()) {
            return CUDA_SO.cudaGetErrorString(error);
        } else {
            return "";
        }
    }

    int get_gpu_count() {
        if (gpu()) {
            int count;
            CUDA_CHECK(CUDA_SO.cudaGetDeviceCount(&count));
            return count;
        } else {
            return 0;
        }
    }

    void set_gpu(int device) {
        if (gpu()) {
            CUDA_CHECK(CUDA_SO.cudaSetDevice(device));
        }
    }

    void *cuda_stream_create() {
        if (gpu()) {
            void *ptr;
            CUDA_CHECK(CUDA_SO.cudaStreamCreate(&ptr));
            return ptr;
        } else {
            return nullptr;
        }
    }

    void cuda_stream_destroy(void *stream) {
        if (gpu()) {
            CUDA_CHECK(CUDA_SO.cudaStreamDestroy(stream));
        }
    }

    void cuda_stream_synchronize(void *stream) {
        if (gpu()) {
            CUDA_CHECK(CUDA_SO.cudaStreamSynchronize(stream));
        }
    }

    void *malloc(size_t size) {
        if (gpu()) {
            void *ptr;
            CUDA_CHECK(CUDA_SO.cudaMalloc(&ptr, size));
            return ptr;
        } else {
            return std::malloc(size);
        }
    }

    void free(void *ptr) {
        if (gpu()) {
            CUDA_CHECK(CUDA_SO.cudaFree(ptr));
        } else {
            std::free(ptr);
        }
    }

    void memcpy(void *dst, const void *src, size_t n, int kind) {
        if (gpu()) {
            CUDA_CHECK(CUDA_SO.cudaMemcpy(dst, src, n, kind));
        } else {
            std::memcpy(dst, src, n);
        }
    }

    void memcpyAsync(void *dst, const void *src, size_t n, int kind, void *stream) {
        if (gpu()) {
            CUDA_CHECK(CUDA_SO.cudaMemcpyAsync(dst, src, n, kind, stream));
        } else {
            std::memcpy(dst, src, n);
        }
    }

    void fill_n(float *dst, size_t n, float value) {
        if (gpu()) {
            auto *dst_cpu = new float[n];
            std::fill_n(dst_cpu, n, value);
            cuBERT::memcpy(dst, dst_cpu, sizeof(float) * n, 1);
            delete[]dst_cpu;
        } else {
            std::fill_n(dst, n, value);
        }
    }

    void *blas_create() {
        if (gpu()) {
            void *ptr;
            CUBLAS_CHECK(CUDA_SO.cublasCreate_v2(&ptr));
            return ptr;
        } else {
            return nullptr;
        }
    }

    void blas_destroy(void *handle) {
        if (gpu()) {
            CUBLAS_CHECK(CUDA_SO.cublasDestroy_v2(handle));
        }
    }

    void *blas_get_stream(void *handle) {
        if (gpu()) {
            void *streamId;
            CUBLAS_CHECK(CUDA_SO.cublasGetStream_v2(handle, &streamId));
            return streamId;
        } else {
            return nullptr;
        }
    }

    void blas_set_stream(void *handle, void *streamId) {
        if (gpu()) {
            CUBLAS_CHECK(CUDA_SO.cublasSetStream_v2(handle, streamId));
        }
    }

    void blas_sgemm(void *handle,
                    const bool TransA, const bool TransB,
                    const int M, const int N, const int K,
                    const float alpha,
                    const float *A, const int lda,
                    const float *B, const int ldb,
                    const float beta,
                    float *C, const int ldc) {
        if (gpu()) {
            CUBLAS_CHECK(CUDA_SO.cublasSgemm_v2(handle,
                                                TransA ? 1 : 0, TransB ? 1 : 0,
                                                M, N, K,
                                                &alpha,
                                                A, lda,
                                                B, ldb,
                                                &beta,
                                                C, ldc));
        } else {
            cblas_sgemm(CblasColMajor,
                        TransA ? CblasTrans : CblasNoTrans, TransB ? CblasTrans : CblasNoTrans,
                        M, N, K,
                        alpha,
                        A, lda,
                        B, ldb,
                        beta,
                        C, ldc);
        }
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
        if (gpu()) {
            CUBLAS_CHECK(CUDA_SO.cublasSgemmBatched(handle,
                                                    TransA ? 1 : 0, TransB ? 1 : 0,
                                                    m, n, k,
                                                    &alpha,
                                                    Aarray, lda,
                                                    Barray, ldb,
                                                    &beta,
                                                    Carray, ldc,
                                                    batchCount));
        } else {
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
        }
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
        if (gpu()) {
            CUBLAS_CHECK(CUDA_SO.cublasSgemmStridedBatched(handle,
                                                           TransA ? 1 : 0, TransB ? 1 : 0,
                                                           m, n, k,
                                                           &alpha,
                                                           A, lda, strideA,
                                                           B, ldb, strideB,
                                                           &beta,
                                                           C, ldc, strideC,
                                                           batchCount));
        } else {
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
        }
    }
}
