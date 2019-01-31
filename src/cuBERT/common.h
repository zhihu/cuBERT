#ifndef CUBERT_COMMON_H
#define CUBERT_COMMON_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mkl.h>

#include <cstdlib>
#include <cstdio>

namespace cuBERT {

    void initialize(bool force_cpu = false);
    void finalize();
    bool gpu();

    int get_gpu_count();
    void set_gpu(int device);

    void *cuda_stream_create();
    void cuda_stream_destroy(void *stream);
    void cuda_stream_synchronize(void *stream);

    void *malloc(size_t size);
    void free(void *ptr);

    void memcpy(void *dst, const void *src, size_t n);
    void memcpy(void *dst, const void *src, size_t n, int kind);
    void memcpyAsync(void *dst, const void *src, size_t n, int kind, void* stream);
    void fill_n(float *dst, size_t n, float value);

//    typedef enum {
//        CUBLAS_OP_N=0,
//        CUBLAS_OP_T=1,
//        CUBLAS_OP_C=2
//    } cublasOperation_t;

//    enum __device_builtin__ cudaMemcpyKind
//    {
//        cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
//        cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
//        cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
//        cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
//        cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
//    };

//    typedef enum {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
//    typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;

    void* blas_create();
    void blas_destroy(void *handle);
    void* blas_get_stream(void* handle);
    void blas_set_stream(void* handle, void* streamId);

    void blas_sgemm(void *handle,
                    const bool TransA, const bool TransB,
                    const int M, const int N, const int K,
                    const float alpha,
                    const float *A, const int lda,
                    const float *B, const int ldb,
                    const float beta,
                    float *C, const int ldc);

    void blas_sgemm_batch(void *handle,
                          const bool TransA, const bool TransB,
                          int m, int n, int k,
                          const float alpha,
                          const float ** Aarray, int lda,
                          const float ** Barray, int ldb,
                          const float beta,
                          float ** Carray, int ldc,
                          int batchCount);

    void blas_sgemm_strided_batch(void *handle,
                                  const bool TransA, const bool TransB,
                                  int m, int n, int k,
                                  const float alpha,
                                  const float *A, int lda, long long int strideA,
                                  const float *B, int ldb, long long int strideB,
                                  const float beta,
                                  float *C, int ldc, long long int strideC,
                                  int batchCount);

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
}

#endif //CUBERT_COMMON_H
