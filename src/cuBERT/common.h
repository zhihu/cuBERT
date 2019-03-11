#ifndef CUBERT_COMMON_H
#define CUBERT_COMMON_H

#include <cstddef>

namespace cuBERT {

    void initialize(bool force_cpu = false);
    void finalize();
    bool gpu();
    const char *get_error_string(int error);

    int get_gpu_count();
    void set_gpu(int device);

    void *cuda_stream_create();
    void cuda_stream_destroy(void *stream);
    void cuda_stream_synchronize(void *stream);

    void *malloc(size_t size);
    void free(void *ptr);

//    enum __device_builtin__ cudaMemcpyKind
//    {
//        cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
//        cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
//        cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
//        cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
//        cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
//    };
    void memcpy(void *dst, const void *src, size_t n, int kind);
    void memcpyAsync(void *dst, const void *src, size_t n, int kind, void* stream);
    void fill_n(float *dst, size_t n, float value);

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
}

#endif //CUBERT_COMMON_H
