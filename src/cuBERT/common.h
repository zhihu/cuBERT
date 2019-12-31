#ifndef CUBERT_COMMON_H
#define CUBERT_COMMON_H

#include <cstddef>
#include <cstring>
#include <algorithm>

#include <half.hpp>
#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#else
typedef half_float::half half;
#endif

namespace cuBERT {

    void initialize();
    void finalize();

    int get_gpu_count();
    void set_gpu(int device);
    void gpu_info(int device);

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

    template <typename T>
    inline void fill_n(T *dst, size_t n, const T& value) {
#ifdef HAVE_CUDA
            auto *dst_cpu = new T[n];
            std::fill_n(dst_cpu, n, value);
            cuBERT::memcpy(dst, dst_cpu, sizeof(T) * n, 1);
            delete[] dst_cpu;
#else
            std::fill_n(dst, n, value);
#endif
    }

    void* blas_create();
    void blas_destroy(void *handle);
    void* blas_get_stream(void* handle);
    void blas_set_stream(void* handle, void* streamId);

    template<typename T>
    void blas_gemm(void *handle,
                   const bool TransA, const bool TransB,
                   const int M, const int N, const int K,
                   const float alpha,
                   const T *A, const int lda,
                   const T *B, const int ldb,
                   const float beta,
                   T *C, const int ldc,
                   int algo = -1);

    template<typename T>
    void blas_gemm_batch(void *handle,
                         const bool TransA, const bool TransB,
                         int m, int n, int k,
                         const float alpha,
                         const T **Aarray, int lda,
                         const T **Barray, int ldb,
                         const float beta,
                         T **Carray, int ldc,
                         int batchCount, 
                         int algo = -1);

    template<typename T>
    void blas_gemm_strided_batch(void *handle,
                                 const bool TransA, const bool TransB,
                                 int m, int n, int k,
                                 const float alpha,
                                 const T *A, int lda, long long int strideA,
                                 const T *B, int ldb, long long int strideB,
                                 const float beta,
                                 T *C, int ldc, long long int strideC,
                                 int batchCount,
                                 int algo = -1);

    inline void float2half(const float* fs, void* hs, size_t n) {
        auto* _hs = (half_float::half*) hs;
        for (int i = 0; i < n; ++i) {
            _hs[i] = fs[i];
        }
    }

    inline void half2float(const void* hs, float* fs, size_t n) {
        auto* _hs = (half_float::half*) hs;
        for (int i = 0; i < n; ++i) {
            fs[i] = (float) _hs[i];
        }
    }

    template <typename T, typename V>
    void T2T(const T* in, V* out, size_t n);

    template <typename T>
    int gemm_algo(const char* env);
}

#endif //CUBERT_COMMON_H
