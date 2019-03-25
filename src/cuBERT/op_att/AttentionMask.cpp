#include "cuBERT/common.h"
#include "AttentionMask.h"

namespace cuBERT {
    const static float ONE = 1.f;

#ifdef HAVE_MKL
    template<>
    void _not<float>(const char *in,
                     float *out,
                     const int N,
                     void *stream) {
#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            out[i] = !in[i];
        }
    }
#endif

    template <typename T>
    AttentionMask<T>::AttentionMask(void* handle,
                                    size_t seq_length, size_t num_attention_heads, size_t max_batch_size) {
        this->handle = handle;
        this->seq_length = seq_length;
        this->num_attention_heads = num_attention_heads;

        this->ones = static_cast<T *>(cuBERT::malloc(sizeof(T) * num_attention_heads * seq_length));
        T one; T2T(&ONE, &one, 1);
        cuBERT:fill_n<T>(ones, num_attention_heads * seq_length, one);

        this->neg = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * seq_length));
    }

    template <typename T>
    AttentionMask<T>::~AttentionMask() {
        cuBERT::free(neg);
        cuBERT::free(ones);
    }

    template <typename T>
    void AttentionMask<T>::compute(size_t batch_size, char *in, T *out_gpu) {
        void *stream = cuBERT::blas_get_stream(handle);
        _not<T>(in, neg, batch_size * seq_length, stream);

        cuBERT::blas_gemm_strided_batch(handle,
                                        false, false,
                                        seq_length, num_attention_heads * seq_length, 1,
                                        1.f,
                                        neg, seq_length, seq_length,
                                        ones, 1, 0,
                                        0.f,
                                        out_gpu, seq_length, seq_length * num_attention_heads * seq_length,
                                        batch_size);
    }

    template class AttentionMask<float>;
#ifdef HAVE_CUDA
    template class AttentionMask<half>;
#endif
}
