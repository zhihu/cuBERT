#ifndef CUBERT_ATTENTIONMASK_H
#define CUBERT_ATTENTIONMASK_H

#include <cstddef>
#include <cublas_v2.h>

namespace cuBERT {
    __host__ void _not(const char *in,
                       float *out,
                       const int N,
                       cudaStream_t stream);

/**
 * 1. compute: 1 - in_gpu
 * 2. broadcast attention mask from (N, seq_length) to (N, num_attention_heads * seq_length, seq_length)
 */
    class AttentionMask {
    public:
        explicit AttentionMask(cublasHandle_t handle, size_t seq_length, size_t num_attention_heads,
                               size_t max_batch_size);

        virtual ~AttentionMask();

        void compute(size_t batch_size, char *in_gpu, float *out_gpu);

    private:
        cublasHandle_t handle;

        size_t seq_length;
        size_t num_attention_heads;

        float *ones_gpu;
        float *neg_gpu;
    };
}

#endif //CUBERT_ATTENTIONMASK_H
