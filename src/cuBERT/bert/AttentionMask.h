#ifndef CUBERT_ATTENTIONMASK_H
#define CUBERT_ATTENTIONMASK_H

#include <cstddef>

namespace cuBERT {
    void _not(const char *in,
              float *out,
              const int N,
              void *stream);

/**
 * 1. compute: 1 - in_gpu
 * 2. broadcast attention mask from (N, seq_length) to (N, num_attention_heads * seq_length, seq_length)
 */
    class AttentionMask {
    public:
        explicit AttentionMask(void* handle, size_t seq_length, size_t num_attention_heads,
                               size_t max_batch_size);

        virtual ~AttentionMask();

        void compute(size_t batch_size, char *in_gpu, float *out_gpu);

    private:
        void* handle;

        size_t seq_length;
        size_t num_attention_heads;

        // cpu/gpu buffer
        float *ones;
        float *neg;
    };
}

#endif //CUBERT_ATTENTIONMASK_H
