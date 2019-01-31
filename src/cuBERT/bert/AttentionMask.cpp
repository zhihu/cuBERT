#include <omp.h>

#include "cuBERT/common.h"
#include "AttentionMask.h"

namespace cuBERT {
    const static float ZERO = 0;
    const static float ONE = 1;

    AttentionMask::AttentionMask(void* handle,
                                 size_t seq_length, size_t num_attention_heads, size_t max_batch_size) {
        this->handle = handle;
        this->seq_length = seq_length;
        this->num_attention_heads = num_attention_heads;

        this->ones = static_cast<float *>(cuBERT::malloc(sizeof(float) * num_attention_heads * seq_length));
        cuBERT:fill_n(ones, num_attention_heads * seq_length, 1.f);

        this->neg = static_cast<float *>(cuBERT::malloc(sizeof(float) * max_batch_size * seq_length));
    }

    AttentionMask::~AttentionMask() {
        cuBERT::free(neg);
        cuBERT::free(ones);
    }

    void AttentionMask::compute(size_t batch_size, char *in, float *out_gpu) {
        if (gpu()) {
            void *stream = cuBERT::blas_get_stream(handle);
            _not(in, neg, batch_size * seq_length, stream);
        } else {
#pragma omp parallel for
            for (int i = 0; i < batch_size * seq_length; ++i) {
                neg[i] = !in[i];
            }
        }

        cuBERT::blas_sgemm_strided_batch(handle,
                                         false, false,
                                         seq_length, num_attention_heads * seq_length, 1,
                                         ONE,
                                         neg, seq_length, seq_length,
                                         ones, 1, 0,
                                         ZERO,
                                         out_gpu, seq_length, seq_length * num_attention_heads * seq_length,
                                         batch_size);
    }
}
