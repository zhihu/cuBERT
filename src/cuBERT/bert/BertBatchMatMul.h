//
// Created by 田露 on 2019/1/26.
//

#ifndef CUBERT_BERTQK_H
#define CUBERT_BERTQK_H

#include <cstddef>
#include <cublas_v2.h>

namespace cuBERT {

    /**
     * Batch MatMul between query and key.
     * Q: [N, seq_length, num_attention_heads, size_per_head]
     * K: [N, seq_length, num_attention_heads, size_per_head]
     *
     * QK = Q * K^T: [N, seq_length, num_attention_heads, seq_length]
     */
    class BertQK {
    public:
        explicit BertQK(cublasHandle_t handle,
                        size_t seq_length, size_t num_attention_heads, size_t size_per_head,
                        float alpha = 1, float beta = 0);

        virtual ~BertQK() = default;

        void compute(size_t batch_size, float* query, float* key, float* out);

    private:
        cublasHandle_t handle;

        size_t seq_length;
        size_t num_attention_heads;
        size_t size_per_head;

        float alpha;
        float beta;
    };
}

#endif //CUBERT_BERTQK_H
