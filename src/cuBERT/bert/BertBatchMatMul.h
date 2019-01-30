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
                        size_t max_batch_size,
                        size_t seq_length, size_t num_attention_heads, size_t size_per_head,
                        float* query, float* key, float* out,
                        float alpha = 1, float beta = 0);

        virtual ~BertQK();

        void compute(size_t batch_size);

        void compute_cpu(size_t batch_size);

    private:
        cublasHandle_t handle;

        size_t seq_length;
        size_t num_attention_heads;
        size_t size_per_head;

        float alpha;
        float beta;

        // gpu buffer
        const float **query_array_gpu;
        const float **key_array_gpu;
        float **out_array_gpu;

        // cpu buffer
        const float **query_array_cpu;
        const float **key_array_cpu;
        float **out_array_cpu;
    };

    class BertQKV {
    public:
        explicit BertQKV(cublasHandle_t handle,
                         size_t max_batch_size,
                         size_t seq_length, size_t num_attention_heads, size_t size_per_head,
                         float* qk, float* value, float* out,
                         float alpha = 1, float beta = 0);

        virtual ~BertQKV();

        void compute(size_t batch_size);

        void compute_cpu(size_t batch_size);

    private:
        cublasHandle_t handle;

        size_t seq_length;
        size_t num_attention_heads;
        size_t size_per_head;

        float alpha;
        float beta;

        // gpu buffer
        const float **qk_array_gpu;
        const float **value_array_gpu;
        float **out_array_gpu;

        // cpu buffer
        const float **qk_array_cpu;
        const float **value_array_cpu;
        float **out_array_cpu;
    };
}

#endif //CUBERT_BERTQK_H
