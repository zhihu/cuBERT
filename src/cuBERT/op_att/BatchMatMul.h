#ifndef CUBERT_BERTQK_H
#define CUBERT_BERTQK_H

#include <cstddef>

namespace cuBERT {

    /**
     * Batch MatMul between query and key.
     * Q: [N, seq_length, num_attention_heads, size_per_head]
     * K: [N, seq_length, num_attention_heads, size_per_head]
     *
     * QK = Q * K^T: [N, seq_length, num_attention_heads, seq_length]
     */
    template <typename T>
    class Att_Q_K {
    public:
        explicit Att_Q_K(void* handle,
                        size_t max_batch_size,
                        size_t seq_length, size_t num_attention_heads, size_t size_per_head,
                        T* query, T* key, T* out,
                        float alpha = 1.f, float beta = 0.f);

        virtual ~Att_Q_K();

        void compute(size_t batch_size);

    private:
        void* handle;

        size_t seq_length;
        size_t num_attention_heads;
        size_t size_per_head;

        float alpha;
        float beta;
        int algo;

        // cpu/gpu buffer
        const T **query_array;
        const T **key_array;
        T **out_array;
    };

    template <typename T>
    class Att_QK_V {
    public:
        explicit Att_QK_V(void* handle,
                         size_t max_batch_size,
                         size_t seq_length, size_t num_attention_heads, size_t size_per_head,
                         T* qk, T* value, T* out,
                          float alpha = 1.f, float beta = 0.f);

        virtual ~Att_QK_V();

        void compute(size_t batch_size);

    private:
        void* handle;

        size_t seq_length;
        size_t num_attention_heads;
        size_t size_per_head;

        float alpha;
        float beta;
        int algo;

        // cpu/gpu buffer
        const T **qk_array;
        const T **value_array;
        T **out_array;
    };
}

#endif //CUBERT_BERTQK_H
