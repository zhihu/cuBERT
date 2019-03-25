#ifndef CUBERT_ATTENTIONSELF_H
#define CUBERT_ATTENTIONSELF_H

#include <string>
#include <unordered_map>

#include "cuBERT/op/Dense.h"
#include "cuBERT/op/Softmax.h"
#include "BatchMatMul.h"

namespace cuBERT {
    template <typename T>
    class AttentionSelf {
    public:
        explicit AttentionSelf(void* cublas,
                               const std::string &var_prefix,
                               const std::unordered_map<std::string, T *> &var,
                               size_t max_batch_size,
                               size_t seq_length,
                               T *context_layer_out,
                               size_t width = 768, size_t num_attention_heads = 12, size_t size_per_head = 64);

        virtual ~AttentionSelf();

        /**
         *
         * @param batch_size
         * @param in_gpu [batch_size, seq_length, width]
         * @param out_gpu [batch_size, seq_length, num_attention_heads * size_per_head]
         */
        void compute(size_t batch_size, T *in_gpu, T *neg_attention_mask);

        void _pre_compute(size_t batch_size);

        void _in_compute(size_t batch_size, T *in_gpu, T *neg_attention_mask);

    private:
        void* cublas;

        size_t seq_length;
        size_t num_attention_heads;
        size_t size_per_head;

        Dense<T> *query_layer;
        Dense<T> *key_layer;
        Dense<T> *value_layer;
        Softmax<T> *softmax;
        Att_Q_K<T> *bqk;
        Att_QK_V<T> *bqkv;

        // cpu/gpu buffers
        T *query_layer_out;
        T *key_layer_out;
        T *value_layer_out;
        T *attention_scores;

        // output
        T *context_layer_out;
    };
}

#endif //CUBERT_ATTENTIONSELF_H
