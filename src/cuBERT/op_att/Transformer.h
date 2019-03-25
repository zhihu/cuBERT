#ifndef CUBERT_TRANSFORMER_H
#define CUBERT_TRANSFORMER_H

#include <vector>
#include <unordered_map>

#include "cuBERT/op/Dense.h"
#include "cuBERT/op/LayerNorm.h"
#include "cuBERT/op/GELU.h"
#include "./AttentionSelf.h"
#include "./AttentionMask.h"

namespace cuBERT {
    template <typename T>
    class Transformer {
    public:
        explicit Transformer(void* cublas,
                             const std::string &var_prefix,
                             const std::unordered_map<std::string, T *> &var,
                             size_t max_batch_size,
                             size_t seq_length,
                             size_t hidden_size = 768,
                             size_t num_hidden_layers = 12,
                             size_t num_attention_heads = 12,
                             size_t intermediate_size = 3072);

        virtual ~Transformer();

        /**
         *
         * @param batch_size
         * @param input_gpu float Tensor of shape [batch_size, seq_length, hidden_size].
         * @param output_gpu
         */
        T *compute(size_t batch_size, T *input_gpu, char *attention_mask);

        void _pre_compute(size_t batch_size);

        T *_in_compute(size_t batch_size, T *input_gpu, char *attention_mask);

    private:
        void* cublas;

        size_t num_hidden_layers;
        size_t seq_length;
        size_t intermediate_size;

        AttentionMask<T> *attention_mask;
        std::vector<Dense<T> *> attention_output_dense;
        std::vector<LayerNorm<T> *> attention_output_norm;
        std::vector<Dense<T> *> intermediate_dense;
        std::vector<GELU<T> *> intermediate_act_fn;
        std::vector<Dense<T> *> output_dense;
        std::vector<LayerNorm<T> *> output_layer_norm;
        std::vector<AttentionSelf<T> *> attention_self;

        // gpu buffer
        T *neg_attention_mask_buffer;
        std::vector<T *> attention_heads;
        std::vector<T *> attention_output;
        std::vector<T *> intermediate_output;
        std::vector<T *> layer_output;
    };
}

#endif //CUBERT_TRANSFORMER_H
