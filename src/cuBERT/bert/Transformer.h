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
    class Transformer {
    public:
        explicit Transformer(cublasHandle_t cublas,
                             const std::string &var_prefix,
                             const std::unordered_map<std::string, float *> &var,
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
        float *compute(size_t batch_size, float *input_gpu, char *attention_mask);

        void _pre_compute(size_t batch_size);

        float *_in_compute(size_t batch_size, float *input_gpu, char *attention_mask);

        float *compute_cpu(size_t batch_size, float *input_cpu, char *attention_mask);

        void _pre_compute_cpu(size_t batch_size);

        float *_in_compute_cpu(size_t batch_size, float *input_cpu, char *attention_mask);

    private:
        cublasHandle_t cublas;

        size_t num_hidden_layers;
        size_t seq_length;
        size_t intermediate_size;

        AttentionMask *attention_mask;
        std::vector<Dense *> attention_output_dense;
        std::vector<LayerNorm *> attention_output_norm;
        std::vector<Dense *> intermediate_dense;
        std::vector<GELU *> intermediate_act_fn;
        std::vector<Dense *> output_dense;
        std::vector<LayerNorm *> output_layer_norm;

        std::vector<AttentionSelf *> attention_self_gpu;
        std::vector<AttentionSelf *> attention_self_cpu;

        // gpu buffer
        float *neg_attention_mask_buffer_gpu;
        std::vector<float *> attention_heads_gpu;
        std::vector<float *> attention_output_gpu;
        std::vector<float *> intermediate_output_gpu;
        std::vector<float *> layer_output_gpu;

        // cpu buffer
        float *neg_attention_mask_buffer_cpu;
        std::vector<float *> attention_heads_cpu;
        std::vector<float *> attention_output_cpu;
        std::vector<float *> intermediate_output_cpu;
        std::vector<float *> layer_output_cpu;
    };
}

#endif //CUBERT_TRANSFORMER_H
