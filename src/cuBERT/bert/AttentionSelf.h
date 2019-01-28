//
// Created by 田露 on 2019/1/18.
//

#ifndef CUBERT_ATTENTIONSELF_H
#define CUBERT_ATTENTIONSELF_H


#include <cublas_v2.h>
#include <cudnn.h>
#include <string>
#include <unordered_map>

#include "cuBERT/op/Dense.h"
#include "cuBERT/op/Softmax.h"
#include "cuBERT/bert/BertBatchMatMul.h"

namespace cuBERT {
    class AttentionSelf {
    public:
        explicit AttentionSelf(cublasHandle_t cublas, cudnnHandle_t cudnn,
                               const std::string &var_prefix,
                               const std::unordered_map<std::string, float *> &var,
                               size_t max_batch_size,
                               size_t seq_length,
                               float *context_layer_out,
                               size_t width = 768, size_t num_attention_heads = 12, size_t size_per_head = 64);

        virtual ~AttentionSelf();

        /**
         *
         * @param batch_size
         * @param in_gpu [batch_size, seq_length, width]
         * @param out_gpu [batch_size, seq_length, num_attention_heads * size_per_head]
         */
        void compute(size_t batch_size, float *in_gpu, float *neg_attention_mask);

        void _pre_compute(size_t batch_size);

        void _in_compute(size_t batch_size, float *in_gpu, float *neg_attention_mask);

    private:
        cublasHandle_t cublas;
        cudnnHandle_t cudnn;

        size_t seq_length;
        size_t num_attention_heads;
        size_t size_per_head;

        Dense *query_layer;
        Dense *key_layer;
        Dense *value_layer;
        BertQK *bqk;
        BertQKV *bqkv;
        Softmax *softmax;

        // buffers
        float *query_layer_out;
        float *key_layer_out;
        float *value_layer_out;
        float *attention_scores;

        // output
        float *context_layer_out;
    };
}

#endif //CUBERT_ATTENTIONSELF_H
