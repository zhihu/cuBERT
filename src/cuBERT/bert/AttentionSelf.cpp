//
// Created by 田露 on 2019/1/18.
//
#include <cmath>
#include <cuda_runtime.h>

#include "cuBERT/common.h"
#include "AttentionSelf.h"

namespace cuBERT {
    AttentionSelf::AttentionSelf(cublasHandle_t cublas, cudnnHandle_t cudnn,
                                 const std::string &var_prefix,
                                 const std::unordered_map<std::string, float *> &var,
                                 size_t max_batch_size,
                                 size_t seq_length,
                                 float *context_layer_out,
                                 size_t width, size_t num_attention_heads, size_t size_per_head) {
        this->cublas = cublas;
        this->cudnn = cudnn;
        this->seq_length = seq_length;
        this->num_attention_heads = num_attention_heads;
        this->size_per_head = size_per_head;

        this->context_layer_out = context_layer_out;

        float *query_layer_kernel = var.at(var_prefix + "/query/kernel");
        float *query_layer_bias = var.at(var_prefix + "/query/bias");
        query_layer = new Dense(cublas,
                                width, num_attention_heads * size_per_head,
                                query_layer_kernel, query_layer_bias,
                                max_batch_size * seq_length);

        float *key_layer_kernel = var.at(var_prefix + "/key/kernel");
        float *key_layer_bias = var.at(var_prefix + "/key/bias");
        key_layer = new Dense(cublas,
                              width, num_attention_heads * size_per_head,
                              key_layer_kernel, key_layer_bias,
                              max_batch_size * seq_length);

        float *value_layer_kernel = var.at(var_prefix + "/value/kernel");
        float *value_layer_bias = var.at(var_prefix + "/value/bias");
        value_layer = new Dense(cublas,
                                width, num_attention_heads * size_per_head,
                                value_layer_kernel, value_layer_bias,
                                max_batch_size * seq_length);

        softmax = new Softmax(cudnn, seq_length);

        CUDA_CHECK(cudaMalloc(&query_layer_out, sizeof(float) * max_batch_size * seq_length * num_attention_heads * size_per_head));
        CUDA_CHECK(cudaMalloc(&key_layer_out, sizeof(float) * max_batch_size * seq_length * num_attention_heads * size_per_head));
        CUDA_CHECK(cudaMalloc(&value_layer_out, sizeof(float) * max_batch_size * seq_length * num_attention_heads * size_per_head));
        CUDA_CHECK(cudaMalloc(&attention_scores, sizeof(float) * max_batch_size * num_attention_heads * seq_length * seq_length));

        bqk = new BertQK(cublas, max_batch_size, seq_length, num_attention_heads, size_per_head,
                         query_layer_out, key_layer_out, attention_scores,
                         1.0 / std::sqrt(size_per_head), -10000.0f);

        bqkv = new BertQKV(cublas, max_batch_size, seq_length, num_attention_heads, size_per_head,
                           attention_scores, value_layer_out, context_layer_out);
    }

    AttentionSelf::~AttentionSelf() {
        delete bqkv;
        delete bqk;

        CUDA_CHECK(cudaFree(attention_scores));
        CUDA_CHECK(cudaFree(value_layer_out));
        CUDA_CHECK(cudaFree(key_layer_out));
        CUDA_CHECK(cudaFree(query_layer_out));

        delete softmax;
        delete value_layer;
        delete key_layer;
        delete query_layer;
    }

    void AttentionSelf::compute(size_t batch_size, float *in_gpu, float *neg_attention_mask) {
        cudaStream_t stream = nullptr;
        CUBLAS_CHECK(cublasGetStream_v2(cublas, &stream));

        query_layer->compute(batch_size * seq_length, in_gpu, query_layer_out);
        key_layer->compute(batch_size * seq_length, in_gpu, key_layer_out);
        value_layer->compute(batch_size * seq_length, in_gpu, value_layer_out);

        CUDA_CHECK(cudaMemcpyAsync(
                attention_scores, neg_attention_mask,
                sizeof(float) * batch_size * num_attention_heads * seq_length * seq_length,
                cudaMemcpyDeviceToDevice,
                stream));

        bqk->compute(batch_size);
        softmax->compute_(batch_size * num_attention_heads * seq_length, attention_scores);

        bqkv->compute(batch_size);
    }
}
