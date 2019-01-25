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
                                 size_t width, size_t num_attention_heads, size_t size_per_head) {
        this->cublas = cublas;
        this->cudnn = cudnn;
        this->seq_length = seq_length;
        this->num_attention_heads = num_attention_heads;

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

        transpose_0 = new Transpose(cudnn,
                                    {-1, seq_length, num_attention_heads, size_per_head},
                                    {0, 2, 1, 3});

        bmm_0 = new BatchMatMul(cublas, false, true,
                                seq_length, seq_length, size_per_head, max_batch_size * num_attention_heads,
                                1.0 / std::sqrt(size_per_head), -10000.0f);

        softmax = new Softmax(cudnn, seq_length);

        bmm_1 = new BatchMatMul(cublas, false, false,
                                seq_length, size_per_head, seq_length, max_batch_size * num_attention_heads);

        transpose_1 = new Transpose(cudnn,
                                    {-1, num_attention_heads, seq_length, size_per_head},
                                    {0, 2, 1, 3});

        CUDA_CHECK(cudaMalloc(&query_layer_out, sizeof(float) * max_batch_size * seq_length * num_attention_heads * size_per_head));
        CUDA_CHECK(cudaMalloc(&key_layer_out, sizeof(float) * max_batch_size * seq_length * num_attention_heads * size_per_head));
        CUDA_CHECK(cudaMalloc(&value_layer_out, sizeof(float) * max_batch_size * seq_length * num_attention_heads * size_per_head));
        CUDA_CHECK(cudaMalloc(&context_layer_out, sizeof(float) * max_batch_size * seq_length * num_attention_heads * size_per_head));
        CUDA_CHECK(cudaMalloc(&query_layer_out_t, sizeof(float) * max_batch_size * seq_length * num_attention_heads * size_per_head));
        CUDA_CHECK(cudaMalloc(&key_layer_out_t, sizeof(float) * max_batch_size * seq_length * num_attention_heads * size_per_head));
        CUDA_CHECK(cudaMalloc(&value_layer_out_t, sizeof(float) * max_batch_size * seq_length * num_attention_heads * size_per_head));
        CUDA_CHECK(cudaMalloc(&attention_scores, sizeof(float) * max_batch_size * num_attention_heads * seq_length * seq_length));
    }

    AttentionSelf::~AttentionSelf() {
        CUDA_CHECK(cudaFree(attention_scores));
        CUDA_CHECK(cudaFree(value_layer_out_t));
        CUDA_CHECK(cudaFree(key_layer_out_t));
        CUDA_CHECK(cudaFree(query_layer_out_t));
        CUDA_CHECK(cudaFree(context_layer_out));
        CUDA_CHECK(cudaFree(value_layer_out));
        CUDA_CHECK(cudaFree(key_layer_out));
        CUDA_CHECK(cudaFree(query_layer_out));

        delete transpose_1;
        delete bmm_1;
        delete softmax;
        delete bmm_0;
        delete transpose_0;
        delete value_layer;
        delete key_layer;
        delete query_layer;
    }

    void AttentionSelf::compute(size_t batch_size, float *in_gpu, float *neg_attention_mask, float *out_gpu) {
        cudaStream_t stream = nullptr;
        CUBLAS_CHECK(cublasGetStream_v2(cublas, &stream));

        query_layer->compute(batch_size * seq_length, in_gpu, query_layer_out);
        key_layer->compute(batch_size * seq_length, in_gpu, key_layer_out);
        value_layer->compute(batch_size * seq_length, in_gpu, value_layer_out);

        transpose_0->compute(batch_size, query_layer_out, query_layer_out_t);
        transpose_0->compute(batch_size, key_layer_out, key_layer_out_t);
        transpose_0->compute(batch_size, value_layer_out, value_layer_out_t);

        CUDA_CHECK(cudaMemcpyAsync(
                attention_scores, neg_attention_mask,
                sizeof(float) * batch_size * num_attention_heads * seq_length * seq_length,
                cudaMemcpyDeviceToDevice,
                stream));

        bmm_0->compute(batch_size * num_attention_heads, query_layer_out_t, key_layer_out_t, attention_scores);
        softmax->compute_(batch_size * num_attention_heads * seq_length, attention_scores);

        bmm_1->compute(batch_size * num_attention_heads, attention_scores, value_layer_out_t, context_layer_out);
        transpose_1->compute(batch_size, context_layer_out, out_gpu);
    }
}
