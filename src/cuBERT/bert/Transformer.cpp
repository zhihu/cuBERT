//
// Created by 田露 on 2019/1/21.
//
#include <string>

#include "Transformer.h"

namespace cuBERT {
    Transformer::Transformer(cublasHandle_t cublas, cudnnHandle_t cudnn,
                             const std::string &var_prefix,
                             const std::unordered_map<std::string, float *> &var,
                             size_t max_batch_size,
                             size_t seq_length,
                             size_t hidden_size,
                             size_t num_hidden_layers,
                             size_t num_attention_heads,
                             size_t intermediate_size)
            : attention_self(num_hidden_layers),
              attention_output_dense(num_hidden_layers),
              attention_output_norm(num_hidden_layers),
              intermediate_dense(num_hidden_layers),
              intermediate_act_fn(num_hidden_layers),
              output_dense(num_hidden_layers),
              output_layer_norm(num_hidden_layers),
              attention_heads(num_hidden_layers),
              attention_output(num_hidden_layers),
              intermediate_output(num_hidden_layers),
              layer_output(num_hidden_layers) {
        this->cublas = cublas;
        this->cudnn = cudnn;
        this->num_hidden_layers = num_hidden_layers;
        this->seq_length = seq_length;
        this->intermediate_size = intermediate_size;

        size_t attention_head_size = hidden_size / num_attention_heads;

        this->attention_mask = new AttentionMask(cublas, seq_length, num_attention_heads, max_batch_size);
        cudaMalloc(&this->neg_attention_mask_buffer,
                   sizeof(float) * max_batch_size * num_attention_heads * seq_length * seq_length);

        for (int layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            attention_self[layer_idx] = new AttentionSelf(cublas, cudnn,
                                                          var_prefix + "/layer_" + std::to_string(layer_idx) +
                                                          "/attention/self",
                                                          var,
                                                          max_batch_size,
                                                          seq_length,
                                                          hidden_size, num_attention_heads, attention_head_size);

            float *attention_output_dense_kernel = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/attention/output/dense/kernel");
            float *attention_output_dense_bias = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/attention/output/dense/bias");
            attention_output_dense[layer_idx] = new Dense(cublas,
                                                          hidden_size, hidden_size,
                                                          attention_output_dense_kernel, attention_output_dense_bias,
                                                          max_batch_size * seq_length);

            float *attention_output_norm_beta = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/attention/output/LayerNorm/beta");
            float *attention_output_norm_gamma = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/attention/output/LayerNorm/gamma");
            attention_output_norm[layer_idx] = new LayerNorm(hidden_size,
                                                             attention_output_norm_beta, attention_output_norm_gamma);

            float *intermediate_dense_kernel = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/intermediate/dense/kernel");
            float *intermediate_dense_bias = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/intermediate/dense/bias");
            intermediate_dense[layer_idx] = new Dense(cublas,
                                                      hidden_size, intermediate_size,
                                                      intermediate_dense_kernel, intermediate_dense_bias,
                                                      max_batch_size * seq_length);
            intermediate_act_fn[layer_idx] = new GELU();

            float *output_dense_kernel = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/output/dense/kernel");
            float *output_dense_bias = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/output/dense/bias");
            output_dense[layer_idx] = new Dense(cublas,
                                                intermediate_size, hidden_size,
                                                output_dense_kernel, output_dense_bias,
                                                max_batch_size * seq_length);

            float *output_norm_beta = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/output/LayerNorm/beta");
            float *output_norm_gamma = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/output/LayerNorm/gamma");
            output_layer_norm[layer_idx] = new LayerNorm(hidden_size,
                                                         output_norm_beta, output_norm_gamma);

            // buffers
            cudaMalloc(&attention_heads[layer_idx], sizeof(float) * max_batch_size * seq_length * hidden_size);
            cudaMalloc(&attention_output[layer_idx], sizeof(float) * max_batch_size * seq_length * hidden_size);
            cudaMalloc(&intermediate_output[layer_idx],
                       sizeof(float) * max_batch_size * seq_length * intermediate_size);
            cudaMalloc(&layer_output[layer_idx], sizeof(float) * max_batch_size * seq_length * hidden_size);
        }
    }

    Transformer::~Transformer() {
        for (int i = 0; i < num_hidden_layers; ++i) {
            cudaFree(layer_output[i]);
            cudaFree(intermediate_output[i]);
            cudaFree(attention_output[i]);
            cudaFree(attention_heads[i]);

            delete output_layer_norm[i];
            delete output_dense[i];
            delete intermediate_act_fn[i];
            delete intermediate_dense[i];
            delete attention_output_norm[i];
            delete attention_output_dense[i];
            delete attention_self[i];
        }

        cudaFree(neg_attention_mask_buffer);
        delete attention_mask;
    }

    float *Transformer::compute(size_t batch_size, float *input_gpu, char *attention_mask) {
        cudaStream_t stream = nullptr;
        cublasGetStream_v2(cublas, &stream);

        // broadcast neg_attention_mask
        this->attention_mask->compute(batch_size, attention_mask, neg_attention_mask_buffer);

        float *prev_output = input_gpu;

        for (int i = 0; i < num_hidden_layers; ++i) {
            float *layer_input = prev_output;

            // attention/self
            attention_self[i]->compute(batch_size, layer_input, neg_attention_mask_buffer, attention_heads[i]);

            // attention/output
            attention_output_dense[i]->compute(batch_size * seq_length, attention_heads[i], attention_output[i]);
            attention_output_norm[i]->compute_(batch_size * seq_length, layer_input, attention_output[i], stream);

            // intermediate
            intermediate_dense[i]->compute(batch_size * seq_length, attention_output[i], intermediate_output[i]);
            intermediate_act_fn[i]->compute_(batch_size * seq_length * intermediate_size, intermediate_output[i],
                                             stream);

            // output
            output_dense[i]->compute(batch_size * seq_length, intermediate_output[i], layer_output[i]);
            output_layer_norm[i]->compute_(batch_size * seq_length, attention_output[i], layer_output[i], stream);

            prev_output = layer_output[i];
        }

        return prev_output;
    }
}
