#include <string>

#include "cuBERT/common.h"
#include "Transformer.h"

namespace cuBERT {
    Transformer::Transformer(cublasHandle_t cublas,
                             const std::string &var_prefix,
                             const std::unordered_map<std::string, float *> &var,
                             size_t max_batch_size,
                             size_t seq_length,
                             size_t hidden_size,
                             size_t num_hidden_layers,
                             size_t num_attention_heads,
                             size_t intermediate_size)
            : attention_self_gpu(num_hidden_layers),
              attention_self_cpu(num_hidden_layers),
              attention_output_dense(num_hidden_layers),
              attention_output_norm(num_hidden_layers),
              intermediate_dense(num_hidden_layers),
              intermediate_act_fn(num_hidden_layers),
              output_dense(num_hidden_layers),
              output_layer_norm(num_hidden_layers),
              attention_heads_gpu(num_hidden_layers),
              attention_output_gpu(num_hidden_layers),
              intermediate_output_gpu(num_hidden_layers),
              layer_output_gpu(num_hidden_layers),
              attention_heads_cpu(num_hidden_layers),
              attention_output_cpu(num_hidden_layers),
              intermediate_output_cpu(num_hidden_layers),
              layer_output_cpu(num_hidden_layers){
        this->cublas = cublas;
        this->num_hidden_layers = num_hidden_layers;
        this->seq_length = seq_length;
        this->intermediate_size = intermediate_size;

        size_t attention_head_size = hidden_size / num_attention_heads;

        this->attention_mask = new AttentionMask(cublas, seq_length, num_attention_heads, max_batch_size);

        this->neg_attention_mask_buffer_cpu = new float[max_batch_size * num_attention_heads * seq_length * seq_length];
        CUDA_CHECK(cudaMalloc(&this->neg_attention_mask_buffer_gpu, sizeof(float) * max_batch_size * num_attention_heads * seq_length * seq_length));

        for (int layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            // buffers
            this->attention_heads_cpu[layer_idx] = new float[max_batch_size * seq_length * hidden_size];
            this->attention_output_cpu[layer_idx] = new float[max_batch_size * seq_length * hidden_size];
            this->intermediate_output_cpu[layer_idx] = new float[max_batch_size * seq_length * intermediate_size];
            this->layer_output_cpu[layer_idx] = new float[max_batch_size * seq_length * hidden_size];

            CUDA_CHECK(cudaMalloc(&attention_heads_gpu[layer_idx], sizeof(float) * max_batch_size * seq_length * hidden_size));
            CUDA_CHECK(cudaMalloc(&attention_output_gpu[layer_idx], sizeof(float) * max_batch_size * seq_length * hidden_size));
            CUDA_CHECK(cudaMalloc(&intermediate_output_gpu[layer_idx], sizeof(float) * max_batch_size * seq_length * intermediate_size));
            CUDA_CHECK(cudaMalloc(&layer_output_gpu[layer_idx], sizeof(float) * max_batch_size * seq_length * hidden_size));

            attention_self_gpu[layer_idx] = new AttentionSelf(cublas,
                                                              var_prefix + "/layer_" + std::to_string(layer_idx) +
                                                              "/attention/self",
                                                              var,
                                                              max_batch_size,
                                                              seq_length,
                                                              attention_heads_gpu[layer_idx],
                                                              hidden_size, num_attention_heads, attention_head_size);
            attention_self_cpu[layer_idx] = new AttentionSelf(cublas,
                                                              var_prefix + "/layer_" + std::to_string(layer_idx) +
                                                              "/attention/self",
                                                              var,
                                                              max_batch_size,
                                                              seq_length,
                                                              attention_heads_cpu[layer_idx],
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
            attention_output_norm[layer_idx] = new LayerNorm(max_batch_size * seq_length, hidden_size,
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
            output_layer_norm[layer_idx] = new LayerNorm(max_batch_size * seq_length, hidden_size,
                                                         output_norm_beta, output_norm_gamma);
        }
    }

    Transformer::~Transformer() {
        for (int i = 0; i < num_hidden_layers; ++i) {
            delete output_layer_norm[i];
            delete output_dense[i];
            delete intermediate_act_fn[i];
            delete intermediate_dense[i];
            delete attention_output_norm[i];
            delete attention_output_dense[i];

            delete attention_self_gpu[i];
            delete attention_self_cpu[i];

            CUDA_CHECK(cudaFree(layer_output_gpu[i]));
            CUDA_CHECK(cudaFree(intermediate_output_gpu[i]));
            CUDA_CHECK(cudaFree(attention_output_gpu[i]));
            CUDA_CHECK(cudaFree(attention_heads_gpu[i]));

            delete[] layer_output_cpu[i];
            delete[] intermediate_output_cpu[i];
            delete[] attention_output_cpu[i];
            delete[] attention_heads_cpu[i];
        }

        CUDA_CHECK(cudaFree(neg_attention_mask_buffer_gpu));
        delete[] neg_attention_mask_buffer_cpu;

        delete attention_mask;
    }

    float *Transformer::compute(size_t batch_size, float *input_gpu, char *attention_mask) {
        _pre_compute(batch_size);
        return _in_compute(batch_size, input_gpu, attention_mask);
    }

    void Transformer::_pre_compute(size_t batch_size) {
        for (int i = 0; i < num_hidden_layers; ++i) {
            attention_self_gpu[i]->_pre_compute(batch_size);
            attention_output_dense[i]->_pre_compute(batch_size * seq_length, attention_output_gpu[i]);
            intermediate_dense[i]->_pre_compute(batch_size * seq_length, intermediate_output_gpu[i]);
            output_dense[i]->_pre_compute(batch_size * seq_length, layer_output_gpu[i]);
        }
    }

    float *Transformer::_in_compute(size_t batch_size, float *input_gpu, char *attention_mask) {
        cudaStream_t stream = nullptr;
        CUBLAS_CHECK(cublasGetStream_v2(cublas, &stream));

        // broadcast neg_attention_mask
        this->attention_mask->compute(batch_size, attention_mask, neg_attention_mask_buffer_gpu);

        float *prev_output = input_gpu;

        for (int i = 0; i < num_hidden_layers; ++i) {
            float *layer_input = prev_output;

            // attention/self
            attention_self_gpu[i]->_in_compute(batch_size, layer_input, neg_attention_mask_buffer_gpu);

            // attention/output
            attention_output_dense[i]->_in_compute(batch_size * seq_length, attention_heads_gpu[i], attention_output_gpu[i]);
            attention_output_norm[i]->compute_(batch_size * seq_length, layer_input, attention_output_gpu[i], stream);

            // intermediate
            intermediate_dense[i]->_in_compute(batch_size * seq_length, attention_output_gpu[i], intermediate_output_gpu[i]);
            intermediate_act_fn[i]->compute_(batch_size * seq_length * intermediate_size, intermediate_output_gpu[i], stream);

            // output
            output_dense[i]->_in_compute(batch_size * seq_length, intermediate_output_gpu[i], layer_output_gpu[i]);
            output_layer_norm[i]->compute_(batch_size * seq_length, attention_output_gpu[i], layer_output_gpu[i], stream);

            prev_output = layer_output_gpu[i];
        }

        return prev_output;
    }

    float *Transformer::compute_cpu(size_t batch_size, float *input_cpu, char *attention_mask) {
        _pre_compute_cpu(batch_size);
        return _in_compute_cpu(batch_size, input_cpu, attention_mask);
    }

    void Transformer::_pre_compute_cpu(size_t batch_size) {
        for (int i = 0; i < num_hidden_layers; ++i) {
            attention_self_cpu[i]->_pre_compute_cpu(batch_size);
            attention_output_dense[i]->_pre_compute_cpu(batch_size * seq_length, attention_output_cpu[i]);
            intermediate_dense[i]->_pre_compute_cpu(batch_size * seq_length, intermediate_output_cpu[i]);
            output_dense[i]->_pre_compute_cpu(batch_size * seq_length, layer_output_cpu[i]);
        }
    }

    float *Transformer::_in_compute_cpu(size_t batch_size, float *input_cpu, char *attention_mask) {
        this->attention_mask->compute_cpu(batch_size, attention_mask, neg_attention_mask_buffer_cpu);

        float *prev_output = input_cpu;

        for (int i = 0; i < num_hidden_layers; ++i) {
            float *layer_input = prev_output;

            // attention/self
            attention_self_cpu[i]->_in_compute_cpu(batch_size, layer_input, neg_attention_mask_buffer_cpu);

            // attention/output
            attention_output_dense[i]->_in_compute_cpu(batch_size * seq_length, attention_heads_cpu[i], attention_output_cpu[i]);
            attention_output_norm[i]->compute_cpu_(batch_size * seq_length, layer_input, attention_output_cpu[i]);

            // intermediate
            intermediate_dense[i]->_in_compute_cpu(batch_size * seq_length, attention_output_cpu[i], intermediate_output_cpu[i]);
            intermediate_act_fn[i]->compute_cpu_(batch_size * seq_length * intermediate_size, intermediate_output_cpu[i]);

            // output
            output_dense[i]->_in_compute_cpu(batch_size * seq_length, intermediate_output_cpu[i], layer_output_cpu[i]);
            output_layer_norm[i]->compute_cpu_(batch_size * seq_length, attention_output_cpu[i], layer_output_cpu[i]);

            prev_output = layer_output_cpu[i];
        }

        return prev_output;
    }
}
