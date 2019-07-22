#include <string>

#include "cuBERT/common.h"
#include "Transformer.h"

namespace cuBERT {
    template <typename T>
    Transformer<T>::Transformer(void* cublas,
                             const std::string &var_prefix,
                             const std::unordered_map<std::string, T *> &var,
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
        this->num_hidden_layers = num_hidden_layers;
        this->seq_length = seq_length;
        this->intermediate_size = intermediate_size;

        size_t attention_head_size = hidden_size / num_attention_heads;

        this->attention_mask = new AttentionMask<T >(cublas, seq_length, num_attention_heads, max_batch_size);
        this->neg_attention_mask_buffer = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * num_attention_heads * seq_length * seq_length));

        for (int layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            // buffers
            this->attention_heads[layer_idx] = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * seq_length * hidden_size));
            this->attention_output[layer_idx] = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * seq_length * hidden_size));
            this->intermediate_output[layer_idx] = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * seq_length * intermediate_size));
            this->layer_output[layer_idx] = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * seq_length * hidden_size));

            attention_self[layer_idx] = new AttentionSelf<T>(cublas,
                                                              var_prefix + "/layer_" + std::to_string(layer_idx) +
                                                              "/attention/self",
                                                              var,
                                                              max_batch_size,
                                                              seq_length,
                                                              attention_heads[layer_idx],
                                                              hidden_size, num_attention_heads, attention_head_size);

            T *attention_output_dense_kernel = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/attention/output/dense/kernel");
            T *attention_output_dense_bias = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/attention/output/dense/bias");
            attention_output_dense[layer_idx] = new Dense<T>(cublas,
                                                          hidden_size, hidden_size,
                                                          attention_output_dense_kernel, attention_output_dense_bias,
                                                          max_batch_size * seq_length,
                                                          gemm_algo<T>("GEMM_ALGO_ATTENTION"));

            T *attention_output_norm_beta = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/attention/output/LayerNorm/beta");
            T *attention_output_norm_gamma = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/attention/output/LayerNorm/gamma");
            attention_output_norm[layer_idx] = new LayerNorm<T>(max_batch_size * seq_length, hidden_size,
                                                                    attention_output_norm_beta, attention_output_norm_gamma);

            // inputs = hidden_size
            // units = intermediate_size
            // max_batch_size = max_batch_size * seq_length
            T *intermediate_dense_kernel = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/intermediate/dense/kernel");
            T *intermediate_dense_bias = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/intermediate/dense/bias");
            intermediate_dense[layer_idx] = new Dense<T>(cublas,
                                                      hidden_size, intermediate_size,
                                                      intermediate_dense_kernel, intermediate_dense_bias,
                                                      max_batch_size * seq_length,
                                                      gemm_algo<T>("GEMM_ALGO_INTERMEDIATE"));
            intermediate_act_fn[layer_idx] = new GELU<T>();

            // inputs = intermediate_size
            // units = hidden_size
            // max_batch_size = max_batch_size * seq_length
            T *output_dense_kernel = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/output/dense/kernel");
            T *output_dense_bias = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/output/dense/bias");
            output_dense[layer_idx] = new Dense<T>(cublas,
                                                intermediate_size, hidden_size,
                                                output_dense_kernel, output_dense_bias,
                                                max_batch_size * seq_length,
                                                gemm_algo<T>("GEMM_ALGO_OUTPUT"));

            T *output_norm_beta = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/output/LayerNorm/beta");
            T *output_norm_gamma = var.at(
                    var_prefix + "/layer_" + std::to_string(layer_idx) + "/output/LayerNorm/gamma");
            output_layer_norm[layer_idx] = new LayerNorm<T>(max_batch_size * seq_length, hidden_size,
                                                                output_norm_beta, output_norm_gamma);
        }
    }

    template <typename T>
    Transformer<T>::~Transformer() {
        for (int i = 0; i < num_hidden_layers; ++i) {
            delete output_layer_norm[i];
            delete output_dense[i];
            delete intermediate_act_fn[i];
            delete intermediate_dense[i];
            delete attention_output_norm[i];
            delete attention_output_dense[i];
            delete attention_self[i];

            cuBERT::free(layer_output[i]);
            cuBERT::free(intermediate_output[i]);
            cuBERT::free(attention_output[i]);
            cuBERT::free(attention_heads[i]);
        }

        cuBERT::free(neg_attention_mask_buffer);
        delete attention_mask;
    }

    template <typename T>
    T *Transformer<T>::compute(size_t batch_size, T *input_gpu, int8_t *attention_mask) {
        _pre_compute(batch_size);
        return _in_compute(batch_size, input_gpu, attention_mask);
    }

    template <typename T>
    void Transformer<T>::_pre_compute(size_t batch_size) {
        for (int i = 0; i < num_hidden_layers; ++i) {
            attention_self[i]->_pre_compute(batch_size);
            attention_output_dense[i]->_pre_compute(batch_size * seq_length, attention_output[i]);
            intermediate_dense[i]->_pre_compute(batch_size * seq_length, intermediate_output[i]);
            output_dense[i]->_pre_compute(batch_size * seq_length, layer_output[i]);
        }
    }

    template <typename T>
    T *Transformer<T>::_in_compute(size_t batch_size, T *input_gpu, int8_t *attention_mask) {
        void* stream = cuBERT::blas_get_stream(cublas);

        // broadcast neg_attention_mask
        this->attention_mask->compute(batch_size, attention_mask, neg_attention_mask_buffer);

        T *prev_output = input_gpu;

        for (int i = 0; i < num_hidden_layers; ++i) {
            T *layer_input = prev_output;

            // attention/self
            attention_self[i]->_in_compute(batch_size, layer_input, neg_attention_mask_buffer);

            // attention/output
            attention_output_dense[i]->_in_compute(batch_size * seq_length, attention_heads[i], attention_output[i]);
            attention_output_norm[i]->compute_(batch_size * seq_length, layer_input, attention_output[i], stream);

            // intermediate
            intermediate_dense[i]->_in_compute(batch_size * seq_length, attention_output[i], intermediate_output[i]);
            intermediate_act_fn[i]->compute_(batch_size * seq_length * intermediate_size, intermediate_output[i], stream);

            // output
            output_dense[i]->_in_compute(batch_size * seq_length, intermediate_output[i], layer_output[i]);
            output_layer_norm[i]->compute_(batch_size * seq_length, attention_output[i], layer_output[i], stream);

            prev_output = layer_output[i];
        }

        return prev_output;
    }

    template class Transformer<float>;
#ifdef HAVE_CUDA
    template class Transformer<half>;
#endif
}
