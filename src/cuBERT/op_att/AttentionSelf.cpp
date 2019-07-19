#include <cmath>

#include "cuBERT/common.h"
#include "AttentionSelf.h"

namespace cuBERT {
    template <typename T>
    AttentionSelf<T>::AttentionSelf(void* cublas,
                                 const std::string &var_prefix,
                                 const std::unordered_map<std::string, T *> &var,
                                 size_t max_batch_size,
                                 size_t seq_length,
                                 T *context_layer_out,
                                 size_t width, size_t num_attention_heads, size_t size_per_head) {
        this->cublas = cublas;
        this->seq_length = seq_length;
        this->num_attention_heads = num_attention_heads;
        this->size_per_head = size_per_head;

        this->context_layer_out = context_layer_out;

        // inputs = hidden_size
        // units = hidden_size
        // max_batch_size = max_batch_size * seq_length
        int gemm_algo_attention = gemm_algo<T>("GEMM_ALGO_ATTENTION");

        T *query_layer_kernel = var.at(var_prefix + "/query/kernel");
        T *query_layer_bias = var.at(var_prefix + "/query/bias");
        query_layer = new Dense<T>(cublas,
                                width, num_attention_heads * size_per_head,
                                query_layer_kernel, query_layer_bias,
                                max_batch_size * seq_length,
                                gemm_algo_attention);

        T *key_layer_kernel = var.at(var_prefix + "/key/kernel");
        T *key_layer_bias = var.at(var_prefix + "/key/bias");
        key_layer = new Dense<T>(cublas,
                              width, num_attention_heads * size_per_head,
                              key_layer_kernel, key_layer_bias,
                              max_batch_size * seq_length,
                              gemm_algo_attention);

        T *value_layer_kernel = var.at(var_prefix + "/value/kernel");
        T *value_layer_bias = var.at(var_prefix + "/value/bias");
        value_layer = new Dense<T>(cublas,
                                width, num_attention_heads * size_per_head,
                                value_layer_kernel, value_layer_bias,
                                max_batch_size * seq_length,
                                gemm_algo_attention);

        softmax = new Softmax<T>(max_batch_size * num_attention_heads * seq_length, seq_length);

        this->query_layer_out = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * seq_length * num_attention_heads * size_per_head));
        this->key_layer_out = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * seq_length * num_attention_heads * size_per_head));
        this->value_layer_out = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * seq_length * num_attention_heads * size_per_head));
        this->attention_scores = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * num_attention_heads * seq_length * seq_length));

        bqk = new Att_Q_K<T>(cublas, max_batch_size, seq_length, num_attention_heads, size_per_head,
                         query_layer_out, key_layer_out, attention_scores,
                         1.0 / std::sqrt(size_per_head), -10000.0f);

        bqkv = new Att_QK_V<T>(cublas, max_batch_size, seq_length, num_attention_heads, size_per_head,
                           attention_scores, value_layer_out, context_layer_out);
    }

    template <typename T>
    AttentionSelf<T>::~AttentionSelf() {
        delete bqkv;
        delete bqk;

        cuBERT::free(attention_scores);
        cuBERT::free(value_layer_out);
        cuBERT::free(key_layer_out);
        cuBERT::free(query_layer_out);

        delete softmax;
        delete value_layer;
        delete key_layer;
        delete query_layer;
    }

    template <typename T>
    void AttentionSelf<T>::compute(size_t batch_size, T *in_gpu, T *neg_attention_mask) {
        _pre_compute(batch_size);
        _in_compute(batch_size, in_gpu, neg_attention_mask);
    }

    template <typename T>
    void AttentionSelf<T>::_pre_compute(size_t batch_size) {
        query_layer->_pre_compute(batch_size * seq_length, query_layer_out);
        key_layer->_pre_compute(batch_size * seq_length, key_layer_out);
        value_layer->_pre_compute(batch_size * seq_length, value_layer_out);
    }

    template <typename T>
    void AttentionSelf<T>::_in_compute(size_t batch_size, T *in_gpu, T *neg_attention_mask) {
        void *stream = cuBERT::blas_get_stream(cublas);

        cuBERT::memcpyAsync(attention_scores, neg_attention_mask,
                            sizeof(T) * batch_size * num_attention_heads * seq_length * seq_length,
                            3, stream);

        query_layer->_in_compute(batch_size * seq_length, in_gpu, query_layer_out);
        key_layer->_in_compute(batch_size * seq_length, in_gpu, key_layer_out);
        value_layer->_in_compute(batch_size * seq_length, in_gpu, value_layer_out);

        bqk->compute(batch_size);
        softmax->compute_(batch_size * num_attention_heads * seq_length, attention_scores, stream);

        bqkv->compute(batch_size);
    }

    template class AttentionSelf<float>;
#ifdef HAVE_CUDA
    template class AttentionSelf<half>;
#endif
}
