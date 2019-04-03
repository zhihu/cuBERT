#include "BatchMatMul.h"
#include "cuBERT/common.h"

template <typename T>
cuBERT::Att_Q_K<T>::Att_Q_K(void* handle,
                            size_t max_batch_size,
                            size_t seq_length, size_t num_attention_heads, size_t size_per_head,
                            T* query, T* key, T* out,
                            float alpha, float beta) {
    this->handle = handle;
    this->seq_length = seq_length;
    this->num_attention_heads = num_attention_heads;
    this->size_per_head = size_per_head;

    this->alpha = alpha;
    this->beta = beta;

    this->query_array = static_cast<const T **>(cuBERT::malloc(sizeof(T *) * max_batch_size * num_attention_heads));
    this->key_array = static_cast<const T **>(cuBERT::malloc(sizeof(T *) * max_batch_size * num_attention_heads));
    this->out_array = static_cast<T **>(cuBERT::malloc(sizeof(T *) * max_batch_size * num_attention_heads));

    const T **query_array_cpu = this->query_array;
    const T **key_array_cpu = this->key_array;
    T **out_array_cpu = this->out_array;
#ifdef HAVE_CUDA
    query_array_cpu = new const T*[max_batch_size * num_attention_heads];
    key_array_cpu = new const T*[max_batch_size * num_attention_heads];
    out_array_cpu = new T*[max_batch_size * num_attention_heads];
#endif
    for (int b = 0; b < max_batch_size; ++b) {
        for (int h = 0; h < num_attention_heads; ++h) {
            size_t idx = b * seq_length * size_per_head * num_attention_heads + size_per_head * h;
            query_array_cpu[b * num_attention_heads + h] = query + idx;
            key_array_cpu[b * num_attention_heads + h] = key + idx;
            out_array_cpu[b * num_attention_heads + h] = out + b * seq_length * seq_length * num_attention_heads + seq_length * h;
        }
    }
#ifdef HAVE_CUDA
    cuBERT::memcpy(query_array, query_array_cpu, sizeof(T *) * max_batch_size * num_attention_heads, 1);
    cuBERT::memcpy(key_array, key_array_cpu, sizeof(T *) * max_batch_size * num_attention_heads, 1);
    cuBERT::memcpy(out_array, out_array_cpu, sizeof(T *) * max_batch_size * num_attention_heads, 1);
    delete []out_array_cpu;
    delete []key_array_cpu;
    delete []query_array_cpu;
#endif
}

template <typename T>
cuBERT::Att_Q_K<T>::~Att_Q_K() {
    cuBERT::free(out_array);
    cuBERT::free(key_array);
    cuBERT::free(query_array);
}

template <typename T>
void cuBERT::Att_Q_K<T>::compute(size_t batch_size) {
    cuBERT::blas_gemm_batch<T>(handle,
                             true, false,
                             seq_length, seq_length, size_per_head,
                             alpha,
                             key_array, size_per_head * num_attention_heads,
                             query_array, size_per_head * num_attention_heads,
                             beta,
                             out_array, seq_length * num_attention_heads,
                             num_attention_heads * batch_size);
}

template class cuBERT::Att_Q_K<float>;
#ifdef HAVE_CUDA
template class cuBERT::Att_Q_K<half>;
#endif


template <typename T>
cuBERT::Att_QK_V<T>::Att_QK_V(void* handle,
                              size_t max_batch_size,
                              size_t seq_length, size_t num_attention_heads, size_t size_per_head,
                              T *qk, T *value, T *out,
                              float alpha, float beta) {
    this->handle = handle;
    this->seq_length = seq_length;
    this->num_attention_heads = num_attention_heads;
    this->size_per_head = size_per_head;

    this->alpha = alpha;
    this->beta = beta;

    this->qk_array = static_cast<const T **>(cuBERT::malloc(sizeof(T *) * max_batch_size * num_attention_heads));
    this->value_array = static_cast<const T **>(cuBERT::malloc(sizeof(T *) * max_batch_size * num_attention_heads));
    this->out_array = static_cast<T **>(cuBERT::malloc(sizeof(T *) * max_batch_size * num_attention_heads));

    const T **qk_array_cpu = this->qk_array;
    const T **value_array_cpu = this->value_array;
    T **out_array_cpu = this->out_array;
#ifdef HAVE_CUDA
    qk_array_cpu = new const T*[max_batch_size * num_attention_heads];
    value_array_cpu = new const T*[max_batch_size * num_attention_heads];
    out_array_cpu = new T*[max_batch_size * num_attention_heads];
#endif
    for (int b = 0; b < max_batch_size; ++b) {
        for (int h = 0; h < num_attention_heads; ++h) {
            size_t idx = b * seq_length * size_per_head * num_attention_heads + size_per_head * h;
            qk_array_cpu[b * num_attention_heads + h] = qk + b * seq_length * seq_length * num_attention_heads + seq_length * h;
            value_array_cpu[b * num_attention_heads + h] = value + idx;
            out_array_cpu[b * num_attention_heads + h] = out + idx;
        }
    }
#ifdef HAVE_CUDA
    cuBERT::memcpy(qk_array, qk_array_cpu, sizeof(T *) * max_batch_size * num_attention_heads, 1);
    cuBERT::memcpy(value_array, value_array_cpu, sizeof(T *) * max_batch_size * num_attention_heads, 1);
    cuBERT::memcpy(out_array, out_array_cpu, sizeof(T *) * max_batch_size * num_attention_heads, 1);
    delete []out_array_cpu;
    delete []value_array_cpu;
    delete []qk_array_cpu;
#endif
}

template <typename T>
cuBERT::Att_QK_V<T>::~Att_QK_V() {
    cuBERT::free(out_array);
    cuBERT::free(value_array);
    cuBERT::free(qk_array);
}

template <typename T>
void cuBERT::Att_QK_V<T>::compute(size_t batch_size) {
    cuBERT::blas_gemm_batch<T>(handle,
                             false, false,
                             size_per_head, seq_length, seq_length,
                             alpha,
                             value_array, size_per_head * num_attention_heads,
                             qk_array, seq_length * num_attention_heads,
                             beta,
                             out_array, size_per_head * num_attention_heads,
                             num_attention_heads * batch_size);
}

template class cuBERT::Att_QK_V<float>;
#ifdef HAVE_CUDA
template class cuBERT::Att_QK_V<half>;
#endif
