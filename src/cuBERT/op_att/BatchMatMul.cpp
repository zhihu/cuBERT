#include "BatchMatMul.h"
#include "cuBERT/common.h"

cuBERT::Att_Q_K::Att_Q_K(void* handle,
                       size_t max_batch_size,
                       size_t seq_length, size_t num_attention_heads, size_t size_per_head,
                       float* query, float* key, float* out,
                       float alpha, float beta) {
    this->handle = handle;
    this->seq_length = seq_length;
    this->num_attention_heads = num_attention_heads;
    this->size_per_head = size_per_head;

    this->alpha = alpha;
    this->beta = beta;

    this->query_array = static_cast<const float **>(cuBERT::malloc(sizeof(float *) * max_batch_size * num_attention_heads));
    this->key_array = static_cast<const float **>(cuBERT::malloc(sizeof(float *) * max_batch_size * num_attention_heads));
    this->out_array = static_cast<float **>(cuBERT::malloc(sizeof(float *) * max_batch_size * num_attention_heads));

    const float **query_array_cpu = this->query_array;
    const float **key_array_cpu = this->key_array;
    float **out_array_cpu = this->out_array;
    if (cuBERT::gpu()) {
        query_array_cpu = new const float*[max_batch_size * num_attention_heads];
        key_array_cpu = new const float*[max_batch_size * num_attention_heads];
        out_array_cpu = new float*[max_batch_size * num_attention_heads];
    }
    for (int b = 0; b < max_batch_size; ++b) {
        for (int h = 0; h < num_attention_heads; ++h) {
            size_t idx = b * seq_length * size_per_head * num_attention_heads + size_per_head * h;
            query_array_cpu[b * num_attention_heads + h] = query + idx;
            key_array_cpu[b * num_attention_heads + h] = key + idx;
            out_array_cpu[b * num_attention_heads + h] = out + b * seq_length * seq_length * num_attention_heads + seq_length * h;
        }
    }
    if (cuBERT::gpu()) {
        cuBERT::memcpy(query_array, query_array_cpu, sizeof(float *) * max_batch_size * num_attention_heads, 1);
        cuBERT::memcpy(key_array, key_array_cpu, sizeof(float *) * max_batch_size * num_attention_heads, 1);
        cuBERT::memcpy(out_array, out_array_cpu, sizeof(float *) * max_batch_size * num_attention_heads, 1);

        delete []out_array_cpu;
        delete []key_array_cpu;
        delete []query_array_cpu;
    }
}

cuBERT::Att_Q_K::~Att_Q_K() {
    cuBERT::free(out_array);
    cuBERT::free(key_array);
    cuBERT::free(query_array);
}

void cuBERT::Att_Q_K::compute(size_t batch_size) {
    cuBERT::blas_sgemm_batch(handle,
                             true, false,
                             seq_length, seq_length, size_per_head,
                             alpha,
                             key_array, size_per_head * num_attention_heads,
                             query_array, size_per_head * num_attention_heads,
                             beta,
                             out_array, seq_length * num_attention_heads,
                             num_attention_heads * batch_size);
}


cuBERT::Att_QK_V::Att_QK_V(void* handle,
                         size_t max_batch_size,
                         size_t seq_length, size_t num_attention_heads, size_t size_per_head,
                         float *qk, float *value, float *out,
                         float alpha, float beta) {
    this->handle = handle;
    this->seq_length = seq_length;
    this->num_attention_heads = num_attention_heads;
    this->size_per_head = size_per_head;

    this->alpha = alpha;
    this->beta = beta;

    this->qk_array = static_cast<const float **>(cuBERT::malloc(sizeof(float *) * max_batch_size * num_attention_heads));
    this->value_array = static_cast<const float **>(cuBERT::malloc(sizeof(float *) * max_batch_size * num_attention_heads));
    this->out_array = static_cast<float **>(cuBERT::malloc(sizeof(float *) * max_batch_size * num_attention_heads));

    const float **qk_array_cpu = this->qk_array;
    const float **value_array_cpu = this->value_array;
    float **out_array_cpu = this->out_array;
    if (cuBERT::gpu()) {
        qk_array_cpu = new const float*[max_batch_size * num_attention_heads];
        value_array_cpu = new const float*[max_batch_size * num_attention_heads];
        out_array_cpu = new float*[max_batch_size * num_attention_heads];
    }
    for (int b = 0; b < max_batch_size; ++b) {
        for (int h = 0; h < num_attention_heads; ++h) {
            size_t idx = b * seq_length * size_per_head * num_attention_heads + size_per_head * h;
            qk_array_cpu[b * num_attention_heads + h] = qk + b * seq_length * seq_length * num_attention_heads + seq_length * h;
            value_array_cpu[b * num_attention_heads + h] = value + idx;
            out_array_cpu[b * num_attention_heads + h] = out + idx;
        }
    }
    if (cuBERT::gpu()) {
        cuBERT::memcpy(qk_array, qk_array_cpu, sizeof(float *) * max_batch_size * num_attention_heads, 1);
        cuBERT::memcpy(value_array, value_array_cpu, sizeof(float *) * max_batch_size * num_attention_heads, 1);
        cuBERT::memcpy(out_array, out_array_cpu, sizeof(float *) * max_batch_size * num_attention_heads, 1);

        delete []out_array_cpu;
        delete []value_array_cpu;
        delete []qk_array_cpu;
    }
}

cuBERT::Att_QK_V::~Att_QK_V() {
    cuBERT::free(out_array);
    cuBERT::free(value_array);
    cuBERT::free(qk_array);
}

void cuBERT::Att_QK_V::compute(size_t batch_size) {
    cuBERT::blas_sgemm_batch(handle,
                             false, false,
                             size_per_head, seq_length, seq_length,
                             alpha,
                             value_array, size_per_head * num_attention_heads,
                             qk_array, seq_length * num_attention_heads,
                             beta,
                             out_array, size_per_head * num_attention_heads,
                             num_attention_heads * batch_size);
}
