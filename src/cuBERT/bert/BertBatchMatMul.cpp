//
// Created by 田露 on 2019/1/26.
//

#include "BertBatchMatMul.h"
#include "cuBERT/common.h"

cuBERT::BertQK::BertQK(cublasHandle_t handle,
                       size_t seq_length, size_t num_attention_heads, size_t size_per_head,
                       float alpha, float beta) {
    this->handle = handle;
    this->seq_length = seq_length;
    this->num_attention_heads = num_attention_heads;
    this->size_per_head = size_per_head;

    this->alpha = alpha;
    this->beta = beta;
}

void cuBERT::BertQK::compute(size_t batch_size, float *query, float *key, float *out) {
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
                                           CUBLAS_OP_T, CUBLAS_OP_N, seq_length, seq_length, size_per_head,
                                           &alpha,
                                           key, size_per_head * num_attention_heads, size_per_head,
                                           query, size_per_head * num_attention_heads, size_per_head,
                                           &beta,
                                           out, seq_length * num_attention_heads, seq_length,
                                           num_attention_heads * batch_size));
}
