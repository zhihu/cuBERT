//
// Created by 田露 on 2019/1/26.
//

#include "BertBatchMatMul.h"
#include "cuBERT/common.h"

cuBERT::BertQK::BertQK(cublasHandle_t handle,
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

    CUDA_CHECK(cudaMalloc(&query_array_gpu, sizeof(float *) * max_batch_size * num_attention_heads));
    CUDA_CHECK(cudaMalloc(&key_array_gpu, sizeof(float *) * max_batch_size * num_attention_heads));
    CUDA_CHECK(cudaMalloc(&out_array_gpu, sizeof(float *) * max_batch_size * num_attention_heads));

    float *query_array[max_batch_size * num_attention_heads];
    float *key_array[max_batch_size * num_attention_heads];
    float *out_array[max_batch_size * num_attention_heads];
    for (int b = 0; b < max_batch_size; ++b) {
        for (int h = 0; h < num_attention_heads; ++h) {
            size_t idx = b * seq_length * size_per_head * num_attention_heads + size_per_head * h;
            query_array[b * num_attention_heads + h] = query + idx;
            key_array[b * num_attention_heads + h] = key + idx;
            out_array[b * num_attention_heads + h] = out + b * seq_length * seq_length * num_attention_heads + seq_length * h;
        }
    }
    CUDA_CHECK(cudaMemcpy(query_array_gpu, query_array, sizeof(float *) * max_batch_size * num_attention_heads, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(key_array_gpu, key_array, sizeof(float *) * max_batch_size * num_attention_heads, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_array_gpu, out_array, sizeof(float *) * max_batch_size * num_attention_heads, cudaMemcpyHostToDevice));
}

cuBERT::BertQK::~BertQK() {
    CUDA_CHECK(cudaFree(out_array_gpu));
    CUDA_CHECK(cudaFree(key_array_gpu));
    CUDA_CHECK(cudaFree(query_array_gpu));
}

void cuBERT::BertQK::compute(size_t batch_size) {
    CUBLAS_CHECK(cublasSgemmBatched(handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N, seq_length, seq_length, size_per_head,
                                    &alpha,
                                    key_array_gpu, size_per_head * num_attention_heads,
                                    query_array_gpu, size_per_head * num_attention_heads,
                                    &beta,
                                    out_array_gpu, seq_length * num_attention_heads,
                                    num_attention_heads * batch_size));

    //  ** On entry to SGEMM  parameter number 15 had an illegal value
//    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
//                                           CUBLAS_OP_T, CUBLAS_OP_N, seq_length, seq_length, size_per_head,
//                                           &alpha,
//                                           key, size_per_head * num_attention_heads, size_per_head,
//                                           query, size_per_head * num_attention_heads, size_per_head,
//                                           &beta,
//                                           out, seq_length * num_attention_heads, seq_length,
//                                           num_attention_heads * batch_size));
}

cuBERT::BertQKV::BertQKV(cublasHandle_t handle,
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

    CUDA_CHECK(cudaMalloc(&qk_array_gpu, sizeof(float *) * max_batch_size * num_attention_heads));
    CUDA_CHECK(cudaMalloc(&value_array_gpu, sizeof(float *) * max_batch_size * num_attention_heads));
    CUDA_CHECK(cudaMalloc(&out_array_gpu, sizeof(float *) * max_batch_size * num_attention_heads));

    float *qk_array[max_batch_size * num_attention_heads];
    float *value_array[max_batch_size * num_attention_heads];
    float *out_array[max_batch_size * num_attention_heads];
    for (int b = 0; b < max_batch_size; ++b) {
        for (int h = 0; h < num_attention_heads; ++h) {
            size_t idx = b * seq_length * size_per_head * num_attention_heads + size_per_head * h;
            qk_array[b * num_attention_heads + h] = qk + b * seq_length * seq_length * num_attention_heads + seq_length * h;
            value_array[b * num_attention_heads + h] = value + idx;
            out_array[b * num_attention_heads + h] = out + idx;
        }
    }
    CUDA_CHECK(cudaMemcpy(qk_array_gpu, qk_array, sizeof(float *) * max_batch_size * num_attention_heads, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(value_array_gpu, value_array, sizeof(float *) * max_batch_size * num_attention_heads, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_array_gpu, out_array, sizeof(float *) * max_batch_size * num_attention_heads, cudaMemcpyHostToDevice));
}

cuBERT::BertQKV::~BertQKV() {
    CUDA_CHECK(cudaFree(out_array_gpu));
    CUDA_CHECK(cudaFree(value_array_gpu));
    CUDA_CHECK(cudaFree(qk_array_gpu));
}

void cuBERT::BertQKV::compute(size_t batch_size) {
    CUBLAS_CHECK(cublasSgemmBatched(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N, size_per_head, seq_length, seq_length,
                                    &alpha,
                                    value_array_gpu, size_per_head * num_attention_heads,
                                    qk_array_gpu, seq_length * num_attention_heads,
                                    &beta,
                                    out_array_gpu, size_per_head * num_attention_heads,
                                    num_attention_heads * batch_size));
}
