#include <mkl.h>

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

    this->query_array_cpu = new const float*[max_batch_size * num_attention_heads];
    this->key_array_cpu = new const float*[max_batch_size * num_attention_heads];
    this->out_array_cpu = new float*[max_batch_size * num_attention_heads];
    for (int b = 0; b < max_batch_size; ++b) {
        for (int h = 0; h < num_attention_heads; ++h) {
            size_t idx = b * seq_length * size_per_head * num_attention_heads + size_per_head * h;
            query_array_cpu[b * num_attention_heads + h] = query + idx;
            key_array_cpu[b * num_attention_heads + h] = key + idx;
            out_array_cpu[b * num_attention_heads + h] = out + b * seq_length * seq_length * num_attention_heads + seq_length * h;
        }
    }

    CUDA_CHECK(cudaMalloc(&query_array_gpu, sizeof(float *) * max_batch_size * num_attention_heads));
    CUDA_CHECK(cudaMalloc(&key_array_gpu, sizeof(float *) * max_batch_size * num_attention_heads));
    CUDA_CHECK(cudaMalloc(&out_array_gpu, sizeof(float *) * max_batch_size * num_attention_heads));

    CUDA_CHECK(cudaMemcpy(query_array_gpu, query_array_cpu, sizeof(float *) * max_batch_size * num_attention_heads, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(key_array_gpu, key_array_cpu, sizeof(float *) * max_batch_size * num_attention_heads, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_array_gpu, out_array_cpu, sizeof(float *) * max_batch_size * num_attention_heads, cudaMemcpyHostToDevice));
}

cuBERT::BertQK::~BertQK() {
    CUDA_CHECK(cudaFree(out_array_gpu));
    CUDA_CHECK(cudaFree(key_array_gpu));
    CUDA_CHECK(cudaFree(query_array_gpu));

    delete []out_array_cpu;
    delete []key_array_cpu;
    delete []query_array_cpu;
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
}

void cuBERT::BertQK::compute_cpu(size_t batch_size) {
    CBLAS_TRANSPOSE transA = CblasTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;
    int m = seq_length;
    int n = seq_length;
    int k = size_per_head;
    int lda = size_per_head * num_attention_heads;
    int ldb = size_per_head * num_attention_heads;
    int ldc = seq_length * num_attention_heads;
    int group_size = num_attention_heads * batch_size;

    cblas_sgemm_batch(CblasColMajor,
                      &transA, &transB, &m, &n, &k,
                      &alpha,
                      key_array_cpu, &lda,
                      query_array_cpu, &ldb,
                      &beta,
                      out_array_cpu, &ldc,
                      1, &group_size);
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

    this->qk_array_cpu = new const float*[max_batch_size * num_attention_heads];
    this->value_array_cpu = new const float*[max_batch_size * num_attention_heads];
    this->out_array_cpu = new float*[max_batch_size * num_attention_heads];
    for (int b = 0; b < max_batch_size; ++b) {
        for (int h = 0; h < num_attention_heads; ++h) {
            size_t idx = b * seq_length * size_per_head * num_attention_heads + size_per_head * h;
            qk_array_cpu[b * num_attention_heads + h] = qk + b * seq_length * seq_length * num_attention_heads + seq_length * h;
            value_array_cpu[b * num_attention_heads + h] = value + idx;
            out_array_cpu[b * num_attention_heads + h] = out + idx;
        }
    }

    CUDA_CHECK(cudaMalloc(&qk_array_gpu, sizeof(float *) * max_batch_size * num_attention_heads));
    CUDA_CHECK(cudaMalloc(&value_array_gpu, sizeof(float *) * max_batch_size * num_attention_heads));
    CUDA_CHECK(cudaMalloc(&out_array_gpu, sizeof(float *) * max_batch_size * num_attention_heads));

    CUDA_CHECK(cudaMemcpy(qk_array_gpu, qk_array_cpu, sizeof(float *) * max_batch_size * num_attention_heads, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(value_array_gpu, value_array_cpu, sizeof(float *) * max_batch_size * num_attention_heads, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_array_gpu, out_array_cpu, sizeof(float *) * max_batch_size * num_attention_heads, cudaMemcpyHostToDevice));
}

cuBERT::BertQKV::~BertQKV() {
    CUDA_CHECK(cudaFree(out_array_gpu));
    CUDA_CHECK(cudaFree(value_array_gpu));
    CUDA_CHECK(cudaFree(qk_array_gpu));

    delete []out_array_cpu;
    delete []value_array_cpu;
    delete []qk_array_cpu;
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

void cuBERT::BertQKV::compute_cpu(size_t batch_size) {
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;
    int m = size_per_head;
    int n = seq_length;
    int k = seq_length;
    int lda = size_per_head * num_attention_heads;
    int ldb = seq_length * num_attention_heads;
    int ldc = size_per_head * num_attention_heads;
    int group_size = num_attention_heads * batch_size;

    cblas_sgemm_batch(CblasColMajor,
                      &transA, &transB, &m, &n, &k,
                      &alpha,
                      value_array_cpu, &lda,
                      qk_array_cpu, &ldb,
                      &beta,
                      out_array_cpu, &ldc,
                      1, &group_size);
}
