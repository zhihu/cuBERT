#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>

#include "AttentionMask.h"

namespace cuBERT {
    const static float ZERO = 0;
    const static float ONE = 1;

    AttentionMask::AttentionMask(cublasHandle_t handle,
                                 size_t seq_length, size_t num_attention_heads, size_t max_batch_size) {
        this->handle = handle;
        this->seq_length = seq_length;
        this->num_attention_heads = num_attention_heads;

        auto *ones = new float[num_attention_heads * seq_length];
        std::fill_n(ones, num_attention_heads * seq_length, 1.f);
        cudaMalloc(&ones_gpu, sizeof(float) * num_attention_heads * seq_length);
        cudaMemcpy(ones_gpu, ones, sizeof(float) * num_attention_heads * seq_length, cudaMemcpyHostToDevice);
        delete[] ones;

        cudaMalloc(&this->neg_gpu, sizeof(float) * max_batch_size * seq_length);
    }

    AttentionMask::~AttentionMask() {
        cudaFree(neg_gpu);
        cudaFree(ones_gpu);
    }

    void AttentionMask::compute(size_t batch_size, char *in_gpu, float *out_gpu) {
        cudaStream_t stream = nullptr;
        cublasGetStream_v2(handle, &stream);

        _not(in_gpu, neg_gpu, batch_size * seq_length, stream);

        cublasSgemmStridedBatched(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  seq_length, num_attention_heads * seq_length, 1,
                                  &ONE,
                                  neg_gpu, seq_length, seq_length,
                                  ones_gpu, 1, 0,
                                  &ZERO,
                                  out_gpu, seq_length, seq_length * num_attention_heads * seq_length,
                                  batch_size);
    }
}
