//
// Created by 田露 on 2019/1/22.
//

#include <cstring>

#include "Embedding.h"

namespace cuBERT {
    Embedding::Embedding(size_t vocab_size, size_t embedding_size, float *embedding_table) {
        this->vocab_size = vocab_size;
        this->embedding_size = embedding_size;
        this->embedding_table = new float[vocab_size * embedding_size];
        std::memcpy(this->embedding_table, embedding_table, vocab_size * embedding_size * sizeof(float));

        cudaMalloc(&this->embedding_table_gpu, vocab_size * embedding_size * sizeof(float));
        cudaMemcpy(this->embedding_table_gpu, embedding_table, vocab_size * embedding_size * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    Embedding::~Embedding() {
        cudaFree(this->embedding_table_gpu);
        delete[]embedding_table;
    }

    template<typename T>
    void Embedding::compute(const T *input_ids_gpu, size_t input_ids_len, float *output_gpu, cudaStream_t stream) {
        embedding(input_ids_gpu, input_ids_len, embedding_table_gpu, embedding_size, output_gpu, stream);
    }

    void Embedding::compute_cpu(const int *input_ids, size_t input_ids_len, float *output) {
        for (int i = 0; i < input_ids_len; ++i) {
            std::memcpy(output, embedding_table + embedding_size * input_ids[i], embedding_size * sizeof(float));
            output += embedding_size;
        }
    }

    template void
    Embedding::compute(const int *input_ids_gpu, size_t input_ids_len, float *output_gpu, cudaStream_t stream);

    template void
    Embedding::compute(const char *input_ids_gpu, size_t input_ids_len, float *output_gpu, cudaStream_t stream);
}
