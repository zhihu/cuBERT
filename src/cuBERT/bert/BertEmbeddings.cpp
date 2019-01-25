//
// Created by 田露 on 2019/1/22.
//
#include <cuda_runtime.h>
#include <algorithm>

#include "BertEmbeddings.h"

namespace cuBERT {
    const static float ONE = 1;

    BertEmbeddings::BertEmbeddings(cublasHandle_t handle,
                                   const std::unordered_map<std::string, float *> &var,
                                   size_t max_batch_size,
                                   size_t vocab_size, size_t type_vocab_size, size_t hidden_size, size_t seq_length) {
        this->handle = handle;

        this->seq_length = seq_length;
        this->hidden_size = hidden_size;

        this->word_embeddings = new Embedding(vocab_size, hidden_size,
                                              var.at("bert/embeddings/word_embeddings"));
        this->token_type_embeddings = new Embedding(type_vocab_size, hidden_size,
                                                    var.at("bert/embeddings/token_type_embeddings"));
        this->layer_norm = new LayerNorm(hidden_size,
                                         var.at("bert/embeddings/LayerNorm/beta"),
                                         var.at("bert/embeddings/LayerNorm/gamma"));

        float *full_position_embeddings = var.at("bert/embeddings/position_embeddings");
        cudaMalloc(&this->position_embeddings_gpu, sizeof(float) * seq_length * hidden_size);
        cudaMemcpy(this->position_embeddings_gpu, full_position_embeddings, sizeof(float) * seq_length * hidden_size,
                   cudaMemcpyHostToDevice);

        auto *ones = new float[max_batch_size];
        std::fill_n(ones, max_batch_size, 1.f);
        cudaMalloc(&this->ones_gpu, sizeof(float) * max_batch_size);
        cudaMemcpy(this->ones_gpu, ones, sizeof(float) * max_batch_size, cudaMemcpyHostToDevice);
        delete[] ones;

        cudaMalloc(&this->token_type_embeddings_out_gpu, sizeof(float) * max_batch_size * seq_length * hidden_size);
    }

    BertEmbeddings::~BertEmbeddings() {
        cudaFree(this->token_type_embeddings_out_gpu);
        cudaFree(this->ones_gpu);
        cudaFree(this->position_embeddings_gpu);

        delete layer_norm;
        delete token_type_embeddings;
        delete word_embeddings;
    }

    void BertEmbeddings::compute(size_t batch_size, int *input_ids_gpu, char *token_type_ids_gpu, float *out_gpu) {
        cudaStream_t stream = nullptr;
        cublasGetStream_v2(handle, &stream);

        word_embeddings->compute(input_ids_gpu, batch_size * seq_length, out_gpu, stream);
        token_type_embeddings->compute(token_type_ids_gpu, batch_size * seq_length, token_type_embeddings_out_gpu,
                                       stream);

        cublasSgemm_v2(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       seq_length * hidden_size, batch_size, 1,
                       &ONE,
                       position_embeddings_gpu, seq_length * hidden_size,
                       ones_gpu, 1,
                       &ONE,
                       out_gpu, seq_length * hidden_size);

        layer_norm->compute_(batch_size * seq_length, token_type_embeddings_out_gpu, out_gpu, stream);
    }
}
