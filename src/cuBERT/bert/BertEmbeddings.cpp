#include <cstring>
#include <mkl.h>
#include <cuda_runtime.h>
#include <algorithm>

#include "cuBERT/common.h"
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
        this->layer_norm = new LayerNorm(max_batch_size * seq_length, hidden_size,
                                         var.at("bert/embeddings/LayerNorm/beta"),
                                         var.at("bert/embeddings/LayerNorm/gamma"));

        float *full_position_embeddings = var.at("bert/embeddings/position_embeddings");
        this->position_embeddings_cpu = new float[seq_length * hidden_size];
        std::memcpy(position_embeddings_cpu, full_position_embeddings, sizeof(float) * seq_length * hidden_size);
        CUDA_CHECK(cudaMalloc(&this->position_embeddings_gpu, sizeof(float) * seq_length * hidden_size));
        CUDA_CHECK(cudaMemcpy(this->position_embeddings_gpu, full_position_embeddings, sizeof(float) * seq_length * hidden_size, cudaMemcpyHostToDevice));

        this->ones_cpu = new float[max_batch_size];
        std::fill_n(ones_cpu, max_batch_size, 1.f);
        CUDA_CHECK(cudaMalloc(&this->ones_gpu, sizeof(float) * max_batch_size));
        CUDA_CHECK(cudaMemcpy(this->ones_gpu, ones_cpu, sizeof(float) * max_batch_size, cudaMemcpyHostToDevice));

        this->token_type_embeddings_out_cpu = new float[max_batch_size * seq_length * hidden_size];
        CUDA_CHECK(cudaMalloc(&this->token_type_embeddings_out_gpu, sizeof(float) * max_batch_size * seq_length * hidden_size));
    }

    BertEmbeddings::~BertEmbeddings() {
        delete[] token_type_embeddings_out_cpu;
        delete[] ones_cpu;
        delete[] position_embeddings_cpu;

        CUDA_CHECK(cudaFree(this->token_type_embeddings_out_gpu));
        CUDA_CHECK(cudaFree(this->ones_gpu));
        CUDA_CHECK(cudaFree(this->position_embeddings_gpu));

        delete layer_norm;
        delete token_type_embeddings;
        delete word_embeddings;
    }

    void BertEmbeddings::compute(size_t batch_size, int *input_ids_gpu, char *token_type_ids_gpu, float *out_gpu) {
        cudaStream_t stream = nullptr;
        CUBLAS_CHECK(cublasGetStream_v2(handle, &stream));

        word_embeddings->compute(input_ids_gpu, batch_size * seq_length, out_gpu, stream);
        token_type_embeddings->compute(token_type_ids_gpu, batch_size * seq_length, token_type_embeddings_out_gpu, stream);

        CUBLAS_CHECK(cublasSgemm_v2(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    seq_length * hidden_size, batch_size, 1,
                                    &ONE,
                                    position_embeddings_gpu, seq_length * hidden_size,
                                    ones_gpu, 1,
                                    &ONE,
                                    out_gpu, seq_length * hidden_size));

        layer_norm->compute_(batch_size * seq_length, token_type_embeddings_out_gpu, out_gpu, stream);
    }

    void BertEmbeddings::compute_cpu(size_t batch_size, int *input_ids_cpu, char *token_type_ids_cpu, float *out_cpu) {
        word_embeddings->compute_cpu(input_ids_cpu, batch_size * seq_length, out_cpu);
        token_type_embeddings->compute_cpu(token_type_ids_cpu, batch_size * seq_length, token_type_embeddings_out_cpu);

        cblas_sgemm(CblasColMajor,
                    CblasNoTrans, CblasNoTrans,
                    seq_length * hidden_size, batch_size, 1,
                    ONE,
                    position_embeddings_cpu, seq_length * hidden_size,
                    ones_cpu, 1,
                    ONE,
                    out_cpu, seq_length * hidden_size);

        layer_norm->compute_cpu_(batch_size * seq_length, token_type_embeddings_out_cpu, out_cpu);
    }
}
