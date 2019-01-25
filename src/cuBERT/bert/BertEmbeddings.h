//
// Created by 田露 on 2019/1/22.
//

#ifndef CUBERT_BERTEMBEDDINGS_H
#define CUBERT_BERTEMBEDDINGS_H


#include <cstddef>
#include <unordered_map>
#include <string>
#include <cublas_v2.h>

#include "cuBERT/op/Embedding.h"
#include "cuBERT/op/LayerNorm.h"

namespace cuBERT {
    class BertEmbeddings {
    public:
        explicit BertEmbeddings(cublasHandle_t handle,
                                const std::unordered_map<std::string, float *> &var,
                                size_t max_batch_size,
                                size_t vocab_size, size_t type_vocab_size, size_t hidden_size, size_t seq_length);

        virtual ~BertEmbeddings();

        void compute(size_t batch_size, int *input_ids_gpu, char *token_type_ids_gpu, float *out_gpu);

    private:
        cublasHandle_t handle;

        size_t seq_length;
        size_t hidden_size;

        Embedding *word_embeddings;
        Embedding *token_type_embeddings;
        LayerNorm *layer_norm;

        float *position_embeddings_gpu;
        float *ones_gpu;
        float *token_type_embeddings_out_gpu;
    };
}

#endif //CUBERT_BERTEMBEDDINGS_H
