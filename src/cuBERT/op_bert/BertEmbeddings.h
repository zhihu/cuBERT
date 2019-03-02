#ifndef CUBERT_BERTEMBEDDINGS_H
#define CUBERT_BERTEMBEDDINGS_H


#include <cstddef>
#include <unordered_map>
#include <string>

#include "cuBERT/op/Embedding.h"
#include "cuBERT/op/LayerNorm.h"

namespace cuBERT {
    class BertEmbeddings {
    public:
        explicit BertEmbeddings(void* handle,
                                const std::unordered_map<std::string, float *> &var,
                                size_t max_batch_size,
                                size_t vocab_size, size_t type_vocab_size, size_t hidden_size, size_t seq_length);

        virtual ~BertEmbeddings();

        void compute(size_t batch_size, int *input_ids_gpu, char *token_type_ids_gpu, float *out_gpu);

    private:
        void* handle;

        size_t seq_length;
        size_t hidden_size;

        Embedding *word_embeddings;
        Embedding *token_type_embeddings;
        LayerNorm *layer_norm;

        // gpu buffer
        float *position_embeddings;
        float *ones;
        float *token_type_embeddings_out;
    };
}

#endif //CUBERT_BERTEMBEDDINGS_H
