#ifndef CUBERT_BERTEMBEDDINGS_H
#define CUBERT_BERTEMBEDDINGS_H


#include <cstddef>
#include <unordered_map>
#include <string>

#include "cuBERT/op/Embedding.h"
#include "cuBERT/op/LayerNorm.h"

namespace cuBERT {

    template <typename T>
    class BertEmbeddings {
    public:
        explicit BertEmbeddings(void* handle,
                                const std::unordered_map<std::string, T *> &var,
                                size_t max_batch_size,
                                size_t vocab_size, size_t type_vocab_size, size_t hidden_size, size_t seq_length);

        virtual ~BertEmbeddings();

        void compute(size_t batch_size, int *input_ids_gpu, char *token_type_ids_gpu, T *out_gpu);

    private:
        void* handle;

        size_t seq_length;
        size_t hidden_size;

        Embedding<int, T> *word_embeddings;
        Embedding<char, T> *token_type_embeddings;
        LayerNorm<T> *layer_norm;

        // gpu buffer
        T *position_embeddings;
        T *ones;
        T *token_type_embeddings_out;
    };
}

#endif //CUBERT_BERTEMBEDDINGS_H
