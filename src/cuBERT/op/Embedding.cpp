#include <cstring>

#include "cuBERT/common.h"
#include "Embedding.h"

namespace cuBERT {

    template <>
    void embedding<true, int>(const int *input_ids,
                              const int input_ids_len,
                              const float *embedding_table,
                              const int embedding_size,
                              float *output,
                              void *stream) {
        for (int i = 0; i < input_ids_len; ++i) {
            std::memcpy(output, embedding_table + embedding_size * input_ids[i], embedding_size * sizeof(float));
            output += embedding_size;
        }
    }

    template <>
    void embedding<true, char>(const char *input_ids,
                               const int input_ids_len,
                               const float *embedding_table,
                               const int embedding_size,
                               float *output,
                               void *stream) {
        for (int i = 0; i < input_ids_len; ++i) {
            std::memcpy(output, embedding_table + embedding_size * input_ids[i], embedding_size * sizeof(float));
            output += embedding_size;
        }
    }

    Embedding::Embedding(size_t vocab_size, size_t embedding_size, float *embedding_table) {
        this->vocab_size = vocab_size;
        this->embedding_size = embedding_size;

        this->embedding_table = static_cast<float *>(cuBERT::malloc(vocab_size * embedding_size * sizeof(float)));
        cuBERT::memcpy(this->embedding_table, embedding_table, vocab_size * embedding_size * sizeof(float), 1);
    }

    Embedding::~Embedding() {
        cuBERT::free(this->embedding_table);
    }

    template<typename T>
    void Embedding::compute(const T *input_ids, size_t input_ids_len, float *output, void* stream) {
        embedding<!cuBERT::gpu()>(input_ids, input_ids_len, embedding_table, embedding_size, output, stream);
    }

    template void
    Embedding::compute(const int *input_ids_gpu, size_t input_ids_len, float *output_gpu, void* stream);

    template void
    Embedding::compute(const char *input_ids_gpu, size_t input_ids_len, float *output_gpu, void* stream);
}
