#include <cstring>

#include "cuBERT/common.h"
#include "Embedding.h"

namespace cuBERT {

#ifdef HAVE_MKL
    template<>
    void embedding<int, float>(const int *input_ids,
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

    template<>
    void embedding<int8_t, float>(const int8_t *input_ids,
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
#endif

    template<typename T, typename V>
    Embedding<T, V>::Embedding(size_t vocab_size, size_t embedding_size, V *embedding_table) {
        this->vocab_size = vocab_size;
        this->embedding_size = embedding_size;

        this->embedding_table = static_cast<V *>(cuBERT::malloc(vocab_size * embedding_size * sizeof(V)));
        cuBERT::memcpy(this->embedding_table, embedding_table, vocab_size * embedding_size * sizeof(V), 1);
    }

    template<typename T, typename V>
    Embedding<T, V>::~Embedding() {
        cuBERT::free(this->embedding_table);
    }

    template<typename T, typename V>
    void Embedding<T, V>::compute(const T *input_ids, size_t input_ids_len, V *output, void* stream) {
        embedding<T, V>(input_ids, input_ids_len, embedding_table, embedding_size, output, stream);
    }

    template class Embedding<int, float>;
    template class Embedding<int8_t, float>;
#ifdef HAVE_CUDA
    template class Embedding<int, half>;
    template class Embedding<int8_t, half>;
#endif
}
