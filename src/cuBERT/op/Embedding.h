#ifndef CUBERT_EMBEDDING_H
#define CUBERT_EMBEDDING_H


#include <cstddef>

namespace cuBERT {

    template<typename T, typename V>
    void embedding(const T *input_ids,
                   const int input_ids_len,
                   const V *embedding_table,
                   const int embedding_size,
                   V *output,
                   void *stream);

    template<typename T, typename V>
    class Embedding {
    public:
        explicit Embedding(size_t vocab_size, size_t embedding_size, V *embedding_table);

        virtual ~Embedding();

        void compute(const T *input_ids_gpu, size_t input_ids_len, V *output_gpu, void* stream);

    private:
        size_t vocab_size;
        size_t embedding_size;

        V *embedding_table;
    };
}

#endif //CUBERT_EMBEDDING_H
