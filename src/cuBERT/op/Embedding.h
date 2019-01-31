#ifndef CUBERT_EMBEDDING_H
#define CUBERT_EMBEDDING_H


#include <cstddef>

namespace cuBERT {

#ifdef HAVE_CUDA
    template<typename T>
    void embedding(const T *input_ids,
                   const int input_ids_len,
                   const float *embedding_table,
                   const int embedding_size,
                   float *output,
                   void *stream);
#endif

    class Embedding {
    public:
        explicit Embedding(size_t vocab_size, size_t embedding_size, float *embedding_table);

        virtual ~Embedding();

        template<typename T>
        void compute(const T *input_ids_gpu, size_t input_ids_len, float *output_gpu, void* stream);

    private:
        size_t vocab_size;
        size_t embedding_size;

        float *embedding_table;
    };
}

#endif //CUBERT_EMBEDDING_H
