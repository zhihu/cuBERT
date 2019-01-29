//
// Created by 田露 on 2019/1/22.
//

#ifndef CUBERT_EMBEDDING_H
#define CUBERT_EMBEDDING_H


#include <cstddef>
#include <cuda_runtime.h>

namespace cuBERT {
    template<typename T>
    __host__ void embedding(const T *input_ids,
                            const int input_ids_len,
                            const float *embedding_table,
                            const int embedding_size,
                            float *output,
                            cudaStream_t stream);

    class Embedding {
    public:
        explicit Embedding(size_t vocab_size, size_t embedding_size, float *embedding_table);

        virtual ~Embedding();

        template<typename T>
        void compute(const T *input_ids_gpu, size_t input_ids_len, float *output_gpu, cudaStream_t stream);

        template<typename T>
        void compute_cpu(const T *input_ids, size_t input_ids_len, float *output);

    private:
        size_t vocab_size;
        size_t embedding_size;

        float *embedding_table;
        float *embedding_table_gpu;
    };
}

#endif //CUBERT_EMBEDDING_H
