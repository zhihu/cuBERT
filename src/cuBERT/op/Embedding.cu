#include <cuda_runtime.h>

#include "Embedding.h"

namespace cuBERT {
    template<typename T>
    __global__ void kernel_embedding(const T *__restrict__ input_ids,
                                     const int input_ids_len,
                                     const float *__restrict__ embedding_table,
                                     const int embedding_size,
                                     float *output) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= input_ids_len) {
            return;
        }

        T input_id = __ldg(input_ids + idx);
        memcpy(output + embedding_size * idx,
               embedding_table + embedding_size * input_id,
               embedding_size * sizeof(float));
    }

    template<bool cpu, typename T>
    __host__ void embedding(const T *input_ids,
                            const int input_ids_len,
                            const float *embedding_table,
                            const int embedding_size,
                            float *output,
                            void *stream) {
        const int blocks = (input_ids_len + 127) / 128;
        kernel_embedding<T> << < blocks, 128, 0, (cudaStream_t) stream >> > (input_ids,
                input_ids_len,
                embedding_table,
                embedding_size,
                output);
    }

    template
    __host__ void embedding<false, int>(const int *input_ids,
                                        const int input_ids_len,
                                        const float *embedding_table,
                                        const int embedding_size,
                                        float *output,
                                        void *stream);

    template
    __host__ void embedding<false, char>(const char *input_ids,
                                         const int input_ids_len,
                                         const float *embedding_table,
                                         const int embedding_size,
                                         float *output,
                                         void *stream);
}
