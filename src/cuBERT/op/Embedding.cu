#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "Embedding.h"

namespace cuBERT {
    template<typename T, typename V>
    __global__ void kernel_embedding(const T *__restrict__ input_ids,
                                     const int input_ids_len,
                                     const V *__restrict__ embedding_table,
                                     const int embedding_size,
                                     V *output) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= input_ids_len) {
            return;
        }

#if __CUDA_ARCH__ >= 350
        T input_id = __ldg(input_ids + idx);
#else
        T input_id = input_ids[idx];
#endif
        memcpy(output + embedding_size * idx,
               embedding_table + embedding_size * input_id,
               embedding_size * sizeof(V));
    }

    template<typename T, typename V>
    __host__ void embedding(const T *input_ids,
                            const int input_ids_len,
                            const V *embedding_table,
                            const int embedding_size,
                            V *output,
                            void *stream) {
        const int blocks = (input_ids_len + 127) / 128;
        kernel_embedding<T, V> << < blocks, 128, 0, (cudaStream_t) stream >> > (input_ids,
                input_ids_len,
                embedding_table,
                embedding_size,
                output);
    }

    template
    __host__ void embedding<int, float>(const int *input_ids,
                                        const int input_ids_len,
                                        const float *embedding_table,
                                        const int embedding_size,
                                        float *output,
                                        void *stream);

    template
    __host__ void embedding<int8_t, float>(const int8_t *input_ids,
                                           const int input_ids_len,
                                           const float *embedding_table,
                                           const int embedding_size,
                                           float *output,
                                           void *stream);

    template
    __host__ void embedding<int, half>(const int *input_ids,
                                       const int input_ids_len,
                                       const half *embedding_table,
                                       const int embedding_size,
                                       half *output,
                                       void *stream);

    template
    __host__ void embedding<int8_t, half>(const int8_t *input_ids,
                                          const int input_ids_len,
                                          const half *embedding_table,
                                          const int embedding_size,
                                          half *output,
                                          void *stream);
}
