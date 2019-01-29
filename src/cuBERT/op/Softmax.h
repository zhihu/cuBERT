#ifndef CUBERT_SOFTMAX_H
#define CUBERT_SOFTMAX_H


#include <cuda_runtime.h>

namespace cuBERT {

    __host__ void softmax_(float *inout,
                           const int batch_size,
                           const int channel,
                           float *sum_gpu,
                           cudaStream_t stream);

/**
 * Performance is expected to be highest with NCHW fully-packed tensors.
 */
    class Softmax {
    public:
        explicit Softmax(size_t max_batch_size, size_t channel);

        virtual ~Softmax();

        void compute_(size_t batch_size, float *inout_gpu, cudaStream_t stream);

        void compute_cpu_(size_t batch_size, float *inout_cpu);

    private:
        size_t channel;

        float* sum_gpu;
    };
}

#endif //CUBERT_SOFTMAX_H
