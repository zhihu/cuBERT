#ifndef CUBERT_DENSE_H
#define CUBERT_DENSE_H


#include <cstddef>

namespace cuBERT {
/**
 * Input: batch_size * inputs
 * Kernel: inputs * units
 * Bias: units
 * Output: batch_size * units
 *
 * Output = Input @ Kernel + Bias
 */
    class Dense {
    public:
        explicit Dense(void* handle,
                       size_t inputs,
                       size_t units,
                       float *kernel,
                       float *bias,
                       size_t max_batch_size);

        virtual ~Dense();

        void _pre_compute(size_t batch_size, float *output);

        void _in_compute(size_t batch_size, float *input, float *output);

        void compute(size_t batch_size, float *input, float *output);

    private:
        void* handle;

        size_t inputs;
        size_t units;

        // gpu/cpu buffer
        float *kernel;
        float *bias;
    };
}

#endif //CUBERT_DENSE_H
