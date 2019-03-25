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
    template<typename T>
    class Dense {
    public:
        explicit Dense(void* handle,
                       size_t inputs,
                       size_t units,
                       T *kernel,
                       T *bias,
                       size_t max_batch_size);

        virtual ~Dense();

        void _pre_compute(size_t batch_size, T *output);

        void _in_compute(size_t batch_size, T *input, T *output);

        void compute(size_t batch_size, T *input, T *output);

    private:
        void* handle;

        size_t inputs;
        size_t units;

        // gpu/cpu buffer
        T *kernel;
        T *bias;
    };
}

#endif //CUBERT_DENSE_H
