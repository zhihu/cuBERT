#ifndef CUBERT_DENSE_H
#define CUBERT_DENSE_H


#include <cstddef>
#include <cublas_v2.h>

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
        explicit Dense(cublasHandle_t handle,
                       size_t inputs,
                       size_t units,
                       float *kernel,
                       float *bias,
                       size_t max_batch_size);

        virtual ~Dense();

        void _pre_compute(size_t batch_size, float *output_gpu);

        void _in_compute(size_t batch_size, float *input_gpu, float *output_gpu);

        void compute(size_t batch_size, float *input_gpu, float *output_gpu);

        void compute_cpu(size_t batch_size, float *input_cpu, float *output_cpu);

    private:
        cublasHandle_t handle;

        size_t inputs;
        size_t units;

        float *kernel_gpu;
        float *bias_gpu;
    };
}

#endif //CUBERT_DENSE_H
